# Paused continuation text — volume 4

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 9988; SHA256 `cfe6f7d1688a33251450bcfe075f166642a34e3c997d3e5b3a48fb8f0807c2f6`.

````text
use super::*;
fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
    InitialPreludeContract::new(
        SourceIdentity::for_new_allocation().unwrap(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        GdnPreludePhase::InitialChunk64,
        operation,
        heads,
    )
    .unwrap()
}
#[test]
fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
    let mut input = vec![0.0; c.elements()];
    input[0] = 1.0;
    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
    input[2] = -1.0;
    input[64] = -0.0;
    input[128] = -1.0;
    let mut output = vec![9.0; input.len()];
    initial_prelude_into(c, &input, &mut output, None).unwrap();
    assert_eq!(output[0].to_bits(), 0x3f80_0000);
    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
    assert_eq!(output[2].to_bits(), 0x3380_0000);
    assert_eq!(output[63].to_bits(), 0x3380_0000);
    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
    assert_eq!(output[127].to_bits(), 0);
    assert_eq!(output[128].to_bits(), 0xbf80_0000);
    assert_eq!(output[191].to_bits(), 0xbf80_0000);
}
#[test]
fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
    let c = contract(InitialPreludeOperation::QueryScale128, 48);
    let mut input = vec![1.0; c.elements()];
    input[..6].copy_from_slice(&[
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
    ]);
    let mut out = vec![0.0; input.len()];
    initial_prelude_into(c, &input, &mut out, None).unwrap();
    assert_eq!(
        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
    );
    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
}
#[test]
fn cached_profile_source_geometry_and_complete_span_refuse() {
    for (source, profile, phase, heads) in [
        (
            SourceIdentity::unbound(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::GgufSourceV1,
            GdnPreludePhase::InitialChunk64,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::Cached,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            32,
        ),
    ] {
        assert!(
            InitialPreludeContract::new(
                source,
                profile,
                phase,
                InitialPreludeOperation::DecayPrefix64,
                heads
            )
            .is_err()
        );
    }
    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
    let mut input = vec![0.0; c.elements()];
    let mut out = vec![7.0; c.elements()];
    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
    input[0] = f32::NAN;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
    input[0] = f32::INFINITY;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
    assert!(out.iter().all(|v| *v == 7.0));
    input[0] = 0.0;
    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
    assert!(out.iter().all(|v| *v == 7.0));
    input[0] = f32::MAX;
    input[1] = f32::MAX;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod physical {
    use super::*;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU64;
    use zerogravity_artifacts::artifact::Artifact;
    use zerogravity_artifacts::cache::TransferAdmission;
    use zerogravity_artifacts::source_weights::{
        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
    };
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const META: u64 = 16 * 1024 * 1024;
    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let p = std::env::temp_dir().join(format!(
                "zg-initial-prefix-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&p).unwrap();
            fs::write(p.join("model.safetensors"), BYTES).unwrap();
            Self(p)
        }
        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
            SourceWeights::prepare_verified_windows(
                INDEX,
                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
                vec![
                    ExpectedSourceShard::new(
                        "model.safetensors".into(),
                        Artifact::from_file(
                            File::open(self.0.join("model.safetensors")).unwrap(),
                            Path::new("model.safetensors"),
                        )
                        .unwrap(),
                        BYTES.len() as u64,
                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
                    )
                    .unwrap(),
                ],
                WindowOpenLimits {
                    max_source_bytes: 1024 * 1024,
                    max_metadata_bytes: META,
                    max_shards: 1,
                    max_tensors: 6,
                    scan_chunk_bytes: 1024,
                },
                h,
            )
            .unwrap()
            .open()
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn host() -> TransferAdmission {
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
        arbiter
            .register(resource.clone(), 64 * 1024 * 1024)
            .unwrap();
        TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
        }
    }
    fn reserved(h: &TransferAdmission) -> u64 {
        h.arbiter.snapshot(&h.resource).unwrap().reserved
    }
    fn raw(c: InitialPreludeContract) -> Vec<u8> {
        1.0f32.to_le_bytes().repeat(c.elements())
    }
    #[test]
    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
        let f = Fixture::new();
        let h = host();
        let source = f.source(&h);
        let foreign = f.source(&h);
        let c = InitialPreludeContract::new(
            source.source_identity(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            InitialPreludeOperation::QueryScale128,
            48,
        )
        .unwrap();
        let raw = raw(c);
        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
        let before = reserved(&h);
        assert!(execute_initial_prelude(&foreign, input, None).is_err());
        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
        assert_eq!(reserved(&h), before);
        let need = InitialPreludeOutput::required_bytes(c) as u64;
        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
        let pressure = source.reserve_workspace(hold as usize).unwrap();
        let pressured = reserved(&h);
        assert!(execute_initial_prelude(&source, input, None).is_err());
        assert_eq!(reserved(&h), pressured);
        drop(pressure);
        assert_eq!(reserved(&h), before);
        let output = execute_initial_prelude(&source, input, None).unwrap();
        let clone = output.clone();
        assert!(output.same_issuer(&source));
        assert!(!output.same_issuer(&foreign));
        assert_eq!(reserved(&h), before + need);
        assert_eq!(output.charged_bytes() as u64, need);
        drop(foreign);
        drop(source);
        assert_eq!(reserved(&h), META + need);
        assert_eq!(clone.imported_input().len(), 48 * 128);
        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
        drop(output);
        assert_eq!(reserved(&h), META + need);
        drop(clone);
        assert_eq!(reserved(&h), 0);
    }
    #[test]
    fn completed_result_cancel_drops_real_admitted_backing() {
        let f = Fixture::new();
        let h = host();
        let source = f.source(&h);
        let c = InitialPreludeContract::new(
            source.source_identity(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            InitialPreludeOperation::DecayPrefix64,
            16,
        )
        .unwrap();
        let raw = raw(c);
        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
        let value = execute_initial_prelude(&source, input, None).unwrap();
        assert!(reserved(&h) > META);
        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
        assert_eq!(reserved(&h), META);
        drop(source);
        assert_eq!(reserved(&h), 0);
    }
}

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5561; SHA256 `2e8cf09b4c8cdfc1b6130abb3a7da9dea0ac3fb96a3d6ff27bac02db64c7b425`.

````text
//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Phase is explicit; cached query division is deliberately unsupported here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnPreludePhase {
    InitialChunk64,
    Cached,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialPreludeOperation {
    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
    QueryScale128,
    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
    DecayPrefix64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitialPreludeContract {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    operation: InitialPreludeOperation,
    heads: usize,
}
impl InitialPreludeContract {
    pub fn new(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        phase: GdnPreludePhase,
        operation: InitialPreludeOperation,
        heads: usize,
    ) -> Result<Self> {
        if phase != GdnPreludePhase::InitialChunk64
            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
        }
        if !source.is_bound() || !matches!(heads, 16 | 48) {
            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
        }
        Ok(Self {
            source,
            profile,
            operation,
            heads,
        })
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn phase(self) -> GdnPreludePhase {
        GdnPreludePhase::InitialChunk64
    }
    pub fn operation(self) -> InitialPreludeOperation {
        self.operation
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn elements(self) -> usize {
        self.heads
            * match self.operation {
                InitialPreludeOperation::QueryScale128 => 128,
                InitialPreludeOperation::DecayPrefix64 => 64,
            }
    }
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("initial GDN leaf cancelled".into()));
    }
    Ok(())
}
fn environment() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut mxcsr = 0u32;
        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
        // Passive read only; exception status bits may change during arithmetic.
        unsafe {
            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
                options(nostack, preserves_flags));
        }
        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
            return Err(Error::Unsupported(
                "initial GDN pinned CPU FP controls".into(),
            ));
        }
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(Error::Unsupported(
        "initial GDN pinned CPU architecture".into(),
    ))
}

/// No allocation/provider callback. Caller owns both complete, disjoint slices.
/// Output remains private on error; successful completion grants no state effect.
pub fn initial_prelude_into(
    contract: InitialPreludeContract,
    input: &[f32],
    output: &mut [f32],
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    environment()?;
    if input.len() != contract.elements()
        || output.len() != input.len()
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Malformed(
            "initial GDN complete finite FP32 span".into(),
        ));
    }
    match contract.operation {
        InitialPreludeOperation::QueryScale128 => {
            // Source query * (128 ** -0.5), scalar converted once to FP32.
            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
                not_cancelled(cancelled)?;
                for (&value, result) in row.iter().zip(target) {
                    *result = value * SCALE;
                }
            }
        }
        InitialPreludeOperation::DecayPrefix64 => {
            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
                not_cancelled(cancelled)?;
                let mut carry = 0.0f64;
                for (&value, result) in row.iter().zip(target) {
                    carry += f64::from(value);
                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
                    if !result.is_finite() {
                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
                    }
                }
            }
        }
    }
    not_cancelled(cancelled)
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod admitted;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
#[cfg(test)]
mod tests;

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/proposed/crates/zerogravity-cpu/src/original_hf.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11227; SHA256 `da11a4a82664c93797ed8dba0847302f364273c9ad546446f6e8fb1d5a6b6f5a`.

````text
//! Original Hugging Face output-first source rows using the existing CPU math.
//! Torch linear matrices are [output,input] in row-major memory. No whole
//! learned matrix transpose or duplicate mathematical runtime is required.
//! This does not enable complete trained Qwen3.8 inference.

pub mod coefficient_preparation;
pub mod initial_prelude;
pub mod conv;
pub mod norm;
pub mod recurrent;
pub mod reduction;
pub mod transcendentals;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_embedding;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_projection;

use crate::accelerator::{MatVecKernel, MatVecRegion, RegionExecution, validate_completed};
use crate::quant::QuantKind;
use crate::weights::{
    MAX_MATRIX_ROWS, SourceProjectionProfile, source_reference_projection_with_profile,
};
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Project one already source-qualified original [output,input] tensor
/// through the same real CPU kernel and optional physical provider.
pub fn output_first_projection(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    output_first_projection_with_profile(
        source_id,
        tensor_name,
        tensor,
        input,
        provider,
        SourceProjectionProfile::Source,
    )
}

/// Closed CPU numerical profile; an unaware physical provider must refuse.
pub fn output_first_projection_with_profile(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
    profile: SourceProjectionProfile,
) -> Result<Vec<f32>> {
    tensor.validate()?;
    if !source_id.is_bound()
        || tensor_name.is_empty()
        || tensor.shape.len() != 2
        || tensor.shape[1] != input.len() as u64
        || tensor.shape[0] > MAX_MATRIX_ROWS
        || input.is_empty()
        || !input.iter().all(|x| x.is_finite())
        || !matches!(
            tensor.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        )
    {
        return Err(Error::Malformed(
            "original output-first source GEMV geometry/encoding invalid".into(),
        ));
    }
    let rows = usize::try_from(tensor.shape[0])
        .map_err(|_| Error::Limit("output-first matrix rows exceed address space".into()))?;
    let columns = usize::try_from(tensor.shape[1])
        .map_err(|_| Error::Limit("output-first matrix columns exceed address space".into()))?;
    let region = MatVecRegion {
        source_identity: source_id,
        tensor_name,
        kind: tensor.kind,
        columns,
        rows,
        packed_weights: tensor.packed_bytes,
        activation: input,
    };
    if let Some(provider) = provider {
        if profile != SourceProjectionProfile::Source {
            return Err(Error::Unsupported(
                "physical provider does not declare the CPU projection profile".into(),
            ));
        }
        match provider.try_execute(region)? {
            RegionExecution::Unsupported => {}
            RegionExecution::Completed(values) => {
                validate_completed(region, &values)?;
                return Ok(values);
            }
        }
    }
    source_reference_projection_with_profile(region, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, RegionExecution, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct UnawareProvider(AtomicBool);
    impl MatVecKernel for UnawareProvider {
        fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Completed(vec![77.0; region.rows]))
        }
    }
    #[test]
    fn explicit_cpu_profile_refuses_unaware_provider_without_invoking_it() {
        let provider = UnawareProvider(AtomicBool::new(false));
        let packed = 0x3f80u16.to_le_bytes();
        let tensor = SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[1, 1],
            packed_bytes: &packed,
        };
        let source = source_id();
        assert!(matches!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                Some(&provider),
                SourceProjectionProfile::TorchCpuBf16Avx2,
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(!provider.0.load(Ordering::Relaxed));
        assert_eq!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                None,
                SourceProjectionProfile::TorchCpuBf16Avx2,
            )
            .unwrap(),
            [1.0]
        );
        assert_eq!(
            output_first_projection(source, "synthetic.weight", tensor, &[1.0], Some(&provider))
                .unwrap(),
            [77.0]
        );
        assert!(provider.0.load(Ordering::Relaxed));
    }

    fn source_id() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn projected(kind: QuantKind, shape: &[u64], bytes: &[u8], input: &[f32]) -> Result<Vec<f32>> {
        output_first_projection(
            source_id(),
            "original.linear.weight",
            SourceTensorRef {
                kind,
                shape,
                packed_bytes: bytes,
            },
            input,
            None,
        )
    }
    #[test]
    fn original_f32_hf_output_first_uses_same_reference_projection_without_transposition() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        assert_eq!(
            projected(QuantKind::F32, &[2, 3], &bytes, &[7., 8., 9.]).unwrap(),
            [50., 122.]
        );
        assert!(projected(QuantKind::F32, &[3, 2], &bytes, &[7., 8., 9.]).is_err());
    }
    #[test]
    fn original_bf16_and_f16_rows_reuse_existing_raw_source_math() {
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x4000, 0x4040, 0x4080] {
            bf16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::Bf16, &[2, 2], &bf16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        let mut f16 = Vec::new();
        for bits in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            f16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::F16, &[2, 2], &f16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
    }
    struct UnsupportedProvider;
    impl MatVecKernel for UnsupportedProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
    }
    struct BadProvider;
    impl MatVecKernel for BadProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(vec![f32::NAN]))
        }
    }
    #[test]
    fn optional_physical_unsupported_allows_cpu_but_invalid_completion_refuses() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        let tensor = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2, 2],
            packed_bytes: &bytes,
        };
        let normal =
            output_first_projection(source_id(), "q.weight", tensor, &[5., 6.], None).unwrap();
        let fallback = output_first_projection(
            source_id(),
            "q.weight",
            tensor,
            &[5., 6.],
            Some(&UnsupportedProvider),
        )
        .unwrap();
        assert_eq!(normal, fallback);
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                tensor,
                &[5., 6.],
                Some(&BadProvider)
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                SourceIdentity::unbound(),
                "q.weight",
                tensor,
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(source_id(), "q.weight", tensor, &[f32::NAN, 6.], None)
                .is_err()
        );
        assert!(output_first_projection(source_id(), "q.weight", tensor, &[1.], None).is_err());
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    kind: QuantKind::Q4_0,
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    shape: &[2, 3],
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        // Original source bound MUST apply before a physical provider
        // is ever permitted to override the CPU's matrix safety cap.
        let large = vec![0u8; 2 * 1_000_001];
        let huge = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &large,
        };
        assert!(
            output_first_projection(
                source_id(),
                "oversized.hf.weight",
                huge,
                &[1.0],
                Some(&UnsupportedProvider)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ObservedProvider<'a>(&'a AtomicBool);
    impl MatVecKernel for ObservedProvider<'_> {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }

    #[test]
    fn oversize_hf_output_rows_refuse_before_accelerator_invocation() {
        let called = AtomicBool::new(false);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let bytes = vec![0u8; 2 * 1_000_001];
        let tensor = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &bytes,
        };
        let result = output_first_projection(
            source,
            "original.large.weight",
            tensor,
            &[1.0],
            Some(&ObservedProvider(&called)),
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
        assert!(!called.load(Ordering::Relaxed));
    }
}

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 12695; SHA256 `72c12e1036956930eec8de9456b1900b21c4421bc652473b7ffa8f7e8aaedcfd`.

````text
//! Sealed synthetic H48 source/metadata/lease checks; no trained math claim.
use super::*;
use crate::accelerator::program::{ProjectionContract, SourceParameter, SourceTensor};
use crate::operator_slice::OperatorSliceSource;
use zerogravity_core::quant::QuantKind;

fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    let text = &mut cfg["text_config"];
    text["num_hidden_layers"] = json!(2);
    text["full_attention_interval"] = json!(2);
    text["layer_types"] = json!(["linear_attention", "full_attention"]);
    text["hidden_size"] = json!(256);
    text["linear_num_key_heads"] = json!(16);
    text["linear_num_value_heads"] = json!(48);
    text["linear_key_head_dim"] = json!(128);
    text["linear_value_head_dim"] = json!(128);
    let cfg = serde_json::to_vec(&cfg).unwrap();
    let mut header = BTreeMap::new();
    let mut map = BTreeMap::new();
    let mut payload = Vec::new();
    for spec in QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap()
    {
        let begin = payload.len();
        let value = if spec.name.ends_with("dt_bias") {
            0
        } else {
            0x3e80u16
        };
        for _ in 0..spec.shape.iter().product::<u64>() {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            spec.name.clone(),
            json!({"dtype":"BF16", "shape":spec.shape,
            "data_offsets":[begin,payload.len()]}),
        );
        map.insert(spec.name, "model.safetensors");
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&payload);
    assert!(bytes.len() < 16 * 1024 * 1024);
    let path = fixture.0.join("model.safetensors");
    fs::write(&path, &bytes).unwrap();
    let index = serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},
        "weight_map":map}))
    .unwrap();
    let index_sha = digest(&index);
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "coefficient-h48-source-test").unwrap();
    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("synthetic", "coefficient-h48").unwrap(),
    };
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let source = SourceWeights::prepare_verified_windows(
        &index,
        &index_sha,
        vec![
            ExpectedSourceShard::new(
                "model.safetensors".into(),
                artifact,
                bytes.len() as u64,
                digest(&bytes),
            )
            .unwrap(),
        ],
        WindowOpenLimits {
            max_source_bytes: 16 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
            scan_chunk_bytes: 64 * 1024,
        },
        &host,
    )
    .unwrap()
    .open()
    .unwrap();
    (source, host, cfg, index_sha)
}

struct MetadataFault<'a> {
    source: &'a OriginalWindowOperators,
    a_log: SourceParameter,
    profile: ProgramNumericalProfile,
}
impl OperatorSliceSource for MetadataFault<'_> {
    fn geometry(&self) -> &crate::model::ModelGeometry {
        self.source.geometry()
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source.source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    fn rms_epsilon(&self) -> f32 {
        self.source.rms_epsilon()
    }
    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        self.source.matrix_descriptor(role)
    }
    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
        if role == QwenParameterRole::GdnA(0) {
            Ok(self.a_log.clone())
        } else {
            self.source.parameter_descriptor(role)
        }
    }
    fn embedding_contract(&self) -> EmbeddingContract {
        self.source.embedding_contract()
    }
    fn projection_contract(&self) -> ProjectionContract {
        self.source.projection_contract()
    }
}

#[test]
fn only_genuine_h48_bf16_unevaluated_a_log_extent_is_admitted() {
    use crate::operator_slice::coefficient_preparation::coefficient_roles;
    let old = Fixture::new();
    let loaded = coefficient_source(&old);
    let (source, host) = windows(&old, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let roles = coefficient_roles(&ops, 0).unwrap();
    assert_eq!(
        (roles.heads, roles.a_log.kind, roles.a_log.bytes),
        (16, QuantKind::F32, 64)
    );
    let mut unsupported = roles.a_log;
    unsupported.kind = QuantKind::Bf16;
    unsupported.bytes = 32;
    assert!(
        coefficient_roles(
            &MetadataFault {
                source: &ops,
                a_log: unsupported,
                profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            },
            0
        )
        .is_err()
    );
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);

    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let roles = coefficient_roles(&ops, 0).unwrap();
    assert_eq!(
        (
            roles.heads,
            roles.a_log.kind,
            roles.a_log.shape.clone(),
            roles.a_log.bytes
        ),
        (48, QuantKind::Bf16, vec![48], 96)
    );
    assert_eq!(
        (roles.dt.kind, roles.dt.shape, roles.dt.bytes),
        (QuantKind::Bf16, vec![48], 96)
    );
    let before = reserved(&host);
    for (kind, bytes, shape, foreign, name, profile) in [
        (
            QuantKind::F16,
            96,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            192,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::F32,
            96,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![47],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![48],
            true,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![48],
            false,
            "foreign.A_log".into(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::GgufSourceV1,
        ),
    ] {
        let mut parameter = roles.a_log.clone();
        parameter.kind = kind;
        parameter.bytes = bytes;
        parameter.shape = shape;
        parameter.name = name;
        if foreign {
            parameter.source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
        }
        assert!(
            coefficient_roles(
                &MetadataFault {
                    source: &ops,
                    a_log: parameter,
                    profile
                },
                0
            )
            .is_err()
        );
        assert_eq!(reserved(&host), before);
    }
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn h48_preparation_preserves_source_lease_cancel_pressure_and_all_private_words() {
    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let a = 0x41a8u16.to_le_bytes().repeat(48);
    let b = 0u16.to_le_bytes().repeat(48);
    let make = |identity| input(identity, &a, &b);
    let before = reserved(&host);
    for (layer, identity, cancel) in [
        (1, source.source_identity(), None),
        (
            0,
            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
            None,
        ),
        (0, source.source_identity(), Some(AtomicBool::new(true))),
    ] {
        assert!(
            ops.execute_gdn_coefficient_preparation_slice(layer, make(identity), cancel.as_ref())
                .is_err()
        );
        assert_eq!(reserved(&host), before);
    }
    let mut pressure = Vec::new();
    loop {
        let remaining = host.arbiter.snapshot(&host.resource).unwrap().remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let pressured = reserved(&host);
    assert!(
        ops.execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
            .is_err()
    );
    assert_eq!(reserved(&host), pressured);
    drop(pressure);
    assert_eq!(reserved(&host), before);
    let output = ops
        .execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
        .unwrap();
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 48]);
    assert_eq!(output.biased_alpha_f32().unwrap(), &[21.0; 48]);
    assert_eq!(output.softplus_f32().unwrap(), &[21.0; 48]);
    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
    assert_eq!(
        output.unevaluated_raw_a_log_parameter().kind,
        QuantKind::Bf16
    );
    assert_eq!(reserved(&host), before + 2 * 1024 * 1024);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 34 * 1024 * 1024);
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
    drop(output);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn h48_recording_adds_only_existing_two_preparation_nodes() {
    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    // Same-builder views exercise transport; no authentic projection is claimed.
    let embed = builder
        .embed_profiled(
            ops.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let a = builder.view(&embed, TensorView::vector(0, 48)).unwrap();
    let b = builder.view(&embed, TensorView::vector(48, 48)).unwrap();
    let values = ops
        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
        .unwrap();
    let plan = builder.finish_operator_slice(values.softplus).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(
        plan.plan()
            .nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TypedGdnAlphaPreparation { .. }
                    | PlanOperation::TypedGdnBetaPreparation { .. }
            ))
            .count(),
        2
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);
}

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8949; SHA256 `0410ae2727f6ba7248b830412a0ba243edbc4dfd979d75141a6560834609f80e`.

````text
//! Synthetic source/transport/lease guards only; never trained coefficients.
use super::*;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, PlanBuilder, PlanOperation, ProgramNumericalProfile,
};
use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
fn coefficient_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_num_value_heads"] = json!(16);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, fallback| {
            if name.ends_with(".dt_bias") {
                0
            } else {
                fallback
            }
        },
        |_| {},
    )
}
fn raw(value: u16) -> Vec<u8> {
    [value.to_le_bytes(); 16].concat()
}
fn input<'a>(
    source: crate::accelerator::SourceIdentity,
    a: &'a [u8],
    b: &'a [u8],
) -> GdnCoefficientPreparationInput<'a> {
    GdnCoefficientPreparationInput::from_bf16_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        a,
        b,
    )
    .unwrap()
}
#[test]
fn synthetic_coefficient_stages_retain_two_aggregate_backings_and_source() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let a = raw(0x41a8);
    let b = raw(0); // a21, b0: public linear softplus and sigmoid0.
    let before = reserved(&admission);
    let output = operators
        .execute_gdn_coefficient_preparation_slice(0, input(source.source_identity(), &a, &b), None)
        .unwrap();
    assert!(
        output
            .biased_alpha_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .softplus_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .beta_bf16()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 0.5f32.to_bits())
    );
    assert_eq!(
        output.raw_dt_bias_parameter().kind,
        zerogravity_core::quant::QuantKind::Bf16
    );
    assert_eq!(
        output.unevaluated_raw_a_log_parameter().kind,
        zerogravity_core::quant::QuantKind::F32
    );
    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 34 * 1024 * 1024);
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 16]);
    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 16]);
    drop(output);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn wrong_layer_foreign_input_cancel_and_pressure_refuse_without_leak() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let before = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                3,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(
                    crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
                    &zero,
                    &zero
                ),
                None
            )
            .is_err()
    );
    let cancelled = AtomicBool::new(true);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                Some(&cancelled)
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    let mut pressure = Vec::new();
    loop {
        let remaining = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let pressured = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), pressured);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn raw_a_log_alias_and_unaware_provider_refuse_before_callback_or_output() {
    struct Provider(AtomicBool);
    impl crate::accelerator::MatVecKernel for Provider {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            panic!("unaware coefficient provider invoked");
        }
    }
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let (a, b) = {
        let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        math.import_coefficient_preparation_input(input(source.source_identity(), &zero, &zero), 16)
            .unwrap()
    };
    let before = reserved(&admission);
    let provider = Provider(AtomicBool::new(false));
    {
        let mut math = crate::host_math::HostMath::for_window_operator_slice(
            &operators,
            None,
            Some(&provider),
        );
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnDt(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    {
        let mut math =
            crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnA(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(a);
    drop(b);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn plan_appends_preparation_without_exp_state_token_or_producer_invention() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (model, _, _) = bind(loaded);
    let mut builder =
        PlanBuilder::new_operator_slice(model.source_identity(), model.program_numerical_profile())
            .unwrap();
    // Synthetic same-builder views are transport ONLY, never actual trained a/b.
    let embed = builder
        .embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let a = builder.view(&embed, TensorView::vector(0, 16)).unwrap();
    let b = builder.view(&embed, TensorView::vector(16, 16)).unwrap();
    let values = model
        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
        .unwrap();
    let plan = builder.finish_operator_slice(values.softplus).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(
        plan.plan()
            .nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TypedGdnAlphaPreparation { .. }
                    | PlanOperation::TypedGdnBetaPreparation { .. }
            ))
            .count(),
        2
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, true).is_err());
}

#[path = "coefficient_preparation_h48_tests.rs"]
mod h48;

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/proposed/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10591; SHA256 `8ad73434fc0b29a92e55cfbee8ef193413004683e13ced27ef72fea0b0072528`.

````text
//! Source-bound coefficient PREPARATION on caller-owned a/b; no g/decay/exp.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues, MatrixAxes, PlanBuilder,
    PlanValue, ProgramNumericalProfile, SourceParameter,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};
use zerogravity_core::quant::QuantKind;

pub(crate) struct CoefficientPreparationRoles {
    pub(crate) heads: usize,
    pub(crate) dt: SourceParameter,
    pub(crate) a_log: SourceParameter,
}
/// Immutable complete role descriptors; A_log is not evaluated by this slice.
pub(crate) fn coefficient_roles(
    source: &(impl OperatorSliceSource + ?Sized),
    layer: usize,
) -> Result<CoefficientPreparationRoles> {
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || !source.source_identity().is_bound()
        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "coefficient preparation original LinearAttention source/profile".into(),
        ));
    }
    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
        .map_err(|_| Error::Limit("coefficient heads".into()))?;
    let hidden = usize::try_from(source.geometry().hidden_size)
        .map_err(|_| Error::Limit("coefficient hidden".into()))?;
    if !matches!(heads, 16 | 48) || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
        return Err(Error::Unsupported(
            "coefficient preparation closed source geometry".into(),
        ));
    }
    let namespace = format!("model.language_model.layers.{layer}.linear_attn");
    for (role, suffix) in [
        (QwenMatrixRole::GdnAlpha(layer), "in_proj_a"),
        (QwenMatrixRole::GdnBeta(layer), "in_proj_b"),
    ] {
        let tensor = source.matrix_descriptor(role)?;
        if tensor.source != source.source_identity()
            || tensor.name != format!("{namespace}.{suffix}.weight")
            || tensor.kind != QuantKind::Bf16
            || tensor.axes != MatrixAxes::OutputInput
            || tensor.rows != heads
            || tensor.columns != hidden
        {
            return Err(Error::State(
                "coefficient preparation original a/b full matrix role".into(),
            ));
        }
    }
    let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
    let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
    // A_log is metadata only; no exp/g or learned-parameter cast is granted.
    let a_log_codec_valid = match a_log.kind {
        QuantKind::F32 => a_log.bytes == heads * 4,
        QuantKind::Bf16 => heads == 48 && a_log.bytes == 96,
        _ => false,
    };
    if dt.source != source.source_identity()
        || dt.name != format!("{namespace}.dt_bias")
        || a_log.source != source.source_identity()
        || a_log.name != format!("{namespace}.A_log")
        || a_log.shape != [heads as u64]
        || !a_log_codec_valid
    {
        return Err(Error::State(
            "coefficient preparation original raw A_log/dt_bias roles".into(),
        ));
    }
    GdnCoefficientPreparationContract::TorchCpuBf16Avx2.validate(&dt, heads, heads)?;
    Ok(CoefficientPreparationRoles { heads, dt, a_log })
}

impl LoadedQwenModel {
    /// Append only beta/biased-alpha/softplus to caller's original plan values.
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
}

/// External actual source/operator a/b bytes; caller authenticates provenance.
/// No hidden state, matrix producer, token or frontier is asserted by import.
#[derive(Clone, Copy)]
pub struct GdnCoefficientPreparationInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    alpha: &'a [u8],
    beta: &'a [u8],
}
impl<'a> GdnCoefficientPreparationInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        alpha: &'a [u8],
        beta: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            alpha,
            beta,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn alpha_bytes(self) -> &'a [u8] {
        self.alpha
    }
    pub fn beta_bytes(self) -> &'a [u8] {
        self.beta
    }
    pub(crate) fn validate(self) -> Result<usize> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::State(
                "coefficient preparation private input source/profile".into(),
            ));
        }
        if !matches!(self.alpha.len(), 32 | 96)
            || self.beta.len() != self.alpha.len()
            || self
                .alpha
                .chunks_exact(2)
                .chain(self.beta.chunks_exact(2))
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "coefficient preparation one finite BF16 a/b call".into(),
            ));
        }
        Ok(self.alpha.len() / 2)
    }
}

#[cfg(feature = "cpu")]
pub struct GdnCoefficientPreparationOutput {
    source: SourceIdentity,
    layer: usize,
    dt: SourceParameter,
    a_log: SourceParameter,
    alpha: crate::host_math::HostValue,
    beta_input: crate::host_math::HostValue,
    values: GdnCoefficientPreparationValues<crate::host_math::HostValue>,
}
#[cfg(feature = "cpu")]
impl GdnCoefficientPreparationOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn raw_dt_bias_parameter(&self) -> &SourceParameter {
        &self.dt
    }
    pub fn unevaluated_raw_a_log_parameter(&self) -> &SourceParameter {
        &self.a_log
    }
    pub fn alpha_bf16(&self) -> Result<&[f32]> {
        self.alpha.as_slice()
    }
    pub fn beta_input_bf16(&self) -> Result<&[f32]> {
        self.beta_input.as_slice()
    }
    pub fn biased_alpha_f32(&self) -> Result<&[f32]> {
        self.values.biased_alpha.as_slice()
    }
    pub fn softplus_f32(&self) -> Result<&[f32]> {
        self.values.softplus.as_slice()
    }
    pub fn beta_bf16(&self) -> Result<&[f32]> {
        self.values.beta.as_slice()
    }
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_gdn_coefficient_preparation_slice(
        &self,
        layer: usize,
        input: GdnCoefficientPreparationInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<GdnCoefficientPreparationOutput> {
        let roles = coefficient_roles(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (alpha, beta_input) = math.import_coefficient_preparation_input(input, roles.heads)?;
        let values = crate::forward::gdn_coefficient_preparation_slice(
            self,
            &mut math,
            layer,
            &alpha,
            &beta_input,
        )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient preparation cancelled".into(),
            ));
        }
        Ok(GdnCoefficientPreparationOutput {
            source: self.source_identity(),
            layer,
            dt: roles.dt,
            a_log: roles.a_log,
            alpha,
            beta_input,
            values,
        })
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn prepare_coefficient_stages(
        &self,
        dt: &SourceParameter,
        alpha: &[f32],
        beta: &[f32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<zerogravity_cpu::original_hf::coefficient_preparation::GdnCoefficientPreparation>
    {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient raw parameter cancelled".into(),
            ));
        }
        let raw = self.source().parameter(&dt.name)?;
        raw.validate_source_parameter(dt)?;
        let window = match cancelled {
            Some(flag) => {
                self.source()
                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
            }
            None => self.source().read_parameter(&raw, 0..raw.elements())?,
        };
        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
            return Err(Error::State(
                "window coefficient complete raw dt_bias".into(),
            ));
        }
        let result =
            zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
                self.source_identity(),
                zerogravity_core::source::SourceTensorRef {
                    kind: raw.tensor().kind,
                    shape: &raw.tensor().shape,
                    packed_bytes: window.as_slice(),
                },
                alpha,
                beta,
            )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient cancelled after synchronous math".into(),
            ));
        }
        Ok(result)
    }
}

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/test_combined_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3013; SHA256 `6c7e42c86481362eb285bfb4b6fc6db5c49761db74a6cd6b2a38b7e28ee7458d`.

````text
"""Prepared source-free metadata fixtures. Author has not executed them."""
from copy import deepcopy
import unittest
from combined_source import merge_rows, report_identity


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fixture():
    prior = [dict(file=f'crates/synthetic/{i}.rs', bytes=1, sha256='a'*64)
             for i in range(521)]
    changes = []
    for index in range(7):
        name = prior[index]['file'] if index < 3 else f'crates/synthetic/new{index}.rs'
        changes.append(dict(file=name, before=prior[index] if index < 3 else None,
                            after=dict(file=name, bytes=2, sha256='b'*64)))
    return prior, changes


class CombinedSourceTests(unittest.TestCase):
    def test_exact_union_retains_every_old_path_and_four_new_paths(self):
        prior, changes = fixture()
        result = merge_rows(prior, changes, require)
        self.assertEqual(len(result), 525)
        self.assertTrue(all(row['file'] in result for row in prior))
        self.assertTrue(all(result[row['file']] == row['after'] for row in changes))

    def test_prior_metadata_cannot_be_retagged(self):
        prior, changes = fixture()
        changes[0]['before'] = dict(changes[0]['before'], bytes=3)
        with self.assertRaises(ValueError):
            merge_rows(prior, changes, require)

    def test_overlapping_deltas_refuse(self):
        prior, changes = fixture()
        changes[-1] = deepcopy(changes[-2])
        with self.assertRaises(ValueError):
            merge_rows(prior, changes, require)

    def test_missing_extra_or_duplicate_sources_refuse(self):
        prior, changes = fixture()
        for bad_prior, bad_changes in ((prior[:-1], changes),
                (prior+[prior[0]], changes), (prior, changes[:-1]),
                (prior, changes+[changes[-1]])):
            with self.assertRaises(ValueError):
                merge_rows(bad_prior, bad_changes, require)

    def test_old_binary_missing_stamp_cannot_pass(self):
        from pathlib import Path
        expected = dict(lane='initial', source_count=525, composition_sha256='a'*64)
        with self.assertRaises(ValueError):
            report_identity(Path('/synthetic'), {'status': 'old_success'},
                            lambda _: expected, require)

    def test_wrong_lane_or_source_stamp_cannot_pass(self):
        from pathlib import Path
        expected = dict(lane='initial', source_count=525, composition_sha256='a'*64)
        for wrong in (dict(expected, lane='h48'), dict(expected, source_count=524),
                      dict(expected, composition_sha256='b'*64)):
            with self.assertRaises(ValueError):
                report_identity(Path('/synthetic'),
                    {'combined525_harness_build_identity': wrong}, lambda _: expected, require)
        report_identity(Path('/synthetic'),
            {'combined525_harness_build_identity': expected}, lambda _: expected, require)


if __name__ == '__main__':
    unittest.main()

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/BASELINE.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 691; SHA256 `2f21e7479808287e3fae99991469c57150fe644d5b41f2b16af4007898371bfe`.

````text
{
  "baseline_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "owned_paths": [
    "crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs",
    "crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs",
    "crates/zerogravity-cpu/src/original_hf/initial_prelude.rs",
    "crates/zerogravity-cpu/src/original_hf.rs"
  ],
  "root_source_freeze": {
    "bytes": 99248,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
    "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
  },
  "source_only": true
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/baseline/original_hf.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11202; SHA256 `8ec1040330f1827a315a36e81c4663402783fbab91b893a0d31d6a1f05b64885`.

````text
//! Original Hugging Face output-first source rows using the existing CPU math.
//! Torch linear matrices are [output,input] in row-major memory. No whole
//! learned matrix transpose or duplicate mathematical runtime is required.
//! This does not enable complete trained Qwen3.8 inference.

pub mod coefficient_preparation;
pub mod conv;
pub mod norm;
pub mod recurrent;
pub mod reduction;
pub mod transcendentals;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_embedding;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_projection;

use crate::accelerator::{MatVecKernel, MatVecRegion, RegionExecution, validate_completed};
use crate::quant::QuantKind;
use crate::weights::{
    MAX_MATRIX_ROWS, SourceProjectionProfile, source_reference_projection_with_profile,
};
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Project one already source-qualified original [output,input] tensor
/// through the same real CPU kernel and optional physical provider.
pub fn output_first_projection(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    output_first_projection_with_profile(
        source_id,
        tensor_name,
        tensor,
        input,
        provider,
        SourceProjectionProfile::Source,
    )
}

/// Closed CPU numerical profile; an unaware physical provider must refuse.
pub fn output_first_projection_with_profile(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
    profile: SourceProjectionProfile,
) -> Result<Vec<f32>> {
    tensor.validate()?;
    if !source_id.is_bound()
        || tensor_name.is_empty()
        || tensor.shape.len() != 2
        || tensor.shape[1] != input.len() as u64
        || tensor.shape[0] > MAX_MATRIX_ROWS
        || input.is_empty()
        || !input.iter().all(|x| x.is_finite())
        || !matches!(
            tensor.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        )
    {
        return Err(Error::Malformed(
            "original output-first source GEMV geometry/encoding invalid".into(),
        ));
    }
    let rows = usize::try_from(tensor.shape[0])
        .map_err(|_| Error::Limit("output-first matrix rows exceed address space".into()))?;
    let columns = usize::try_from(tensor.shape[1])
        .map_err(|_| Error::Limit("output-first matrix columns exceed address space".into()))?;
    let region = MatVecRegion {
        source_identity: source_id,
        tensor_name,
        kind: tensor.kind,
        columns,
        rows,
        packed_weights: tensor.packed_bytes,
        activation: input,
    };
    if let Some(provider) = provider {
        if profile != SourceProjectionProfile::Source {
            return Err(Error::Unsupported(
                "physical provider does not declare the CPU projection profile".into(),
            ));
        }
        match provider.try_execute(region)? {
            RegionExecution::Unsupported => {}
            RegionExecution::Completed(values) => {
                validate_completed(region, &values)?;
                return Ok(values);
            }
        }
    }
    source_reference_projection_with_profile(region, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, RegionExecution, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct UnawareProvider(AtomicBool);
    impl MatVecKernel for UnawareProvider {
        fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Completed(vec![77.0; region.rows]))
        }
    }
    #[test]
    fn explicit_cpu_profile_refuses_unaware_provider_without_invoking_it() {
        let provider = UnawareProvider(AtomicBool::new(false));
        let packed = 0x3f80u16.to_le_bytes();
        let tensor = SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[1, 1],
            packed_bytes: &packed,
        };
        let source = source_id();
        assert!(matches!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                Some(&provider),
                SourceProjectionProfile::TorchCpuBf16Avx2,
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(!provider.0.load(Ordering::Relaxed));
        assert_eq!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                None,
                SourceProjectionProfile::TorchCpuBf16Avx2,
            )
            .unwrap(),
            [1.0]
        );
        assert_eq!(
            output_first_projection(source, "synthetic.weight", tensor, &[1.0], Some(&provider))
                .unwrap(),
            [77.0]
        );
        assert!(provider.0.load(Ordering::Relaxed));
    }

    fn source_id() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn projected(kind: QuantKind, shape: &[u64], bytes: &[u8], input: &[f32]) -> Result<Vec<f32>> {
        output_first_projection(
            source_id(),
            "original.linear.weight",
            SourceTensorRef {
                kind,
                shape,
                packed_bytes: bytes,
            },
            input,
            None,
        )
    }
    #[test]
    fn original_f32_hf_output_first_uses_same_reference_projection_without_transposition() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        assert_eq!(
            projected(QuantKind::F32, &[2, 3], &bytes, &[7., 8., 9.]).unwrap(),
            [50., 122.]
        );
        assert!(projected(QuantKind::F32, &[3, 2], &bytes, &[7., 8., 9.]).is_err());
    }
    #[test]
    fn original_bf16_and_f16_rows_reuse_existing_raw_source_math() {
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x4000, 0x4040, 0x4080] {
            bf16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::Bf16, &[2, 2], &bf16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        let mut f16 = Vec::new();
        for bits in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            f16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::F16, &[2, 2], &f16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
    }
    struct UnsupportedProvider;
    impl MatVecKernel for UnsupportedProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
    }
    struct BadProvider;
    impl MatVecKernel for BadProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(vec![f32::NAN]))
        }
    }
    #[test]
    fn optional_physical_unsupported_allows_cpu_but_invalid_completion_refuses() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        let tensor = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2, 2],
            packed_bytes: &bytes,
        };
        let normal =
            output_first_projection(source_id(), "q.weight", tensor, &[5., 6.], None).unwrap();
        let fallback = output_first_projection(
            source_id(),
            "q.weight",
            tensor,
            &[5., 6.],
            Some(&UnsupportedProvider),
        )
        .unwrap();
        assert_eq!(normal, fallback);
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                tensor,
                &[5., 6.],
                Some(&BadProvider)
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                SourceIdentity::unbound(),
                "q.weight",
                tensor,
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(source_id(), "q.weight", tensor, &[f32::NAN, 6.], None)
                .is_err()
        );
        assert!(output_first_projection(source_id(), "q.weight", tensor, &[1.], None).is_err());
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    kind: QuantKind::Q4_0,
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    shape: &[2, 3],
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        // Original source bound MUST apply before a physical provider
        // is ever permitted to override the CPU's matrix safety cap.
        let large = vec![0u8; 2 * 1_000_001];
        let huge = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &large,
        };
        assert!(
            output_first_projection(
                source_id(),
                "oversized.hf.weight",
                huge,
                &[1.0],
                Some(&UnsupportedProvider)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ObservedProvider<'a>(&'a AtomicBool);
    impl MatVecKernel for ObservedProvider<'_> {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }

    #[test]
    fn oversize_hf_output_rows_refuse_before_accelerator_invocation() {
        let called = AtomicBool::new(false);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let bytes = vec![0u8; 2 * 1_000_001];
        let tensor = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &bytes,
        };
        let result = output_first_projection(
            source,
            "original.large.weight",
            tensor,
            &[1.0],
            Some(&ObservedProvider(&called)),
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
        assert!(!called.load(Ordering::Relaxed));
    }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8063; SHA256 `3cda106a5aee61dedb96b4255295acbfc556a29a9b80d59493f3f86f56d5cb39`.

````text
{
  "MAIN_modified": false,
  "expected_output_words": 9216,
  "expected_retained_input_words": 9216,
  "files": [
    {
      "bytes": 691,
      "file": "BASELINE.json",
      "sha256": "2f21e7479808287e3fae99991469c57150fe644d5b41f2b16af4007898371bfe"
    },
    {
      "bytes": 23886,
      "file": "NEW_DELTA.patch",
      "sha256": "f6f2179ba4ee1f2c78c937c72d18231406fe5ecea1b6d7092afdfaff38ad6307"
    },
    {
      "bytes": 3980,
      "file": "README.md",
      "sha256": "42d060a3fb843609a10d3e340fedd90a72a556e10b1fd8530af26495aa457cbd"
    },
    {
      "bytes": 286,
      "file": "ROOT_WIRING.patch",
      "sha256": "971e77af71031eefcb479f2524f63f417257ea237d83bfa859ecfa661175fa64"
    },
    {
      "bytes": 672,
      "file": "TODO.md",
      "sha256": "40c1525761b5b741df7429b2830d96fcfefce9b8cd67d1cad92ee12316789330"
    },
    {
      "bytes": 11202,
      "file": "baseline/original_hf.rs",
      "sha256": "8ec1040330f1827a315a36e81c4663402783fbab91b893a0d31d6a1f05b64885"
    },
    {
      "bytes": 1043,
      "file": "owned-delta.json",
      "sha256": "7af432ac3d718de0f41b0d82b65b15c009e31047d3f87370aba7a360ab49530e"
    },
    {
      "bytes": 6111,
      "file": "pinned-source/AccumulateType.h",
      "sha256": "f1defbdc257362abe5c4530baf558bb58880d8d5522b411da12b466cf6de7c08"
    },
    {
      "bytes": 3464,
      "file": "pinned-source/LICENSE-PYTORCH",
      "sha256": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43"
    },
    {
      "bytes": 20804,
      "file": "pinned-source/ReduceOpsKernel.cpp",
      "sha256": "4b85f8c711278cea2066f2a3418fd804cb4072fecfb2c3f3ce549a2c37e2b271"
    },
    {
      "bytes": 1483,
      "file": "pinned-source/SOURCE.json",
      "sha256": "5df21e2106e935c9e931aa9fc1836e1f00dc8ebebcd41d375219be921b837a46"
    },
    {
      "bytes": 3578,
      "file": "proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs",
      "sha256": "16145cf1ce0ee6d3e2336fcd06169097a5c70c60142811359e0a5819d11b9199"
    },
    {
      "bytes": 9988,
      "file": "proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs",
      "sha256": "cfe6f7d1688a33251450bcfe075f166642a34e3c997d3e5b3a48fb8f0807c2f6"
    },
    {
      "bytes": 3464,
      "file": "proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE",
      "sha256": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43"
    },
    {
      "bytes": 5561,
      "file": "proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs",
      "sha256": "2e8cf09b4c8cdfc1b6130abb3a7da9dea0ac3fb96a3d6ff27bac02db64c7b425"
    },
    {
      "bytes": 11227,
      "file": "proposed/crates/zerogravity-cpu/src/original_hf.rs",
      "sha256": "da11a4a82664c93797ed8dba0847302f364273c9ad546446f6e8fb1d5a6b6f5a"
    },
    {
      "bytes": 772,
      "file": "replay/Cargo.toml",
      "sha256": "8a214e5c6e5b7f333ad741cb49b843bdf1966a2a1796a0b5500d19e5e20f23ab"
    },
    {
      "bytes": 2017,
      "file": "replay/ROOT.md",
      "sha256": "95f0b357cd1f20cbd309627283c81bacfacf24a6418bbf13c4b63a53163a3058"
    },
    {
      "bytes": 6287,
      "file": "replay/contracts.py",
      "sha256": "4d544e248e71d61cb46c459679929184df9d3a3146c86effd3ad94feb2dc77f8"
    },
    {
      "bytes": 29779,
      "file": "replay/factory.rs",
      "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
    },
    {
      "bytes": 20064,
      "file": "replay/main.rs",
      "sha256": "df242ea2dc34ea024b6b8bca80940dae2a0385c578001a80e4bd948c000c9b73"
    },
    {
      "bytes": 3702,
      "file": "replay/owned_lifetime.py",
      "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
    },
    {
      "bytes": 5707,
      "file": "replay/owned_native.py",
      "sha256": "05577904b1915f08aa92dd319e5412f297965e60ce67bc855317e6181652aec2"
    },
    {
      "bytes": 5616,
      "file": "replay/prepare_reference.py",
      "sha256": "e03197d639ff993426c7616325a9b9f3fef9fba2d365ff1a511f63cb45cc135f"
    },
    {
      "bytes": 3891,
      "file": "replay/run_replay.py",
      "sha256": "197e5f51513125b0a62259a39ce71b04307e32160e1b841ed5329aab2e955f91"
    },
    {
      "bytes": 1428,
      "file": "replay/sigterm_cancellation.py",
      "sha256": "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128"
    },
    {
      "bytes": 10387,
      "file": "replay/source-bindings.json",
      "sha256": "df9c0a7cf55f85feaeb85e6f5f4e10593556e51a06585e826bee28d6af490f6f"
    },
    {
      "bytes": 7058,
      "file": "replay/source-pin.json",
      "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
    },
    {
      "bytes": 4947,
      "file": "replay/source_contract.py",
      "sha256": "a6780e9737b48c64f33611118a3f253906cf275de6039579db9e8161a4bfc2b8"
    },
    {
      "bytes": 1184,
      "file": "replay/test_contracts.py",
      "sha256": "28faad90311dbdd1df445cbc0dae6d0988199555aaa0c9fd505a6d7ae7920d76"
    },
    {
      "bytes": 6640,
      "file": "replay/validate_replay.py",
      "sha256": "865d81e6069c6787c24e3feb7d5beadf5a6eca2983c282b42514545fe6928a07"
    },
    {
      "bytes": 1015,
      "file": "replay/write_implementation.py",
      "sha256": "462c3f4a6fcfb71b4cb8207eefe69cee32da7064da7e0be8451d22299fae9150"
    }
  ],
  "full_model_state_session_provider_GPU_or_performance_qualified": false,
  "parents": [
    {
      "bytes": 99248,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
      "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
    },
    {
      "bytes": 233492,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/manifest.json",
      "sha256": "62e82637c7d24a58b733f2358b3061efd767725901314df4ff7378707d946ab8"
    },
    {
      "bytes": 596,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/validation.json",
      "sha256": "951f74e33de53c4b6b5254bd980df1f6c05f6b9869a46bf617c7fa93184d344d"
    },
    {
      "bytes": 22448,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/invocation.json",
      "sha256": "61b1787ca857a1464fd61c8132475dbe7bac2b4282b3b7ff4db7bc15d427512f"
    },
    {
      "bytes": 2322,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/prepared-manifest-v3.json",
      "sha256": "d3614b48bb41033a4052d765d3f7633d456812670783049db75ceadf33d8817d"
    },
    {
      "bytes": 15423,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json",
      "sha256": "658280444d4e3d1b6a3a8f2ce4b97e4be7cbff2c8a6ea7c66b3c733c78f31638"
    },
    {
      "bytes": 7058,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source-pin.json",
      "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
    },
    {
      "bytes": 84104,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/prevalidation.json",
      "sha256": "17bd8f5430931cb33a22af32ac056f7357b02df32c9224d1c67a6361be20f46b"
    },
    {
      "bytes": 95685,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py",
      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
    }
  ],
  "runtime_or_payload_read_by_preparer": false,
  "schema": "zerogravity.initial-prefix.prepared-source.v1",
  "status": "SOURCE_ONLY_UNEXECUTED_REQUIRES_ROOT_REVIEW_AND_QUALIFICATION"
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/BASELINE.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 691; SHA256 `2f21e7479808287e3fae99991469c57150fe644d5b41f2b16af4007898371bfe`.

````text
{
  "baseline_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "owned_paths": [
    "crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs",
    "crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs",
    "crates/zerogravity-cpu/src/original_hf/initial_prelude.rs",
    "crates/zerogravity-cpu/src/original_hf.rs"
  ],
  "root_source_freeze": {
    "bytes": 99248,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
    "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
  },
  "source_only": true
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/baseline/original_hf.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11202; SHA256 `8ec1040330f1827a315a36e81c4663402783fbab91b893a0d31d6a1f05b64885`.

````text
//! Original Hugging Face output-first source rows using the existing CPU math.
//! Torch linear matrices are [output,input] in row-major memory. No whole
//! learned matrix transpose or duplicate mathematical runtime is required.
//! This does not enable complete trained Qwen3.8 inference.

pub mod coefficient_preparation;
pub mod conv;
pub mod norm;
pub mod recurrent;
pub mod reduction;
pub mod transcendentals;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_embedding;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_projection;

use crate::accelerator::{MatVecKernel, MatVecRegion, RegionExecution, validate_completed};
use crate::quant::QuantKind;
use crate::weights::{
    MAX_MATRIX_ROWS, SourceProjectionProfile, source_reference_projection_with_profile,
};
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Project one already source-qualified original [output,input] tensor
/// through the same real CPU kernel and optional physical provider.
pub fn output_first_projection(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    output_first_projection_with_profile(
        source_id,
        tensor_name,
        tensor,
        input,
        provider,
        SourceProjectionProfile::Source,
    )
}

/// Closed CPU numerical profile; an unaware physical provider must refuse.
pub fn output_first_projection_with_profile(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
    profile: SourceProjectionProfile,
) -> Result<Vec<f32>> {
    tensor.validate()?;
    if !source_id.is_bound()
        || tensor_name.is_empty()
        || tensor.shape.len() != 2
        || tensor.shape[1] != input.len() as u64
        || tensor.shape[0] > MAX_MATRIX_ROWS
        || input.is_empty()
        || !input.iter().all(|x| x.is_finite())
        || !matches!(
            tensor.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        )
    {
        return Err(Error::Malformed(
            "original output-first source GEMV geometry/encoding invalid".into(),
        ));
    }
    let rows = usize::try_from(tensor.shape[0])
        .map_err(|_| Error::Limit("output-first matrix rows exceed address space".into()))?;
    let columns = usize::try_from(tensor.shape[1])
        .map_err(|_| Error::Limit("output-first matrix columns exceed address space".into()))?;
    let region = MatVecRegion {
        source_identity: source_id,
        tensor_name,
        kind: tensor.kind,
        columns,
        rows,
        packed_weights: tensor.packed_bytes,
        activation: input,
    };
    if let Some(provider) = provider {
        if profile != SourceProjectionProfile::Source {
            return Err(Error::Unsupported(
                "physical provider does not declare the CPU projection profile".into(),
            ));
        }
        match provider.try_execute(region)? {
            RegionExecution::Unsupported => {}
            RegionExecution::Completed(values) => {
                validate_completed(region, &values)?;
                return Ok(values);
            }
        }
    }
    source_reference_projection_with_profile(region, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, RegionExecution, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct UnawareProvider(AtomicBool);
    impl MatVecKernel for UnawareProvider {
        fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Completed(vec![77.0; region.rows]))
        }
    }
    #[test]
    fn explicit_cpu_profile_refuses_unaware_provider_without_invoking_it() {
        let provider = UnawareProvider(AtomicBool::new(false));
        let packed = 0x3f80u16.to_le_bytes();
        let tensor = SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[1, 1],
            packed_bytes: &packed,
        };
        let source = source_id();
        assert!(matches!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                Some(&provider),
                SourceProjectionProfile::TorchCpuBf16Avx2,
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(!provider.0.load(Ordering::Relaxed));
        assert_eq!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                None,
                SourceProjectionProfile::TorchCpuBf16Avx2,
            )
            .unwrap(),
            [1.0]
        );
        assert_eq!(
            output_first_projection(source, "synthetic.weight", tensor, &[1.0], Some(&provider))
                .unwrap(),
            [77.0]
        );
        assert!(provider.0.load(Ordering::Relaxed));
    }

    fn source_id() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn projected(kind: QuantKind, shape: &[u64], bytes: &[u8], input: &[f32]) -> Result<Vec<f32>> {
        output_first_projection(
            source_id(),
            "original.linear.weight",
            SourceTensorRef {
                kind,
                shape,
                packed_bytes: bytes,
            },
            input,
            None,
        )
    }
    #[test]
    fn original_f32_hf_output_first_uses_same_reference_projection_without_transposition() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        assert_eq!(
            projected(QuantKind::F32, &[2, 3], &bytes, &[7., 8., 9.]).unwrap(),
            [50., 122.]
        );
        assert!(projected(QuantKind::F32, &[3, 2], &bytes, &[7., 8., 9.]).is_err());
    }
    #[test]
    fn original_bf16_and_f16_rows_reuse_existing_raw_source_math() {
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x4000, 0x4040, 0x4080] {
            bf16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::Bf16, &[2, 2], &bf16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        let mut f16 = Vec::new();
        for bits in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            f16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::F16, &[2, 2], &f16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
    }
    struct UnsupportedProvider;
    impl MatVecKernel for UnsupportedProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
    }
    struct BadProvider;
    impl MatVecKernel for BadProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(vec![f32::NAN]))
        }
    }
    #[test]
    fn optional_physical_unsupported_allows_cpu_but_invalid_completion_refuses() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        let tensor = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2, 2],
            packed_bytes: &bytes,
        };
        let normal =
            output_first_projection(source_id(), "q.weight", tensor, &[5., 6.], None).unwrap();
        let fallback = output_first_projection(
            source_id(),
            "q.weight",
            tensor,
            &[5., 6.],
            Some(&UnsupportedProvider),
        )
        .unwrap();
        assert_eq!(normal, fallback);
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                tensor,
                &[5., 6.],
                Some(&BadProvider)
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                SourceIdentity::unbound(),
                "q.weight",
                tensor,
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(source_id(), "q.weight", tensor, &[f32::NAN, 6.], None)
                .is_err()
        );
        assert!(output_first_projection(source_id(), "q.weight", tensor, &[1.], None).is_err());
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    kind: QuantKind::Q4_0,
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    shape: &[2, 3],
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        // Original source bound MUST apply before a physical provider
        // is ever permitted to override the CPU's matrix safety cap.
        let large = vec![0u8; 2 * 1_000_001];
        let huge = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &large,
        };
        assert!(
            output_first_projection(
                source_id(),
                "oversized.hf.weight",
                huge,
                &[1.0],
                Some(&UnsupportedProvider)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ObservedProvider<'a>(&'a AtomicBool);
    impl MatVecKernel for ObservedProvider<'_> {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }

    #[test]
    fn oversize_hf_output_rows_refuse_before_accelerator_invocation() {
        let called = AtomicBool::new(false);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let bytes = vec![0u8; 2 * 1_000_001];
        let tensor = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &bytes,
        };
        let result = output_first_projection(
            source,
            "original.large.weight",
            tensor,
            &[1.0],
            Some(&ObservedProvider(&called)),
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
        assert!(!called.load(Ordering::Relaxed));
    }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/owned-delta.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1043; SHA256 `7af432ac3d718de0f41b0d82b65b15c009e31047d3f87370aba7a360ab49530e`.

````text
{
  "auxiliary_files": [
    {
      "bytes": 3464,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE",
      "sha256": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43"
    }
  ],
  "files": [
    {
      "bytes": 3578,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs",
      "sha256": "16145cf1ce0ee6d3e2336fcd06169097a5c70c60142811359e0a5819d11b9199"
    },
    {
      "bytes": 9988,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs",
      "sha256": "cfe6f7d1688a33251450bcfe075f166642a34e3c997d3e5b3a48fb8f0807c2f6"
    },
    {
      "bytes": 5561,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude.rs",
      "sha256": "2e8cf09b4c8cdfc1b6130abb3a7da9dea0ac3fb96a3d6ff27bac02db64c7b425"
    },
    {
      "bytes": 11227,
      "file": "crates/zerogravity-cpu/src/original_hf.rs",
      "sha256": "da11a4a82664c93797ed8dba0847302f364273c9ad546446f6e8fb1d5a6b6f5a"
    }
  ],
  "source_only": true
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/pinned-source/AccumulateType.h

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6111; SHA256 `f1defbdc257362abe5c4530baf558bb58880d8d5522b411da12b466cf6de7c08`.

````text
#pragma once
#include <ATen/Config.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>

// Defines the accumulation type for a scalar type.
// Example:
//   using accscalar_t = acc_type<scalar_t, /*is_cuda*/true>;
//
// Accumulation types are an important concept in numeric computing
// because you frequently want to perform intermediate computations
// at a higher precision than the input and output precision, to avoid
// compounding internal rounding errors.  Accumulation is the most
// well-known intermediate computation (it is of great importance for
// sum reduction and matrix multiply, for example), but in PyTorch
// acc_type ends up getting used for all sorts of other intermediate
// computations, so it perhaps would be more accurately (ahem) called an
// "accurate" type.  acc_type is especially important for reduced
// precision operations like float16 and bfloat16, where relatively
// benign looking inputs can easily end up overflowing/underflowing.
//
// acc_type is parametrized by whether or not you are running on CUDA
// or not, because on CUDA double precision operations are expensive
// and so by default, we don't actually want to use double as an
// acc_type on CUDA.  A lot of things are typed out below, but
// basically, the table is generated by a few rules:
//
//  If bool:
//      Use 'bool' as acc_type.
//  If floating point:
//      If CUDA, use 'float' as acc_type (unless scalar_t is double),
//      otherwise (CPU) use 'double'
//  If integral:
//      Use 'int64_t' as acc_type
//
// You're not forced to use this template; if you happen to know
// something specific about your use case, you can specify your own
// desired behavior.  This template, however, will give you a reasonable
// default that will work for all dtypes supported in PyTorch.

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

namespace at {

template <typename T, c10::DeviceType D>
struct AccumulateTypeDevice {};

template <typename T, bool>
struct AccumulateType {};

template <typename T>
struct AccumulateType<T, false> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CPU>::type;
};

template <typename T>
struct AccumulateType<T, true> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CUDA>::type;
};

template <typename T, c10::DeviceType device>
using acc_type_device = typename AccumulateTypeDevice<T, device>::type;

template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

#define ACC_TYPE(t, acc_t, device_type)         \
  template <>                                   \
  struct AccumulateTypeDevice<t, device_type> { \
    using type = acc_t;                         \
  };
#define MPS_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::MPS)
#define XPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::XPU)
#define CUDA_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CUDA)
#define CPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CPU)

MPS_ACC_TYPE(BFloat16, float)
MPS_ACC_TYPE(Half, float)
MPS_ACC_TYPE(Float8_e5m2, float)
MPS_ACC_TYPE(Float8_e4m3fn, float)
MPS_ACC_TYPE(Float8_e5m2fnuz, float)
MPS_ACC_TYPE(Float8_e4m3fnuz, float)
MPS_ACC_TYPE(float, float)
MPS_ACC_TYPE(double, float)
MPS_ACC_TYPE(int8_t, int64_t)
MPS_ACC_TYPE(uint8_t, int64_t)
MPS_ACC_TYPE(char, int64_t)
MPS_ACC_TYPE(int16_t, int64_t)
MPS_ACC_TYPE(int32_t, int64_t)
MPS_ACC_TYPE(int64_t, int64_t)
MPS_ACC_TYPE(bool, bool)
MPS_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
MPS_ACC_TYPE(c10::complex<float>, c10::complex<float>)
MPS_ACC_TYPE(c10::complex<double>, c10::complex<float>)

XPU_ACC_TYPE(BFloat16, float)
XPU_ACC_TYPE(Half, float)
XPU_ACC_TYPE(Float8_e5m2, float)
XPU_ACC_TYPE(Float8_e4m3fn, float)
XPU_ACC_TYPE(Float8_e5m2fnuz, float)
XPU_ACC_TYPE(Float8_e4m3fnuz, float)
XPU_ACC_TYPE(float, float)
XPU_ACC_TYPE(double, double)
XPU_ACC_TYPE(int8_t, int64_t)
XPU_ACC_TYPE(uint8_t, int64_t)
XPU_ACC_TYPE(char, int64_t)
XPU_ACC_TYPE(int16_t, int64_t)
XPU_ACC_TYPE(int32_t, int64_t)
XPU_ACC_TYPE(int64_t, int64_t)
XPU_ACC_TYPE(bool, bool)
XPU_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
XPU_ACC_TYPE(c10::complex<float>, c10::complex<float>)
XPU_ACC_TYPE(c10::complex<double>, c10::complex<double>)

#if defined(__CUDACC__) || defined(__HIPCC__)
CUDA_ACC_TYPE(half, float)
#endif
CUDA_ACC_TYPE(BFloat16, float)
CUDA_ACC_TYPE(Half, float)
CUDA_ACC_TYPE(Float8_e5m2, float)
CUDA_ACC_TYPE(Float8_e4m3fn, float)
CUDA_ACC_TYPE(Float8_e5m2fnuz, float)
CUDA_ACC_TYPE(Float8_e4m3fnuz, float)
CUDA_ACC_TYPE(float, float)
CUDA_ACC_TYPE(double, double)
CUDA_ACC_TYPE(int8_t, int64_t)
CUDA_ACC_TYPE(uint8_t, int64_t)
CUDA_ACC_TYPE(char, int64_t)
CUDA_ACC_TYPE(int16_t, int64_t)
CUDA_ACC_TYPE(int32_t, int64_t)
CUDA_ACC_TYPE(int64_t, int64_t)
CUDA_ACC_TYPE(bool, bool)
CUDA_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
CUDA_ACC_TYPE(c10::complex<float>, c10::complex<float>)
CUDA_ACC_TYPE(c10::complex<double>, c10::complex<double>)

CPU_ACC_TYPE(BFloat16, float)
CPU_ACC_TYPE(Half, float)
CPU_ACC_TYPE(Float8_e5m2, float)
CPU_ACC_TYPE(Float8_e4m3fn, float)
CPU_ACC_TYPE(Float8_e5m2fnuz, float)
CPU_ACC_TYPE(Float8_e4m3fnuz, float)
CPU_ACC_TYPE(float, double)
CPU_ACC_TYPE(double, double)
CPU_ACC_TYPE(int8_t, int64_t)
CPU_ACC_TYPE(uint8_t, int64_t)
CPU_ACC_TYPE(char, int64_t)
CPU_ACC_TYPE(int16_t, int64_t)
CPU_ACC_TYPE(int32_t, int64_t)
CPU_ACC_TYPE(int64_t, int64_t)
CPU_ACC_TYPE(bool, bool)
CPU_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
CPU_ACC_TYPE(c10::complex<float>, c10::complex<double>)
CPU_ACC_TYPE(c10::complex<double>, c10::complex<double>)

TORCH_API c10::ScalarType toAccumulateType(
    c10::ScalarType type,
    c10::DeviceType device);
TORCH_API c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda);

} // namespace at

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/pinned-source/ReduceOpsKernel.cpp

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 20804; SHA256 `4b85f8c711278cea2066f2a3418fd804cb4072fecfb2c3f3ce549a2c37e2b271`.

````text
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/LogAddExp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

#include <c10/util/irange.h>
#include <ATen/AccumulateType.h>

namespace at::native { namespace {

using namespace vec;

template <typename scalar_t, typename func_t>
inline void cpu_cum_base_kernel(const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const func_t& f,
    scalar_t init_val) {
  if (result.sizes() != self.sizes()) {
    at::native::resize_output(result, self.sizes());
  }
  if (self.numel() == 0) {
    return;
  }
  const auto input_ndim = self.dim();
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  // TODO This probably should be using at::native::make_reduction
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dims=*/dim)
    .add_output(result)
    .add_const_input(self)
    .build();

  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for ([[maybe_unused]] const auto i : c10::irange(n)) {
      f((scalar_t*)result_data_bytes,
        result_dim_stride,
        (scalar_t*)self_data_bytes,
        self_dim_stride,
        init_val);
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, self.size(dim));
  iter.for_each(loop, grain_size);
}

void cumsum_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumsum_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          cum_number += self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 0
    );
  });
}

void cumprod_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumprod_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          cum_number *= self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 1
    );
  });
}

void logcumsumexp_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "logcumsumexp_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto cum_number = (accscalar_t)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          accscalar_t x = self_data[i * self_dim_stride];

          cum_number = _log_add_exp_helper(x, cum_number);
          result_data[i * result_dim_stride] = static_cast<scalar_t>(cum_number);
        }
      }, /*init_val=*/ -std::numeric_limits<scalar_t>::infinity()
    );
  });
}

void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
        iter,
        WelfordOps<
            scalar_t,
            double,
            int64_t,
            std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
        WelfordData<double, int64_t>());
  });
}

void prod_kernel_impl(TensorIterator& iter) {
  // Workaround for the error: '*' in boolean context, suggest '&&' instead
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    binary_kernel_reduce_vec(
        iter,
        [=](scalar_t a, scalar_t b)
            -> scalar_t { return a && b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
            { return a && b; },
        /*ident=*/1);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "prod_out_cpu", [&] {
      binary_kernel_reduce_vec(
          iter,
          [=](scalar_t a, scalar_t b)
              -> scalar_t { return a * b; },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
              { return a * b; },
          /*ident=*/1);
    });
  }
}

template <typename scalar_t, typename acc_t>
inline void norm_two_reduce_step(Vectorized<acc_t>& acc_vec, Vectorized<scalar_t>& data_vec) {
  acc_vec += data_vec * data_vec;
}

template <>
inline void norm_two_reduce_step(Vectorized<float>& acc_fvec, Vectorized<BFloat16>& data_bvec) {
  auto [data_fvec0, data_fvec1] = convert_bfloat16_float(data_bvec);
  acc_fvec += data_fvec0 * data_fvec0;
  acc_fvec += data_fvec1 * data_fvec1;
}

template <typename scalar_t, typename out_t=typename scalar_value_type<scalar_t>::type>
void norm_kernel_cpu_impl(TensorIterator& iter, const double& val) {
  // This reduction accumulates results as the type `acc_t`.
  using acc_t = at::opmath_type<typename scalar_value_type<scalar_t>::type>;
  if (val == 0.0) {
    binary_kernel_reduce(iter, NormZeroOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == 1.0) {
    binary_kernel_reduce(iter, NormOneOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == 2.0) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == INFINITY) {
    binary_kernel_reduce(iter, AbsMaxOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == -INFINITY) {
    binary_kernel_reduce(iter, AbsMinOps<scalar_t, acc_t, out_t>(), std::numeric_limits<acc_t>::infinity());
  } else {
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t, out_t>{acc_t(val)}, acc_t(0));
  }
}

void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    const Scalar& p) {
  double val = 0;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<double>();
  } else {
    TORCH_CHECK(false, "norm_kernel_cpu expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((val < 0) ? INFINITY : 0);
    return;
  }

  if (val == 2.0 && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    // If we can vectorize over the last dimension and the dtype
    // of the output is the same as that of the input,
    // then we go through the vectorised path.
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
        // use float as accumulate type for BFloat16
        using acc_t = at::opmath_type<scalar_t>;
        binary_kernel_reduce_lastdim(iter, [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
          scalar_t* result_data = (scalar_t*)result_data_bytes;
          scalar_t* self_data = (scalar_t*)self_data_bytes;

          using Vec = Vectorized<scalar_t>;
          using fVec = Vectorized<acc_t>;
          fVec acc_vec{acc_t(0)};
          acc_t buffer[fVec::size()];
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(self_data + d);
            norm_two_reduce_step(acc_vec, data_vec);
          }
          acc_vec.store(buffer);
          for (int j = 1; j < fVec::size(); j++) {
            buffer[0] = buffer[0] + buffer[j];
          }
          for (; d < size; d++) {
            acc_t data_val = acc_t(self_data[d]);
            buffer[0] += data_val * data_val;
          }
          result_data[0] = scalar_t(std::sqrt(buffer[0]));
        });
      });
  } else {
    if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
      // type promotion that does cast and reduction in a single kernel
      norm_kernel_cpu_impl<at::Half, float>(iter, val); return;
    } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
      // type promotion that does cast and reduction in a single kernel
      norm_kernel_cpu_impl<at::BFloat16, float>(iter, val); return;
    }

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kComplexHalf, iter.input_dtype(), "norm_cpu", [&] {
      norm_kernel_cpu_impl<scalar_t>(iter, val);
    });

    // For complex outputs, the above kernels do not touch the imaginary values,
    // so we must zero them out
    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

void and_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a && b) ? 1 : 0; },
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          // NB: != returns 0xFF rather than 0x01, so we must negate to get
          // the desired result
          return (a != Vectorized<uint8_t>(0)).neg() & (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/true);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a && b; },
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          // Adding the implementation here instead of in vec256_base to avoid
          // return value inconsistency. Other comparison operators in
          // vec256_base return -1/0 (all bit 1 / all bit 0) as true/false to
          // follow the AVX2 convention. This would be convenient when combined
          // with other vectorized operations. For example, one can use the
          // logical operation results as a mask for a bit operation to
          // retrieve/reset multiple elements in a vector.
          //
          // In this method, users would expect, e.g., all(), to return 1/0 as
          // true/false.
          Vectorized<bool> c = Vectorized<bool>();

          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] && b[i];
          }
          return c;
        },
        /*ident=*/true);
  }
}

void or_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a || b) ? 1 : 0; },
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          return (a != Vectorized<uint8_t>(0)).neg() | (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/false);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a || b; },
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          Vectorized<bool> c = Vectorized<bool>();

          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] || b[i];
          }
          return c;
        },
        /*ident=*/false);
  }
}

template<typename scalar_t>
struct MinValuesOps: public at::native::MinOps<scalar_t> {
  using arg_t = typename MinOps<scalar_t>::arg_t;
  static scalar_t project(arg_t arg) {
    return arg.first;
  }
};

void min_values_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == kLong) {
    // This case is special because of Vectorized<int64_t> does not
    // handle upper_bound<int64_t>().
    // See: https://github.com/pytorch/pytorch/issues/43254
    using scalar_t = int64_t;
    binary_kernel_reduce(
      iter,
      MinValuesOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return minimum(a, b); },
      static_cast<double>(upper_bound<scalar_t>()));
  });
}

void max_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return maximum(a, b); },
      lower_bound<scalar_t>());
  });
}

void argmax_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmax_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMaxOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(lower_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMaxOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(lower_bound<scalar_t>(), 0));
  });
}

void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmin_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMinOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(upper_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), 0));
  });
}

template <typename scalar_t, typename acc_t = uint64_t, typename out_t = acc_t>
struct XorSumOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    if (std::is_same<scalar_t, bool>::value) {
      return acc ^ (data ? 1 : 0);
    } else if (
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, at::BFloat16>::value ||
        std::is_same<scalar_t, at::Half>::value) {
      union {
        double d;
        uint64_t u;
      } converter;
      converter.d = static_cast<double>(data);
      return acc ^ converter.u;
    } else {
      return acc ^ static_cast<uint64_t>(data);
    }
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a ^ b;
  }

  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

void xor_sum_kernel_impl(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(1), "xor_sum_cpu", [&] {
        binary_kernel_reduce(
            iter, XorSumOps<scalar_t>(), static_cast<uint64_t>(0));
      });
}

// powsum: computes sum(|x|^p) without the final root
template <typename scalar_t, typename out_t=typename scalar_value_type<scalar_t>::type>
void powsum_kernel_cpu_impl(TensorIterator& iter, const double& val) {
  using acc_t = at::opmath_type<typename scalar_value_type<scalar_t>::type>;
  if (val == 2.0) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t, false>(), acc_t(0));
  } else {
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t, out_t, false>{acc_t(val)}, acc_t(0));
  }
}

void powsum_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    const Scalar& p) {
  double val = 0;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<double>();
  } else {
    TORCH_CHECK(false, "powsum_kernel_cpu expects ord to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }

  if (val == 2.0 && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    // Vectorized path for L2 powsum (no sqrt at end)
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "powsum_cpu", [&] {
        using acc_t = at::opmath_type<scalar_t>;
        binary_kernel_reduce_lastdim(iter, [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
          scalar_t* result_data = (scalar_t*)result_data_bytes;
          scalar_t* self_data = (scalar_t*)self_data_bytes;

          using Vec = Vectorized<scalar_t>;
          using fVec = Vectorized<acc_t>;
          fVec acc_vec{acc_t(0)};
          acc_t buffer[fVec::size()];
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(self_data + d);
            norm_two_reduce_step(acc_vec, data_vec);
          }
          acc_vec.store(buffer);
          for (int j = 1; j < fVec::size(); j++) {
            buffer[0] = buffer[0] + buffer[j];
          }
          for (; d < size; d++) {
            acc_t data_val = acc_t(self_data[d]);
            buffer[0] += data_val * data_val;
          }
          result_data[0] = scalar_t(buffer[0]);  // No sqrt!
        });
      });
  } else {
    if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
      powsum_kernel_cpu_impl<at::Half, float>(iter, val); return;
    } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
      powsum_kernel_cpu_impl<at::BFloat16, float>(iter, val); return;
    }

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kComplexHalf, iter.input_dtype(), "powsum_cpu", [&] {
      powsum_kernel_cpu_impl<scalar_t>(iter, val);
    });

    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

}  // anonymous namespace

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl)
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl)
// mean implementation for CPU is in aten/src/ATen/native/ReduceOps.cpp
// but mean_stub must be defined for CPU as well
REGISTER_DISPATCH(mean_stub, nullptr)
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl)
REGISTER_DISPATCH(powsum_stub, &powsum_kernel_tensor_iterator_impl)
REGISTER_DISPATCH(and_stub, &and_kernel_impl)
REGISTER_DISPATCH(or_stub, &or_kernel_impl)
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl)
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl)
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl)
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl)
REGISTER_DISPATCH(xor_sum_stub, &xor_sum_kernel_impl)

REGISTER_DISPATCH(cumprod_stub, &cumprod_cpu_kernel)
REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel)
REGISTER_DISPATCH(logcumsumexp_stub, &logcumsumexp_cpu_kernel)

}  // namespace at::native

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/pinned-source/SOURCE.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1483; SHA256 `5df21e2106e935c9e931aa9fc1836e1f00dc8ebebcd41d375219be921b837a46`.

````text
{
  "files": [
    {
      "bytes": 20804,
      "file": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/ReduceOpsKernel.cpp",
      "sha256": "4b85f8c711278cea2066f2a3418fd804cb4072fecfb2c3f3ce549a2c37e2b271",
      "url": "https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp"
    },
    {
      "bytes": 6111,
      "file": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/AccumulateType.h",
      "sha256": "f1defbdc257362abe5c4530baf558bb58880d8d5522b411da12b466cf6de7c08",
      "url": "https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/AccumulateType.h"
    },
    {
      "bytes": 3464,
      "file": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/LICENSE-PYTORCH",
      "sha256": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43",
      "url": "https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/LICENSE"
    }
  ],
  "reviewed_schedule": {
    "accumulator": "CPU_ACC_TYPE(float,double)",
    "cumsum": "cpu_cum_base_kernel +cumsum_cpu_kernel, F64 carry/+0/F32 each store"
  },
  "revision": "cf30153c4c131c8164ee7798e5022d810682e2cb",
  "scope": "primary code only; no operator executed"
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3578; SHA256 `16145cf1ce0ee6d3e2336fcd06169097a5c70c60142811359e0a5819d11b9199`.

````text
//! Actual same-source host backing, not a producer/session/provider authority.
use super::*;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};

/// External authenticated operator bytes. Caller proves archive/producer origin;
/// importing them never claims a current state, model projection or decay math.
#[derive(Clone, Copy)]
pub struct InitialPreludeInput<'a> {
    contract: InitialPreludeContract,
    raw: &'a [u8],
}
impl<'a> InitialPreludeInput<'a> {
    pub fn from_f32le(contract: InitialPreludeContract, raw: &'a [u8]) -> Result<Self> {
        if raw.len() != contract.elements() * 4
            || raw
                .chunks_exact(4)
                .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
        {
            return Err(Error::Malformed(
                "initial GDN external finite raw FP32".into(),
            ));
        }
        Ok(Self { contract, raw })
    }
    pub fn contract(self) -> InitialPreludeContract {
        self.contract
    }
}
struct Backing {
    contract: InitialPreludeContract,
    input: Vec<f32>,
    output: Vec<f32>,
    // Actual buffers retire before their admission/source owner.
    workspace: WindowWorkspace,
}
/// Clones retain the SAME source/backing/lease; no uncharged Vec escape.
#[derive(Clone)]
pub struct InitialPreludeOutput(Arc<Backing>);
impl InitialPreludeOutput {
    pub fn contract(&self) -> InitialPreludeContract {
        self.0.contract
    }
    pub fn imported_input(&self) -> &[f32] {
        &self.0.input
    }
    pub fn output(&self) -> &[f32] {
        &self.0.output
    }
    pub fn charged_bytes(&self) -> usize {
        self.0.workspace.bytes()
    }
    pub fn same_issuer(&self, source: &VerifiedTensorSource) -> bool {
        self.0.workspace.source().same_issuer(source)
    }
    pub fn required_bytes(contract: InitialPreludeContract) -> usize {
        contract.elements() * 8 + 256
    }
}

/// One pure synchronous leaf. All payload allocation follows aggregate admission.
/// No GPU/BLAS/provider path and no fallback; cancelled output is discarded.
pub fn execute_initial_prelude(
    source: &VerifiedTensorSource,
    input: InitialPreludeInput<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<InitialPreludeOutput> {
    not_cancelled(cancelled)?;
    let contract = input.contract;
    if contract.source_identity() != source.source_identity() {
        return Err(Error::State("initial GDN foreign raw source issuer".into()));
    }
    let bytes = contract.elements() * 4;
    let workspace = source.reserve_operator_buffers(bytes, bytes + 256)?;
    not_cancelled(cancelled)?;
    let mut imported = Vec::with_capacity(contract.elements());
    imported.extend(
        input
            .raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    );
    let mut output = vec![0.0f32; contract.elements()];
    if imported.capacity() != contract.elements() || output.capacity() != contract.elements() {
        return Err(Error::Limit("initial GDN exact payload capacities".into()));
    }
    initial_prelude_into(contract, &imported, &mut output, cancelled)?;
    let result = InitialPreludeOutput(Arc::new(Backing {
        contract,
        input: imported,
        output,
        workspace,
    }));
    completed(result, cancelled)
}
pub(super) fn completed(
    result: InitialPreludeOutput,
    cancelled: Option<&AtomicBool>,
) -> Result<InitialPreludeOutput> {
    not_cancelled(cancelled)?;
    Ok(result)
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 9988; SHA256 `cfe6f7d1688a33251450bcfe075f166642a34e3c997d3e5b3a48fb8f0807c2f6`.

````text
use super::*;
fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
    InitialPreludeContract::new(
        SourceIdentity::for_new_allocation().unwrap(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        GdnPreludePhase::InitialChunk64,
        operation,
        heads,
    )
    .unwrap()
}
#[test]
fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
    let mut input = vec![0.0; c.elements()];
    input[0] = 1.0;
    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
    input[2] = -1.0;
    input[64] = -0.0;
    input[128] = -1.0;
    let mut output = vec![9.0; input.len()];
    initial_prelude_into(c, &input, &mut output, None).unwrap();
    assert_eq!(output[0].to_bits(), 0x3f80_0000);
    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
    assert_eq!(output[2].to_bits(), 0x3380_0000);
    assert_eq!(output[63].to_bits(), 0x3380_0000);
    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
    assert_eq!(output[127].to_bits(), 0);
    assert_eq!(output[128].to_bits(), 0xbf80_0000);
    assert_eq!(output[191].to_bits(), 0xbf80_0000);
}
#[test]
fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
    let c = contract(InitialPreludeOperation::QueryScale128, 48);
    let mut input = vec![1.0; c.elements()];
    input[..6].copy_from_slice(&[
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
    ]);
    let mut out = vec![0.0; input.len()];
    initial_prelude_into(c, &input, &mut out, None).unwrap();
    assert_eq!(
        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
    );
    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
}
#[test]
fn cached_profile_source_geometry_and_complete_span_refuse() {
    for (source, profile, phase, heads) in [
        (
            SourceIdentity::unbound(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::GgufSourceV1,
            GdnPreludePhase::InitialChunk64,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::Cached,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            32,
        ),
    ] {
        assert!(
            InitialPreludeContract::new(
                source,
                profile,
                phase,
                InitialPreludeOperation::DecayPrefix64,
                heads
            )
            .is_err()
        );
    }
    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
    let mut input = vec![0.0; c.elements()];
    let mut out = vec![7.0; c.elements()];
    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
    input[0] = f32::NAN;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
    input[0] = f32::INFINITY;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
    assert!(out.iter().all(|v| *v == 7.0));
    input[0] = 0.0;
    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
    assert!(out.iter().all(|v| *v == 7.0));
    input[0] = f32::MAX;
    input[1] = f32::MAX;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod physical {
    use super::*;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU64;
    use zerogravity_artifacts::artifact::Artifact;
    use zerogravity_artifacts::cache::TransferAdmission;
    use zerogravity_artifacts::source_weights::{
        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
    };
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const META: u64 = 16 * 1024 * 1024;
    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let p = std::env::temp_dir().join(format!(
                "zg-initial-prefix-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&p).unwrap();
            fs::write(p.join("model.safetensors"), BYTES).unwrap();
            Self(p)
        }
        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
            SourceWeights::prepare_verified_windows(
                INDEX,
                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
                vec![
                    ExpectedSourceShard::new(
                        "model.safetensors".into(),
                        Artifact::from_file(
                            File::open(self.0.join("model.safetensors")).unwrap(),
                            Path::new("model.safetensors"),
                        )
                        .unwrap(),
                        BYTES.len() as u64,
                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
                    )
                    .unwrap(),
                ],
                WindowOpenLimits {
                    max_source_bytes: 1024 * 1024,
                    max_metadata_bytes: META,
                    max_shards: 1,
                    max_tensors: 6,
                    scan_chunk_bytes: 1024,
                },
                h,
            )
            .unwrap()
            .open()
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn host() -> TransferAdmission {
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
        arbiter
            .register(resource.clone(), 64 * 1024 * 1024)
            .unwrap();
        TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
        }
    }
    fn reserved(h: &TransferAdmission) -> u64 {
        h.arbiter.snapshot(&h.resource).unwrap().reserved
    }
    fn raw(c: InitialPreludeContract) -> Vec<u8> {
        1.0f32.to_le_bytes().repeat(c.elements())
    }
    #[test]
    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
        let f = Fixture::new();
        let h = host();
        let source = f.source(&h);
        let foreign = f.source(&h);
        let c = InitialPreludeContract::new(
            source.source_identity(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            InitialPreludeOperation::QueryScale128,
            48,
        )
        .unwrap();
        let raw = raw(c);
        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
        let before = reserved(&h);
        assert!(execute_initial_prelude(&foreign, input, None).is_err());
        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
        assert_eq!(reserved(&h), before);
        let need = InitialPreludeOutput::required_bytes(c) as u64;
        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
        let pressure = source.reserve_workspace(hold as usize).unwrap();
        let pressured = reserved(&h);
        assert!(execute_initial_prelude(&source, input, None).is_err());
        assert_eq!(reserved(&h), pressured);
        drop(pressure);
        assert_eq!(reserved(&h), before);
        let output = execute_initial_prelude(&source, input, None).unwrap();
        let clone = output.clone();
        assert!(output.same_issuer(&source));
        assert!(!output.same_issuer(&foreign));
        assert_eq!(reserved(&h), before + need);
        assert_eq!(output.charged_bytes() as u64, need);
        drop(foreign);
        drop(source);
        assert_eq!(reserved(&h), META + need);
        assert_eq!(clone.imported_input().len(), 48 * 128);
        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
        drop(output);
        assert_eq!(reserved(&h), META + need);
        drop(clone);
        assert_eq!(reserved(&h), 0);
    }
    #[test]
    fn completed_result_cancel_drops_real_admitted_backing() {
        let f = Fixture::new();
        let h = host();
        let source = f.source(&h);
        let c = InitialPreludeContract::new(
            source.source_identity(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            InitialPreludeOperation::DecayPrefix64,
            16,
        )
        .unwrap();
        let raw = raw(c);
        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
        let value = execute_initial_prelude(&source, input, None).unwrap();
        assert!(reserved(&h) > META);
        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
        assert_eq!(reserved(&h), META);
        drop(source);
        assert_eq!(reserved(&h), 0);
    }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5561; SHA256 `2e8cf09b4c8cdfc1b6130abb3a7da9dea0ac3fb96a3d6ff27bac02db64c7b425`.

````text
//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Phase is explicit; cached query division is deliberately unsupported here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnPreludePhase {
    InitialChunk64,
    Cached,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialPreludeOperation {
    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
    QueryScale128,
    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
    DecayPrefix64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitialPreludeContract {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    operation: InitialPreludeOperation,
    heads: usize,
}
impl InitialPreludeContract {
    pub fn new(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        phase: GdnPreludePhase,
        operation: InitialPreludeOperation,
        heads: usize,
    ) -> Result<Self> {
        if phase != GdnPreludePhase::InitialChunk64
            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
        }
        if !source.is_bound() || !matches!(heads, 16 | 48) {
            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
        }
        Ok(Self {
            source,
            profile,
            operation,
            heads,
        })
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn phase(self) -> GdnPreludePhase {
        GdnPreludePhase::InitialChunk64
    }
    pub fn operation(self) -> InitialPreludeOperation {
        self.operation
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn elements(self) -> usize {
        self.heads
            * match self.operation {
                InitialPreludeOperation::QueryScale128 => 128,
                InitialPreludeOperation::DecayPrefix64 => 64,
            }
    }
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("initial GDN leaf cancelled".into()));
    }
    Ok(())
}
fn environment() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut mxcsr = 0u32;
        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
        // Passive read only; exception status bits may change during arithmetic.
        unsafe {
            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
                options(nostack, preserves_flags));
        }
        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
            return Err(Error::Unsupported(
                "initial GDN pinned CPU FP controls".into(),
            ));
        }
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(Error::Unsupported(
        "initial GDN pinned CPU architecture".into(),
    ))
}

/// No allocation/provider callback. Caller owns both complete, disjoint slices.
/// Output remains private on error; successful completion grants no state effect.
pub fn initial_prelude_into(
    contract: InitialPreludeContract,
    input: &[f32],
    output: &mut [f32],
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    environment()?;
    if input.len() != contract.elements()
        || output.len() != input.len()
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Malformed(
            "initial GDN complete finite FP32 span".into(),
        ));
    }
    match contract.operation {
        InitialPreludeOperation::QueryScale128 => {
            // Source query * (128 ** -0.5), scalar converted once to FP32.
            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
                not_cancelled(cancelled)?;
                for (&value, result) in row.iter().zip(target) {
                    *result = value * SCALE;
                }
            }
        }
        InitialPreludeOperation::DecayPrefix64 => {
            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
                not_cancelled(cancelled)?;
                let mut carry = 0.0f64;
                for (&value, result) in row.iter().zip(target) {
                    carry += f64::from(value);
                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
                    if !result.is_finite() {
                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
                    }
                }
            }
        }
    }
    not_cancelled(cancelled)
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod admitted;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
#[cfg(test)]
mod tests;

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/proposed/crates/zerogravity-cpu/src/original_hf.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11227; SHA256 `da11a4a82664c93797ed8dba0847302f364273c9ad546446f6e8fb1d5a6b6f5a`.

````text
//! Original Hugging Face output-first source rows using the existing CPU math.
//! Torch linear matrices are [output,input] in row-major memory. No whole
//! learned matrix transpose or duplicate mathematical runtime is required.
//! This does not enable complete trained Qwen3.8 inference.

pub mod coefficient_preparation;
pub mod initial_prelude;
pub mod conv;
pub mod norm;
pub mod recurrent;
pub mod reduction;
pub mod transcendentals;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_embedding;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_projection;

use crate::accelerator::{MatVecKernel, MatVecRegion, RegionExecution, validate_completed};
use crate::quant::QuantKind;
use crate::weights::{
    MAX_MATRIX_ROWS, SourceProjectionProfile, source_reference_projection_with_profile,
};
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Project one already source-qualified original [output,input] tensor
/// through the same real CPU kernel and optional physical provider.
pub fn output_first_projection(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    output_first_projection_with_profile(
        source_id,
        tensor_name,
        tensor,
        input,
        provider,
        SourceProjectionProfile::Source,
    )
}

/// Closed CPU numerical profile; an unaware physical provider must refuse.
pub fn output_first_projection_with_profile(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
    profile: SourceProjectionProfile,
) -> Result<Vec<f32>> {
    tensor.validate()?;
    if !source_id.is_bound()
        || tensor_name.is_empty()
        || tensor.shape.len() != 2
        || tensor.shape[1] != input.len() as u64
        || tensor.shape[0] > MAX_MATRIX_ROWS
        || input.is_empty()
        || !input.iter().all(|x| x.is_finite())
        || !matches!(
            tensor.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        )
    {
        return Err(Error::Malformed(
            "original output-first source GEMV geometry/encoding invalid".into(),
        ));
    }
    let rows = usize::try_from(tensor.shape[0])
        .map_err(|_| Error::Limit("output-first matrix rows exceed address space".into()))?;
    let columns = usize::try_from(tensor.shape[1])
        .map_err(|_| Error::Limit("output-first matrix columns exceed address space".into()))?;
    let region = MatVecRegion {
        source_identity: source_id,
        tensor_name,
        kind: tensor.kind,
        columns,
        rows,
        packed_weights: tensor.packed_bytes,
        activation: input,
    };
    if let Some(provider) = provider {
        if profile != SourceProjectionProfile::Source {
            return Err(Error::Unsupported(
                "physical provider does not declare the CPU projection profile".into(),
            ));
        }
        match provider.try_execute(region)? {
            RegionExecution::Unsupported => {}
            RegionExecution::Completed(values) => {
                validate_completed(region, &values)?;
                return Ok(values);
            }
        }
    }
    source_reference_projection_with_profile(region, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, RegionExecution, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct UnawareProvider(AtomicBool);
    impl MatVecKernel for UnawareProvider {
        fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Completed(vec![77.0; region.rows]))
        }
    }
    #[test]
    fn explicit_cpu_profile_refuses_unaware_provider_without_invoking_it() {
        let provider = UnawareProvider(AtomicBool::new(false));
        let packed = 0x3f80u16.to_le_bytes();
        let tensor = SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[1, 1],
            packed_bytes: &packed,
        };
        let source = source_id();
        assert!(matches!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                Some(&provider),
                SourceProjectionProfile::TorchCpuBf16Avx2,
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(!provider.0.load(Ordering::Relaxed));
        assert_eq!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                None,
                SourceProjectionProfile::TorchCpuBf16Avx2,
            )
            .unwrap(),
            [1.0]
        );
        assert_eq!(
            output_first_projection(source, "synthetic.weight", tensor, &[1.0], Some(&provider))
                .unwrap(),
            [77.0]
        );
        assert!(provider.0.load(Ordering::Relaxed));
    }

    fn source_id() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn projected(kind: QuantKind, shape: &[u64], bytes: &[u8], input: &[f32]) -> Result<Vec<f32>> {
        output_first_projection(
            source_id(),
            "original.linear.weight",
            SourceTensorRef {
                kind,
                shape,
                packed_bytes: bytes,
            },
            input,
            None,
        )
    }
    #[test]
    fn original_f32_hf_output_first_uses_same_reference_projection_without_transposition() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        assert_eq!(
            projected(QuantKind::F32, &[2, 3], &bytes, &[7., 8., 9.]).unwrap(),
            [50., 122.]
        );
        assert!(projected(QuantKind::F32, &[3, 2], &bytes, &[7., 8., 9.]).is_err());
    }
    #[test]
    fn original_bf16_and_f16_rows_reuse_existing_raw_source_math() {
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x4000, 0x4040, 0x4080] {
            bf16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::Bf16, &[2, 2], &bf16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        let mut f16 = Vec::new();
        for bits in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            f16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::F16, &[2, 2], &f16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
    }
    struct UnsupportedProvider;
    impl MatVecKernel for UnsupportedProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
    }
    struct BadProvider;
    impl MatVecKernel for BadProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(vec![f32::NAN]))
        }
    }
    #[test]
    fn optional_physical_unsupported_allows_cpu_but_invalid_completion_refuses() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        let tensor = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2, 2],
            packed_bytes: &bytes,
        };
        let normal =
            output_first_projection(source_id(), "q.weight", tensor, &[5., 6.], None).unwrap();
        let fallback = output_first_projection(
            source_id(),
            "q.weight",
            tensor,
            &[5., 6.],
            Some(&UnsupportedProvider),
        )
        .unwrap();
        assert_eq!(normal, fallback);
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                tensor,
                &[5., 6.],
                Some(&BadProvider)
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                SourceIdentity::unbound(),
                "q.weight",
                tensor,
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(source_id(), "q.weight", tensor, &[f32::NAN, 6.], None)
                .is_err()
        );
        assert!(output_first_projection(source_id(), "q.weight", tensor, &[1.], None).is_err());
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    kind: QuantKind::Q4_0,
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    shape: &[2, 3],
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        // Original source bound MUST apply before a physical provider
        // is ever permitted to override the CPU's matrix safety cap.
        let large = vec![0u8; 2 * 1_000_001];
        let huge = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &large,
        };
        assert!(
            output_first_projection(
                source_id(),
                "oversized.hf.weight",
                huge,
                &[1.0],
                Some(&UnsupportedProvider)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ObservedProvider<'a>(&'a AtomicBool);
    impl MatVecKernel for ObservedProvider<'_> {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }

    #[test]
    fn oversize_hf_output_rows_refuse_before_accelerator_invocation() {
        let called = AtomicBool::new(false);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let bytes = vec![0u8; 2 * 1_000_001];
        let tensor = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &bytes,
        };
        let result = output_first_projection(
            source,
            "original.large.weight",
            tensor,
            &[1.0],
            Some(&ObservedProvider(&called)),
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
        assert!(!called.load(Ordering::Relaxed));
    }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 772; SHA256 `8a214e5c6e5b7f333ad741cb49b843bdf1966a2a1796a0b5500d19e5e20f23ab`.

````text
[package]
name = "original27b-initial-prefix-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu", features = ["cache"] }
serde_json = "=1.0.151"
sha2 = "=0.10.9"
libc = "=0.2.190"

[profile.release]
opt-level = 3
codegen-units = 1

[[bin]]
name = "original27b-initial-prefix-replay"
path = "main.rs"

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6287; SHA256 `4d544e248e71d61cb46c459679929184df9d3a3146c86effd3ad94feb2dc77f8`.

````text
"""ROOT-only bounded metadata/raw transport; no model/provider imports/calls."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HERE = Path(__file__).resolve().parent
INPUTS = ('query_input', 'prefix_input')
OUTPUTS = ('query_output', 'prefix_output')
PROFILE = 'original27B-raw-source-bf16-v1'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    with Path(path).open('rb') as opened:
        return hashlib.file_digest(opened, 'sha256').hexdigest()

def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out

def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

def read(path, cap=2*1024**2, row=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
        before = os.fstat(opened.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
        raw = opened.read(cap+1)
        require(len(raw) == before.st_size and identity(before) == identity(os.fstat(opened.fileno())),
                'bounded unchanged metadata FD')
    if row is not None:
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'exact metadata bytes/SHA')
    return raw

def pin(path, row, cap=2*1024**2):
    read(path, cap, row)

def read_json(path):
    return json.loads(read(path), object_pairs_hook=unique)

def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    require(len(raw) <= 2*1024**2, 'bounded metadata receipt')
    with Path(path).open('xb') as opened:
        opened.write(raw)

def frozen():
    from source_contract import frozen as checked_frozen
    return checked_frozen(HERE, read_json, pin, require, sha)

def implementation(path):
    from source_contract import implementation as checked_implementation
    return checked_implementation(HERE, path, read_json, pin, require, sha)


def finite(raw, dtype):
    require(dtype in ('BF16', 'F32'), 'closed observed dtype')
    width, fmt, mask = (2, '<H', 0x7f80) if dtype == 'BF16' else (4, '<I', 0x7f800000)
    require(raw and len(raw) % width == 0 and
            all(word & mask != mask for word, in struct.iter_unpack(fmt, raw)),
            'finite complete observed words; no signed-zero canonicalization')

def widened(raw, dtype):
    finite(raw, dtype)
    return b''.join((word << 16).to_bytes(4, 'little') for word, in struct.iter_unpack('<H', raw)) \
        if dtype == 'BF16' else raw

class RawArchive:
    """Whole-file streaming authentication; bounded selected reads on SAME FD."""
    def __init__(self, row):
        self.row, self.opened = row, None
    def __enter__(self):
        self.opened = os.fdopen(os.open(self.row['path'], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb')
        try:
            self.before = os.fstat(self.opened.fileno())
            require(stat.S_ISREG(self.before.st_mode) and self.before.st_size == self.row['bytes'] and
                    8 < self.before.st_size <= 64*1024**2, 'bounded full raw archive FD')
            self._authenticate()
            size, = struct.unpack('<Q', os.pread(self.opened.fileno(), 8, 0))
            require(1 < size <= 1024**2 and 8+size < self.before.st_size,
                    'header bound BEFORE reading declared extent')
            raw = os.pread(self.opened.fileno(), size, 8)
            require(len(raw) == size, 'complete original header read')
            self.header = json.loads(raw, object_pairs_hook=unique)
            require(type(self.header) is dict and 1 <= len(self.header) <= 2048, 'bounded archive fields')
            self.start, spans = 8+size, []
            for name, item in self.header.items():
                if name == '__metadata__':
                    continue
                require(item['dtype'] in ('BF16', 'F32') and type(item['shape']) is list and
                        len(item['shape']) <= 8, 'original rank/dtype')
                words = 1
                for axis in item['shape']:
                    require(type(axis) is int and 0 < axis <= 262144, 'bounded original axis')
                    words *= axis
                require(type(item['data_offsets']) is list and len(item['data_offsets']) == 2,
                        'two original offsets')
                lo, hi = item['data_offsets']
                require(type(lo) is type(hi) is int and 0 <= lo < hi <= self.before.st_size-self.start and
                        hi-lo == words*(2 if item['dtype'] == 'BF16' else 4), 'complete actual archive span')
                spans.append((lo, hi))
            end = 0
            for lo, hi in sorted(spans):
                require(lo == end, 'no overlapping/gapped actual archive data')
                end = hi
            require(self.start+end == self.before.st_size, 'no unaccounted trailing archive bytes')
            return self
        except BaseException:
            self.opened.close()
            raise
    def _authenticate(self):
        self.opened.seek(0)
        require(hashlib.file_digest(self.opened, 'sha256').hexdigest() == self.row['sha256'] and
                identity(os.fstat(self.opened.fileno())) == identity(self.before),
                'complete actual archive digest/FD stable')
    def raw(self, row):
        item = self.header[row['name']]
        require(item['dtype'] == row['dtype'] and item['shape'] == row['shape'], 'actual observed name/dtype/shape')
        lo, hi = item['data_offsets']
        require(0 < hi-lo == row['bytes'] <= 24576, 'bounded actual selected stage')
        raw = os.pread(self.opened.fileno(), hi-lo, self.start+lo)
        require(len(raw) == hi-lo and digest(raw) == row['raw_sha256'] and
                identity(os.fstat(self.opened.fileno())) == identity(self.before), 'all raw stage bits and same retained FD')
        finite(raw, row['dtype'])
        return raw
    def __exit__(self, typ, value, trace):
        try:
            if typ is None:
                self._authenticate()
        finally:
            self.opened.close()

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/factory.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 29779; SHA256 `2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69`.

````text
#![allow(dead_code)]
//! Root-only actual-source byte/window harness. No model or GPU execution.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::ops::Range;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zerogravity_artifacts::Error as ZgError;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::source_weights::{
    ExpectedSourceShard, SourceWeights, VerifiedMatrix, VerifiedTensorSource, VerifiedWindowPlan,
    WindowOpenLimits,
};
use zerogravity_core::accelerator::program::MatrixAxes;
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

type Proof<T> = Result<T, Box<dyn Error>>;
const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
const REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
const REPORT_SHA: &str = "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a";
const INDEX_SHA: &str = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
const BUNDLE_SHA: &str = "7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2";
const SELECTED_BYTES: u64 = 55_586_113_293;
const WEIGHT_BYTES: u64 = 55_563_006_776;
const PAYLOAD_BYTES: u64 = 55_562_855_904;
const META: u64 = 64 * 1024 * 1024;
const HOST: u64 = 128 * 1024 * 1024;
const SCAN: usize = 4 * 1024 * 1024;
const SOURCE_DIR: &str = "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files";

struct Shard {
    name: String,
    expected_bytes: u64,
    expected_sha: String,
    // Both Artifact and independent raw reference derive from this one open FD.
    file: File,
}

fn ensure(condition: bool, message: impl Into<String>) -> Proof<()> {
    if !condition {
        let message: String = message.into();
        return Err(message.into());
    }
    Ok(())
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bounded_file(path: &Path, maximum: usize) -> Proof<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    ensure(file.metadata()?.is_file(), "control source is not regular")?;
    ensure(
        file.metadata()?.len() <= maximum as u64,
        "control source byte cap",
    )?;
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure(bytes.len() <= maximum, "control source grew beyond cap")?;
    Ok(bytes)
}

fn reserved(host: &TransferAdmission) -> Proof<u64> {
    Ok(host.arbiter.snapshot(&host.resource)?.reserved)
}

fn ledger(host: &TransferAdmission, metadata: u64, staging: u64, other: u64) -> Proof<()> {
    let current = host.arbiter.snapshot(&host.resource)?;
    ensure(!current.fenced, "host window pool unexpectedly fenced")?;
    ensure(current.limit == HOST, "configured pool limit changed")?;
    ensure(
        current.reserved == metadata + staging + other,
        "live total ledger mismatch",
    )?;
    for (class, expected) in [
        (AllocationClass::Scratch, metadata),
        (AllocationClass::TransferStaging, staging),
        (AllocationClass::Other, other),
    ] {
        ensure(
            current.per_class.get(&class).copied().unwrap_or(0) == expected,
            format!("live allocation-class mismatch: {class:?}"),
        )?;
    }
    ensure(
        current.remaining + current.reserved == HOST,
        "remaining ledger mismatch",
    )?;
    Ok(())
}

fn limits() -> WindowOpenLimits {
    WindowOpenLimits {
        max_source_bytes: 64 * 1024 * 1024 * 1024,
        max_metadata_bytes: META,
        max_shards: 18,
        max_tensors: 1199,
        scan_chunk_bytes: SCAN,
    }
}

fn expected_shards(shards: &[Shard]) -> Proof<Vec<ExpectedSourceShard>> {
    shards
        .iter()
        .map(|shard| {
            let clone = shard.file.try_clone()?;
            let original = shard.file.metadata()?;
            let same = clone.metadata()?;
            ensure(
                (same.dev(), same.ino(), same.len())
                    == (original.dev(), original.ino(), original.len()),
                "descriptor clone changed source",
            )?;
            Ok(ExpectedSourceShard::new(
                shard.name.clone(),
                Artifact::from_file(clone, Path::new(&shard.name))?,
                shard.expected_bytes,
                shard.expected_sha.clone(),
            )?)
        })
        .collect()
}

fn plan(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<VerifiedWindowPlan> {
    Ok(SourceWeights::prepare_verified_windows(
        index,
        INDEX_SHA,
        expected_shards(shards)?,
        limits(),
        host,
    )?)
}

fn expected_limit<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
    match result {
        Err(ZgError::Limit(message)) => Ok(message),
        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
        Ok(value) => {
            drop(value);
            Err(format!("{label}: operation unexpectedly succeeded").into())
        }
    }
}

fn expected_cancel<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
    match result {
        Err(ZgError::State(message)) if message.contains("cancelled") => Ok(message),
        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
        Ok(value) => {
            drop(value);
            Err(format!("{label}: operation unexpectedly succeeded").into())
        }
    }
}

fn guards(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<Value> {
    let mut small_metadata = limits();
    small_metadata.max_metadata_bytes = 1;
    let metadata = expected_limit(
        SourceWeights::prepare_verified_windows(
            index,
            INDEX_SHA,
            expected_shards(shards)?,
            small_metadata,
            host,
        ),
        "metadata envelope",
    )?;
    ledger(host, 0, 0, 0)?;

    let mut source_limit = limits();
    source_limit.max_source_bytes = WEIGHT_BYTES - 1;
    let source = expected_limit(
        SourceWeights::prepare_verified_windows(
            index,
            INDEX_SHA,
            expected_shards(shards)?,
            source_limit,
            host,
        ),
        "whole-source file bound",
    )?;
    ledger(host, 0, 0, 0)?;

    let held_bytes = HOST - META + 1;
    let held = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: held_bytes,
        }],
    )?;
    let unavailable = expected_limit(
        SourceWeights::prepare_verified_windows(
            index,
            INDEX_SHA,
            expected_shards(shards)?,
            limits(),
            host,
        ),
        "metadata live-ledger admission",
    )?;
    ledger(host, 0, 0, held_bytes)?;
    drop(held);
    ledger(host, 0, 0, 0)?;

    let pending = plan(index, shards, host)?;
    ledger(host, META, 0, 0)?;
    let cancelled = expected_cancel(
        pending.open_cancellable(&AtomicBool::new(true)),
        "pre-cancelled factory open",
    )?;
    ledger(host, 0, 0, 0)?;
    Ok(json!({
        "metadata_envelope_refused":metadata,
        "source_file_bound_refused":source,
        "metadata_live_ledger_refused":unavailable,
        "pre_cancelled_factory_refused":cancelled,
        "scope":"real headers inspected; factory pre-cancel before whole payload hash scan; no source published",
        "post_guard_reserved_bytes":0,
    }))
}

fn expected_staging(shard_bytes: u64, offset: u64, bytes: usize) -> Proof<u64> {
    let end = offset
        .checked_add(bytes as u64)
        .ok_or("reference range overflow")?;
    ensure(
        bytes != 0 && end <= shard_bytes,
        "reference range outside source",
    )?;
    let first = offset / SCAN as u64;
    let last = (end - 1) / SCAN as u64;
    let staged = ((last + 1) * SCAN as u64).min(shard_bytes) - first * SCAN as u64;
    let offsets = (last - first + 1) * std::mem::size_of::<(usize, Range<usize>)>() as u64;
    Ok(staged + offsets)
}

fn compare_rows(
    source: &VerifiedTensorSource,
    matrix: &VerifiedMatrix,
    raw: &Shard,
    rows: Range<usize>,
    label: &str,
    host: &TransferAdmission,
) -> Proof<Value> {
    ledger(host, META, 0, 0)?;
    let offset = matrix.tensor().byte_offset + rows.start as u64 * matrix.columns() as u64 * 2;
    let bytes = (rows.end - rows.start) * matrix.columns() * 2;
    ensure(bytes <= 256 * 1024, "raw reference buffer cap")?;
    let window = source.read_rows(matrix, rows.clone())?;
    let charge = expected_staging(raw.expected_bytes, offset, bytes)?;
    ensure(
        window.charged_staging_bytes() as u64 == charge,
        "exact staged payload+offset charge",
    )?;
    ledger(host, META, charge, 0)?;
    ensure(
        window.row_range() == rows && window.source_offset() == offset,
        "window row/offset mismatch",
    )?;
    ensure(
        window.as_slice().len() == bytes,
        "window omitted full-width source row",
    )?;
    ensure(
        window.matrix().source_identity() == source.source_identity(),
        "window source identity changed",
    )?;
    ensure(
        window.verified_bundle_sha256() == BUNDLE_SHA,
        "window bundle changed",
    )?;
    ensure(
        window.shard_basename() == raw.name && window.shard_sha256() == raw.expected_sha,
        "window original shard binding changed",
    )?;
    ensure(
        window.extent_index_sha256().len() == 64,
        "extent receipt absent",
    )?;

    // The harness's independent comparison copy has a separate, explicit charge.
    let reference_lease = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: bytes as u64,
        }],
    )?;
    let mut reference = vec![0u8; bytes];
    raw.file.read_exact_at(&mut reference, offset)?;
    ledger(host, META, charge, bytes as u64)?;
    ensure(
        reference.as_slice() == window.as_slice(),
        "actual source row byte mismatch",
    )?;
    let record = json!({
        "matrix":matrix.tensor().name,
        "case":label,
        "rows":[rows.start,rows.end],
        "shape":[matrix.rows(),matrix.columns()],
        "dtype":"BF16",
        "axes":"OutputInput",
        "source_identity":format!("{:?}",source.source_identity()),
        "shard":raw.name,
        "shard_sha256":raw.expected_sha,
        "source_offset":offset,
        "requested_bytes":bytes,
        "charged_staging_bytes":charge,
        "crosses_scan_extent":offset / SCAN as u64 != (offset + bytes as u64 - 1) / SCAN as u64,
        "byte_exact":true,
        "raw_and_window_sha256":sha(&reference),
        "extent_index_sha256":window.extent_index_sha256(),
    });
    drop(reference);
    drop(reference_lease);
    ledger(host, META, charge, 0)?;
    drop(window);
    ledger(host, META, 0, 0)?;
    Ok(record)
}

fn exercise(
    source: VerifiedTensorSource,
    shards: &[Shard],
    host: &TransferAdmission,
) -> Proof<Value> {
    ensure(
        source.source_identity().is_bound(),
        "factory did not mint bound source identity",
    )?;
    let identity = source.source_identity();
    ensure(
        source.source_bytes() == WEIGHT_BYTES,
        "whole weight byte count",
    )?;
    ensure(
        source.metadata_reserved_bytes() == META,
        "source metadata charge",
    )?;
    ensure(
        source.verified_shard_index_sha256() == INDEX_SHA,
        "source index pin",
    )?;
    ensure(
        source.verified_bundle_sha256() == BUNDLE_SHA,
        "source bundle pin",
    )?;
    ledger(host, META, 0, 0)?;
    let clone = source.clone();
    ensure(
        clone.source_identity() == identity,
        "source clone minted another identity",
    )?;
    drop(clone);

    let specimens = [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            10_240,
            5_120,
            1_120_792,
            "model-00001-of-00018.safetensors",
        ),
        (
            "model.language_model.embed_tokens.weight",
            248_320,
            5_120,
            152,
            "model-00003-of-00018.safetensors",
        ),
        (
            "model.language_model.layers.63.mlp.down_proj.weight",
            5_120,
            17_408,
            1_364_259_120,
            "model-00017-of-00018.safetensors",
        ),
        (
            "lm_head.weight",
            248_320,
            5_120,
            1_760,
            "model-00018-of-00018.safetensors",
        ),
    ];
    let mut matrices = Vec::new();
    let mut samples = Vec::new();
    for (name, rows, columns, absolute_offset, basename) in specimens {
        let matrix = source.matrix(name, MatrixAxes::OutputInput)?;
        ensure(
            matrix.source_identity() == identity,
            "matrix changed source identity",
        )?;
        ensure(
            matrix.axes() == MatrixAxes::OutputInput,
            "matrix axes changed",
        )?;
        ensure(
            (matrix.rows(), matrix.columns()) == (rows, columns),
            "actual matrix shape differs from pinned header",
        )?;
        ensure(
            matrix.tensor().shape == [rows as u64, columns as u64],
            "raw source axes mismatch",
        )?;
        ensure(
            matrix.tensor().kind == QuantKind::Bf16,
            "raw source is not BF16",
        )?;
        ensure(
            matrix.tensor().byte_len == (rows * columns * 2) as u64,
            "raw source byte geometry",
        )?;
        ensure(
            matrix.tensor().byte_offset == absolute_offset,
            "actual source offset differs from independently inspected original header",
        )?;
        let raw = shards
            .iter()
            .find(|shard| shard.name == basename)
            .ok_or("missing raw source FD")?;
        samples.push(compare_rows(
            &source,
            &matrix,
            raw,
            0..1,
            "first_full_width_row",
            host,
        )?);
        samples.push(compare_rows(
            &source,
            &matrix,
            raw,
            rows - 1..rows,
            "last_full_width_row",
            host,
        )?);
        let start = matrix.tensor().byte_offset;
        let boundary = (start / SCAN as u64 + 1) * SCAN as u64;
        let split_row = ((boundary - start) / (columns * 2) as u64) as usize;
        let split = split_row.saturating_sub(1)..(split_row + 1).min(rows);
        let requested_start = start + split.start as u64 * (columns * 2) as u64;
        let requested_end = start + split.end as u64 * (columns * 2) as u64;
        ensure(
            requested_start < boundary && requested_end > boundary,
            "fixture does not straddle actual scan boundary",
        )?;
        samples.push(compare_rows(
            &source,
            &matrix,
            raw,
            split,
            "scan_extent_split_full_width_rows",
            host,
        )?);
        matrices.push(matrix);
    }

    let matrix = &matrices[0];
    let cancelled = expected_cancel(
        source.read_rows_cancellable(matrix, 0..1, &AtomicBool::new(true)),
        "pre-cancelled row",
    )?;
    ledger(host, META, 0, 0)?;
    ensure(
        source
            .matrix(&matrix.tensor().name, MatrixAxes::InputOutput)
            .is_err(),
        "wrong axes admitted",
    )?;
    ensure(
        source
            .matrix(
                "model.language_model.layers.0.input_layernorm.weight",
                MatrixAxes::OutputInput,
            )
            .is_err(),
        "actual rank-one parameter admitted as matrix",
    )?;
    ensure(
        source
            .matrix("__not_in_the_original_index__", MatrixAxes::OutputInput)
            .is_err(),
        "unknown matrix admitted",
    )?;
    for rows in [
        0..0,
        matrix.rows()..matrix.rows() + 1,
        usize::MAX..usize::MAX,
    ] {
        ensure(
            source.read_rows(matrix, rows).is_err(),
            "invalid row range admitted",
        )?;
        ledger(host, META, 0, 0)?;
    }
    let oversized = expected_limit(
        source.read_rows(&matrices[1], 0..matrices[1].rows()),
        "whole embedding tensor materialization",
    )?;
    ledger(host, META, 0, 0)?;

    let pin = matrices[0].clone();
    let last = &matrices[3];
    let retained = source.read_rows(last, last.rows() - 1..last.rows())?;
    let retained_sha = sha(retained.as_slice());
    let charge = retained.charged_staging_bytes() as u64;
    ledger(host, META, charge, 0)?;
    drop(matrices);
    drop(source);
    ensure(
        pin.source_identity() == identity && retained.matrix().source_identity() == identity,
        "descriptor/window lost original source identity",
    )?;
    ledger(host, META, charge, 0)?;
    drop(pin);
    // Only the owned immutable window retains source metadata and its live FD set.
    ledger(host, META, charge, 0)?;
    ensure(
        sha(retained.as_slice()) == retained_sha,
        "window bytes changed after all caller source handles dropped",
    )?;
    drop(retained);
    ledger(host, 0, 0, 0)?;
    Ok(json!({
        "source_identity":format!("{identity:?}"),
        "source_bytes":WEIGHT_BYTES,
        "source_bundle_sha256":BUNDLE_SHA,
        "index_sha256":INDEX_SHA,
        "matrix_count":4,
        "samples":samples,
        "pre_cancelled_row_refused":cancelled,
        "wrong_axes_missing_name_rank_one_bad_rows_refused":true,
        "whole_tensor_materialization_refused":oversized,
        "only_window_retains_source_metadata_and_stage_after_all_caller_sources_drop":true,
        "window_after_source_drop_sha256":retained_sha,
        "final_reserved_bytes":0,
    }))
}

fn run(host: &TransferAdmission) -> Proof<Value> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
    ensure(
        sha(&report_bytes) == REPORT_SHA,
        "acquired source report pin mismatch",
    )?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    ensure(
        report["snapshot"] == SNAPSHOT
            && report["repository"] == "Qwen/Qwen3.8-27B"
            && report["revision"] == REVISION
            && report["reference_directory"] == SOURCE_DIR,
        "acquired snapshot provenance differs",
    )?;
    ensure(
        report["selected_bytes"] == SELECTED_BYTES
            && report["weight_bytes"] == WEIGHT_BYTES
            && report["payload_bytes"] == PAYLOAD_BYTES
            && report["tensor_count"] == 1199,
        "acquired byte/count pin differs",
    )?;
    let files = report["files"]
        .as_array()
        .ok_or("source report files absent")?;
    ensure(files.len() == 31, "selected source file count")?;
    let mut pinned = BTreeMap::new();
    let mut total = 0u64;
    for entry in files {
        let name = entry["name"].as_str().ok_or("entry name absent")?;
        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
        total = total
            .checked_add(bytes)
            .ok_or("selected byte sum overflow")?;
        ensure(
            pinned
                .insert(name.to_string(), (bytes, hash.to_string()))
                .is_none(),
            "duplicate source report entry",
        )?;
    }
    ensure(
        total == SELECTED_BYTES,
        "selected report bytes do not close",
    )?;
    let index = bounded_file(
        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
        1024 * 1024,
    )?;
    ensure(
        sha(&index) == INDEX_SHA,
        "actual original index SHA256 mismatch",
    )?;
    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
    ensure(
        index_info.declared_total_size() == PAYLOAD_BYTES
            && index_info.tensors().len() == 1199
            && index_info.shards().len() == 18,
        "actual original index closure differs",
    )?;
    let mut shards = Vec::with_capacity(18);
    let mut descriptors = Vec::new();
    for number in 1..=18 {
        let name = format!("model-{number:05}-of-00018.safetensors");
        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
        ensure(
            index_info.shards().contains(name.as_str()),
            "pin has extra shard",
        )?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(Path::new(SOURCE_DIR).join(&name))?;
        let stat = file.metadata()?;
        ensure(
            stat.is_file() && stat.len() == *bytes,
            "opened original shard type/length differs",
        )?;
        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
        shards.push(Shard {
            name,
            expected_bytes: *bytes,
            expected_sha: hash.clone(),
            file,
        });
    }
    ensure(
        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
        "whole original weight-file byte sum differs",
    )?;
    eprintln!("phase=real_header_only_guards source_weight_bytes={WEIGHT_BYTES}");
    let refusals = guards(&index, &shards, host)?;
    let pending = plan(&index, &shards, host)?;
    ensure(
        pending.source_bytes() == WEIGHT_BYTES && pending.metadata_reserved_bytes() == META,
        "prepared whole-source bounds differ",
    )?;
    ledger(host, META, 0, 0)?;
    eprintln!(
        "phase=whole_source_sha256_scan_start shard_count=18 scan_buffer_bytes={SCAN} metadata_reserved_bytes={META}"
    );
    let started = Instant::now();
    let source = pending.open()?;
    let scan_milliseconds = started.elapsed().as_millis();
    eprintln!("phase=whole_source_sha256_scan_ready elapsed_ms={scan_milliseconds}");
    let windows = exercise(source, &shards, host)?;
    ledger(host, 0, 0, 0)?;
    Ok(json!({
        "schema":"zerogravity.original-owned-window-qualification.v1",
        "status":"PASS_SOURCE_WINDOWS_ONLY",
        "repository":"Qwen/Qwen3.8-27B",
        "revision":REVISION,
        "snapshot":SNAPSHOT,
        "selected_artifact_bytes":SELECTED_BYTES,
        "weight_file_bytes_scanned":WEIGHT_BYTES,
        "tensor_payload_bytes":PAYLOAD_BYTES,
        "whole_source_sha256_verified":true,
        "retained_raw_reference_fds":descriptors,
        "scan_chunk_bytes":SCAN,
        "whole_source_scan_milliseconds":scan_milliseconds,
        "configured_metadata_reserved_bytes":META,
        "configured_host_pool_limit_bytes":HOST,
        "refusals":refusals,
        "windows":windows,
        "physical_host_ram_admission_qualified":false,
        "trained_math_or_model_inference_qualified":false,
        "ssd_gpu_streaming_inference_qualified":false,
        "gpu_or_state_root_publication_performed":false,
        "scope":"one source issuer; whole weight SHA/index/header closure; owned bounded raw BF16 row bytes; configured host ledger only",
    }))
}

fn main() {
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "explicit-owned-window-qualification-pool")
        .expect("fixed resource label");
    arbiter
        .register_matching(resource.clone(), HOST)
        .expect("fresh explicit pool registration");
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new(
            "Qwen/Qwen3.8-27B",
            "actual-original-source-window-qualification",
        )
        .expect("fixed source owner"),
    };
    let result = run(&host);
    let final_state = host.arbiter.snapshot(&host.resource);
    let clean = matches!(&final_state, Ok(state) if state.reserved == 0 && state.reservations == 0 && !state.fenced);
    let success = result.is_ok() && clean;
    let mut report = match result {
        Ok(report) => report,
        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string()}),
    };
    report["final_owned_ledger_clean"] = json!(clean);
    report["final_owned_ledger_reserved_bytes"] =
        json!(final_state.as_ref().ok().map(|state| state.reserved));
    if !clean {
        report["status"] = json!("FAIL_UNQUALIFIED");
        report["ledger_cleanup_error"] = json!(format!("{final_state:?}"));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("bounded report serialization")
    );
    if !success {
        std::process::exit(2);
    }
}

/// Only this additive wrapper is called by the new replay. Original factory
/// source below remains retained verbatim; its old main/exercise are not called.
pub struct SourceReady {
    pub source: VerifiedTensorSource,
    pub config: Vec<u8>,
    pub descriptors: Vec<Value>,
    pub scan_milliseconds: u128,
}
pub fn open_source(host: &TransferAdmission) -> Proof<SourceReady> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
    ensure(
        sha(&report_bytes) == REPORT_SHA,
        "acquired source report pin mismatch",
    )?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    ensure(
        report["snapshot"] == SNAPSHOT
            && report["repository"] == "Qwen/Qwen3.8-27B"
            && report["revision"] == REVISION
            && report["reference_directory"] == SOURCE_DIR,
        "acquired snapshot provenance differs",
    )?;
    ensure(
        report["selected_bytes"] == SELECTED_BYTES
            && report["weight_bytes"] == WEIGHT_BYTES
            && report["payload_bytes"] == PAYLOAD_BYTES
            && report["tensor_count"] == 1199,
        "acquired byte/count pin differs",
    )?;
    let files = report["files"]
        .as_array()
        .ok_or("source report files absent")?;
    ensure(files.len() == 31, "selected source file count")?;
    let mut pinned = BTreeMap::new();
    let mut total = 0u64;
    for entry in files {
        let name = entry["name"].as_str().ok_or("entry name absent")?;
        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
        total = total
            .checked_add(bytes)
            .ok_or("selected byte sum overflow")?;
        ensure(
            pinned
                .insert(name.to_string(), (bytes, hash.to_string()))
                .is_none(),
            "duplicate source report entry",
        )?;
    }
    ensure(
        total == SELECTED_BYTES,
        "selected report bytes do not close",
    )?;
    let index = bounded_file(
        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
        1024 * 1024,
    )?;
    ensure(
        sha(&index) == INDEX_SHA,
        "actual original index SHA256 mismatch",
    )?;
    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
    ensure(
        index_info.declared_total_size() == PAYLOAD_BYTES
            && index_info.tensors().len() == 1199
            && index_info.shards().len() == 18,
        "actual original index closure differs",
    )?;
    let mut shards = Vec::with_capacity(18);
    let mut descriptors = Vec::new();
    for number in 1..=18 {
        let name = format!("model-{number:05}-of-00018.safetensors");
        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
        ensure(
            index_info.shards().contains(name.as_str()),
            "pin has extra shard",
        )?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(Path::new(SOURCE_DIR).join(&name))?;
        let stat = file.metadata()?;
        ensure(
            stat.is_file() && stat.len() == *bytes,
            "opened original shard type/length differs",
        )?;
        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
        shards.push(Shard {
            name,
            expected_bytes: *bytes,
            expected_sha: hash.clone(),
            file,
        });
    }
    ensure(
        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
        "whole original weight-file byte sum differs",
    )?;
    let config = bounded_file(&Path::new(SOURCE_DIR).join("config.json"), 64 * 1024)?;
    ensure(
        sha(&config) == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
        "original config SHA",
    )?;
    let pending = plan(&index, &shards, host)?;
    let started = Instant::now();
    let source = pending.open()?;
    ensure(
        source.source_bytes() == WEIGHT_BYTES
            && source.metadata_reserved_bytes() == META
            && source.verified_shard_index_sha256() == INDEX_SHA
            && source.verified_bundle_sha256() == BUNDLE_SHA,
        "source factory closure",
    )?;
    Ok(SourceReady {
        source,
        config,
        descriptors,
        scan_milliseconds: started.elapsed().as_millis(),
    })
}

````

