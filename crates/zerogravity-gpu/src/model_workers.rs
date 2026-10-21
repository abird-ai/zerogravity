//! P07-G2b opt-in real physical two-model CubeCL GPU worker qualification.
//! Source GGUF tensors and runtime quant math originate from one Rust engine.
//! Two different original checkpoints run on distinct CPU request threads,
//! sharing ONE physical-card fair GPU worker queue and memory budget.
//! This is a physical Q4_0 region test, NOT full GPU language-model execution.

use crate::{resident_provider::CubeResidentProvider, source_accumulation::SourceAccumulation};
use cubecl::Device;
use std::{
    path::Path,
    sync::{Arc, Barrier},
};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result as ZgResult;
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::resource_arbiter::{AllocationClass, ResourceArbiter, ResourceOwner};
use zerogravity_cpu::weights::CpuWeights;

const TENSOR: &str = "blk.0.attn_gate.weight";
const CARD_CAP: u64 = 512 * 1024 * 1024;
const PROVIDER_CAP: usize = 256 * 1024 * 1024;

fn tensor_activation(weights: &CpuWeights, salt: usize) -> Result<Vec<f32>, String> {
    let tensor = weights.tensor(TENSOR).map_err(|e| e.to_string())?;
    if tensor.kind != QuantKind::Q4_0 || tensor.shape.len() != 2 {
        return Err("physical worker checkpoint tensor is not a Q4_0 matrix".into());
    }
    let cols = usize::try_from(tensor.shape[0]).map_err(|_| "tensor width overflow")?;
    Ok((0..cols)
        .map(|i| (((i * 113 + salt * 31) % 997) as f32 - 498.0) / 371.0)
        .collect())
}

fn checked_parity(
    reference: &[f32],
    actual: &[f32],
    model: &str,
    step: usize,
) -> Result<(), String> {
    if actual.len() != reference.len() {
        return Err(format!(
            "{model}: physical output length mismatch at step {step}"
        ));
    }
    let differing = reference
        .iter()
        .zip(actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = reference
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "P07_G2B_REAL_GPU_WORKER_SOURCE model={model} step={step} rows={} bit_differences={differing} max_abs={max_abs}",
        actual.len()
    );
    if differing != 0 {
        return Err(format!(
            "{model}: source-numeric GPU mismatch on physical card; bit_differences={differing}, max_abs={max_abs}"
        ));
    }
    Ok(())
}

/// Source-independent provider rejection check against a genuine trained GGUF
/// region. Neither a too-small envelope nor a malformed activation may
/// submit GPU work, fence an idle card or poison another model's admission.
fn validate_pre_submission_rejections(
    device: &Device,
    weights: &CpuWeights,
    input: &[f32],
    expected: &[f32],
) -> Result<(), String> {
    let arbiter = ResourceArbiter::process_shared();
    let tiny = CubeResidentProvider::new(
        device.clone(),
        [TENSOR.to_owned()],
        8192,
        1_100_000, // Qwen0.8B source fits; source+scratch cannot.
        CARD_CAP,
        ResourceOwner::new("qwen35-review-budget", "too-small").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let card = tiny.physical_key();
    if !matches!(
        weights.matvec_with(TENSOR, input, Some(&tiny)),
        Err(Error::Limit(_))
    ) {
        return Err("GPU preflight did not reject too-small whole working set".into());
    }
    let view = arbiter.snapshot(&card).map_err(|e| e.to_string())?;
    let stats = tiny.worker_stats().map_err(|e| e.to_string())?;
    if view.fenced
        || view.reserved != 1_100_000
        || stats.selected != 0
        || stats.completed != 0
        || tiny
            .quarantined_physical_handles()
            .map_err(|e| e.to_string())?
            != 0
    {
        return Err("refused unsatisfiable source/scratch polluted physical GPU".into());
    }
    drop(tiny);
    if arbiter.snapshot(&card).map_err(|e| e.to_string())?.reserved != 0 {
        return Err("safe rejected source allocation retained its global charge".into());
    }

    struct CorruptActivation<'a>(&'a CubeResidentProvider);
    impl MatVecKernel for CorruptActivation<'_> {
        fn try_execute(&self, region: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
            if region.tensor_name != TENSOR {
                return Ok(RegionExecution::Unsupported);
            }
            // Rust Qwen normally rejects NaN before the kernel boundary:
            // the wrapper deliberately corrupts it *at* the GPU interface
            // to prove the provider itself refuses it pre-submission.
            let mut invalid = region.activation.to_vec();
            invalid[0] = f32::NAN;
            self.0.try_execute(MatVecRegion {
                activation: &invalid,
                ..region
            })
        }
    }

    let normal = CubeResidentProvider::new(
        device.clone(),
        [TENSOR.to_owned()],
        8192,
        2 * 1024 * 1024,
        CARD_CAP,
        ResourceOwner::new("qwen35-review-budget", "recoverable-error")
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !matches!(
        weights.matvec_with(TENSOR, input, Some(&CorruptActivation(&normal))),
        Err(Error::Malformed(_))
    ) {
        return Err("GPU failed to reject corrupted activation before submission".into());
    }
    struct CorruptScale<'a>(&'a CubeResidentProvider);
    impl MatVecKernel for CorruptScale<'_> {
        fn try_execute(&self, region: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
            if region.tensor_name != TENSOR {
                return Ok(RegionExecution::Unsupported);
            }
            let mut invalid = region.packed_weights.to_vec();
            // The first Q4_0 block's FP16 weight scale becomes NaN. This
            // must fail source validation before creating a GPU buffer.
            invalid[0] = 0;
            invalid[1] = 0x7e;
            self.0.try_execute(MatVecRegion {
                packed_weights: &invalid,
                ..region
            })
        }
    }
    if !matches!(
        weights.matvec_with(TENSOR, input, Some(&CorruptScale(&normal))),
        Err(Error::Malformed(_))
    ) {
        return Err("GPU source NaN scale was not rejected before submission".into());
    }
    let after = arbiter.snapshot(&card).map_err(|e| e.to_string())?;
    let before = normal.worker_stats().map_err(|e| e.to_string())?;
    if after.fenced
        || before.selected != 1 // source scale error was dequeued, not GPU-launched
        || before.completed != 0
        || normal.stats().map_err(|e| e.to_string())?.source_uploads != 0
        || normal.quarantined_physical_handles().map_err(|e| e.to_string())? != 0
    {
        return Err("invalid input incorrectly poisoned an unused physical GPU".into());
    }
    // A valid submission on the SAME provider/card must still complete and
    // match the original trained source bit-for-bit after both refusals.
    let actual = weights
        .matvec_with(TENSOR, input, Some(&normal))
        .map_err(|e| format!("recoverable GPU preflight error poisoned later use: {e}"))?;
    checked_parity(expected, &actual, "qwen35-preflight-recovery", 0)?;
    let complete = normal.worker_stats().map_err(|e| e.to_string())?;
    if complete.completed != 1 || complete.active || complete.queued != 0 {
        return Err("GPU failed real readback after recoverable preflight refusals".into());
    }
    drop(normal);
    let released = arbiter.snapshot(&card).map_err(|e| e.to_string())?;
    if released.fenced || released.reserved != 0 {
        return Err("preflight-review GPU did not safely retire normal source".into());
    }
    println!(
        "P07_REVIEW_PRE_SUBMISSION_REFUSALS_SAFE card={card:?} refused_capacity=1 refused_activation_nan=1 refused_source_scale_nan=1 subsequent_real_gpu_readback=1"
    );
    Ok(())
}

pub fn run(
    device: Device,
    model_08: &Path,
    model_2b: &Path,
    native_f32_diagnostic: bool,
) -> Result<(), String> {
    let accumulation = if native_f32_diagnostic {
        SourceAccumulation::NativeF32
    } else {
        SourceAccumulation::WideF64
    };
    let weights_08 = CpuWeights::open(model_08).map_err(|e| e.to_string())?;
    let weights_2b = CpuWeights::open(model_2b).map_err(|e| e.to_string())?;
    let input_08 = tensor_activation(&weights_08, 0)?;
    let input_2b = tensor_activation(&weights_2b, 1)?;
    // An actual source reference, not an idealized F32 dequantization.
    let ref_08 = weights_08
        .matvec(TENSOR, &input_08)
        .map_err(|e| e.to_string())?;
    let ref_2b = weights_2b
        .matvec(TENSOR, &input_2b)
        .map_err(|e| e.to_string())?;
    validate_pre_submission_rejections(&device, &weights_08, &input_08, &ref_08)?;
    let provider_08 = Arc::new(
        CubeResidentProvider::new(
            device.clone(),
            [TENSOR.to_string()],
            8192,
            PROVIDER_CAP,
            CARD_CAP,
            ResourceOwner::new("qwen35-0.8b", "worker-test").map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .with_source_diagnostic_accumulation(accumulation),
    );
    let provider_2b = Arc::new(
        CubeResidentProvider::new(
            device,
            [TENSOR.to_string()],
            8192,
            PROVIDER_CAP,
            CARD_CAP,
            ResourceOwner::new("qwen35-2b", "worker-test").map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .with_source_diagnostic_accumulation(accumulation),
    );

    let key = provider_08.physical_key();
    if provider_2b.physical_key() != key {
        return Err("two GPU providers reported different physical cards".into());
    }
    let arbiter = ResourceArbiter::process_shared();
    let charged = arbiter.snapshot(&key).map_err(|e| e.to_string())?;
    if charged.limit != CARD_CAP
        || charged.reserved != CARD_CAP
        || charged.reservations != 2
        || charged.per_class.get(&AllocationClass::ProviderEnvelope) != Some(&CARD_CAP)
        || charged.active_models.len() != 2
    {
        return Err(format!(
            "physical shared-provider admission accounting invalid: {charged:?}"
        ));
    }
    let barrier = Barrier::new(3);
    let (result_08, result_2b) = std::thread::scope(|scope| {
        let barrier_08 = &barrier;
        let provider_ref_08 = &provider_08;
        let task_08 = scope.spawn(move || -> Result<(), String> {
            barrier_08.wait();
            for step in 0..2 {
                let actual = weights_08
                    .matvec_with(TENSOR, &input_08, Some(provider_ref_08.as_ref()))
                    .map_err(|e| e.to_string())?;
                checked_parity(&ref_08, &actual, "qwen35-0.8b", step)?;
            }
            Ok(())
        });
        let barrier_2b = &barrier;
        let provider_ref_2b = &provider_2b;
        let task_2b = scope.spawn(move || -> Result<(), String> {
            barrier_2b.wait();
            for step in 0..2 {
                let actual = weights_2b
                    .matvec_with(TENSOR, &input_2b, Some(provider_ref_2b.as_ref()))
                    .map_err(|e| e.to_string())?;
                checked_parity(&ref_2b, &actual, "qwen35-2b", step)?;
            }
            Ok(())
        });
        barrier.wait();
        (task_08.join(), task_2b.join())
    });
    // All device tasks have joined. Audit actual readback and physical
    // account cleanup even when numerical qualification fails (e.g. AMD RADV).
    let error_08 = result_08
        .map_err(|_| "0.8B GPU worker panicked".to_owned())?
        .err();
    let error_2b = result_2b
        .map_err(|_| "2B GPU worker panicked".to_owned())?
        .err();
    let numerical_pass = error_08.is_none() && error_2b.is_none();
    let worker = provider_08.worker_stats().map_err(|e| e.to_string())?;
    if worker != provider_2b.worker_stats().map_err(|e| e.to_string())?
        || worker.selected != if numerical_pass { 4 } else { worker.completed }
        || worker.active
        || worker.queued != 0
        || (numerical_pass && worker.completed != 4)
    {
        return Err(format!(
            "cross-model GPU worker completion/fair queue failed: {worker:?}"
        ));
    }
    for (name, provider) in [("0.8b", &provider_08), ("2b", &provider_2b)] {
        let stats = provider.stats().map_err(|e| e.to_string())?;
        if stats.source_uploads != 1
            || (numerical_pass && (stats.source_cache_hits != 1 || stats.completed_regions != 2))
            || provider
                .in_flight_device_bytes()
                .map_err(|e| e.to_string())?
                != 0
            || provider
                .quarantined_physical_handles()
                .map_err(|e| e.to_string())?
                != 0
        {
            return Err(format!(
                "{name}: GPU residency/cache/event safety failed: {stats:?}"
            ));
        }
    }
    drop(provider_08);
    drop(provider_2b);
    let freed = arbiter.snapshot(&key).map_err(|e| e.to_string())?;
    if freed.reserved != 0 || freed.fenced {
        return Err(format!(
            "completed physical source workers retained a stale charge: {freed:?}"
        ));
    }
    if !numerical_pass {
        println!(
            "P07_G2B_REAL_GPU_SOURCE_NUMERICALLY_UNQUALIFIED card={key:?} physical_completed={} bytes_reserved_after_teardown=0",
            worker.completed
        );
        return Err(format!(
            "real physical GPU workers completed/readback/retired safely but source numerics failed: 0.8B={error_08:?}, 2B={error_2b:?}"
        ));
    }
    println!(
        "P07_G3_REAL_GPU_TWO_MODELS_SOURCE_EXACT strategy={accumulation:?} card={key:?} shared_envelopes=2 fair_choices={} physically_completed={} resident_uploads=2",
        worker.selected, worker.completed
    );
    Ok(())
}
