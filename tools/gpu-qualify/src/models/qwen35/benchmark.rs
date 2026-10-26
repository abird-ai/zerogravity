//! Reproducible, source-audited physical Q4 GPU region timing experiment.
//!
//! Timing includes CubeCL submission, source resident cache lookup, dynamic
//! activation quant/transfer and genuine readback, NOT a complete Qwen token.
//! Output equality is checked for EVERY real device sample. Diagnostic-only,
//! never used as an unchecked runtime auto-selection or benchmarked model TPS.
use std::{path::Path, time::Instant};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::resource_arbiter::{ResourceArbiter, ResourceOwner};
use zerogravity_cpu::source::CpuWeights;
use zerogravity_gpu::CubeResidentProvider;
use zerogravity_gpu::Device;
use zerogravity_gpu::qualification::{
    SourceAccumulation, read_driver_memory, report_driver_memory,
};

const TENSOR: &str = "blk.0.attn_gate.weight";
const CARD_CAP: u64 = 512 * 1024 * 1024;

pub fn run(device: Device, model: &Path) -> Result<(), String> {
    let weights = CpuWeights::open(model).map_err(|e| e.to_string())?;
    let tensor = weights.tensor(TENSOR).map_err(|e| e.to_string())?;
    if tensor.kind != QuantKind::Q4_0 || tensor.shape.len() != 2 {
        return Err("source-trained Q4_0 measurement tensor missing".into());
    }
    let cols = usize::try_from(tensor.shape[0]).map_err(|_| "source columns overflow")?;
    let input = (0..cols)
        .map(|i| ((i.wrapping_mul(113) % 997) as f32 - 498.0) / 371.0)
        .collect::<Vec<_>>();
    let cpu_source = weights.matvec(TENSOR, &input).map_err(|e| e.to_string())?;
    let key = {
        let probe = CubeResidentProvider::new(
            device.clone(),
            [TENSOR.to_owned()],
            8192,
            256 * 1024 * 1024,
            CARD_CAP,
            ResourceOwner::new("qwen-benchmark", "identify-card").map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        probe.physical_key()
    };
    let baseline = read_driver_memory(&key);
    report_driver_memory(&key, "before_gpu_source", baseline);
    for strategy in [SourceAccumulation::NativeF32, SourceAccumulation::WideF64] {
        let backend = CubeResidentProvider::new(
            device.clone(),
            [TENSOR.to_owned()],
            8192,
            256 * 1024 * 1024,
            CARD_CAP,
            ResourceOwner::new(
                "qwen-benchmark",
                match strategy {
                    SourceAccumulation::NativeF32 => "native-f32",
                    SourceAccumulation::WideF64 => "wide-f64",
                },
            )
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .with_source_diagnostic_accumulation(strategy);
        let mut times = Vec::with_capacity(16);
        let mut mismatched = 0usize;
        let mut maximum_error = 0f32;
        // 2 warmed original CubeCL submissions, then 16 full readback samples.
        for iteration in 0..18 {
            let begin = Instant::now();
            let actual = weights
                .matvec_with(TENSOR, &input, Some(&backend))
                .map_err(|e| e.to_string())?;
            let elapsed = begin.elapsed();
            if actual.len() != cpu_source.len() {
                return Err("GPU returned wrong source row count".into());
            }
            for (&reference, &observed) in cpu_source.iter().zip(actual.iter()) {
                mismatched += usize::from(reference.to_bits() != observed.to_bits());
                maximum_error = maximum_error.max((reference - observed).abs());
            }
            if iteration >= 2 {
                times.push(elapsed.as_micros());
            } else if iteration == 1 {
                report_driver_memory(&key, "after_gpu_source_warmup", baseline);
            }
        }
        times.sort_unstable();
        let middle = times[times.len() / 2];
        let p95 = times[(times.len() * 95 / 100).min(times.len() - 1)];
        let min = times[0];
        let source_quality = if mismatched == 0 {
            "EXACT"
        } else {
            "UNQUALIFIED"
        };
        println!(
            "P07_PHYSICAL_GPU_Q4_LATENCY card={key:?} strategy={strategy:?} rows={} cols={} warmup=2 measured=16 min_us={min} median_us={middle} p95_us={p95} total_mismatched_f32_bits={mismatched} max_abs_error={maximum_error:.9} source_quality={source_quality} scope=source_region_h2d_kernel_readback_not_model_tps",
            tensor.shape[1], tensor.shape[0]
        );
        if strategy == SourceAccumulation::WideF64 && mismatched != 0 {
            return Err(
                "strict wide-f64 source-region numerics failed on this physical GPU".into(),
            );
        }
        let source_stats = backend.stats().map_err(|e| e.to_string())?;
        if source_stats.source_uploads != 1 || source_stats.completed_regions != 18 {
            return Err(format!(
                "bench did not reuse one resident GPU source: {source_stats:?}"
            ));
        }
        report_driver_memory(&key, "after_16_real_readbacks_before_teardown", baseline);
        drop(backend);
        report_driver_memory(&key, "after_verified_teardown", baseline);
        let budget = ResourceArbiter::process_shared()
            .snapshot(&key)
            .map_err(|e| e.to_string())?;
        if budget.fenced || budget.reserved != 0 {
            return Err("benchmark retained physical memory after verified readback".into());
        }
    }
    Ok(())
}
