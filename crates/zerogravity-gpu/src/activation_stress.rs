//! Original trained source matrices and independent physical quant intermediates.
use crate::resident_provider::CubeResidentProvider;
use cubecl::Device;
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Result as ZgResult;
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::resource_arbiter::ResourceOwner;
use zerogravity_cpu::weights::source_reference_projection;
use zerogravity_model_qwen35::LoadedQwenModel;
struct Sample {
    kind: QuantKind,
    columns: usize,
    source: Vec<u8>,
    activation: Vec<f32>,
}
struct Capture(Mutex<BTreeMap<QuantKind, Sample>>);
impl MatVecKernel for Capture {
    fn try_execute(&self, r: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        if r.kind == QuantKind::F32 {
            return Ok(RegionExecution::Unsupported);
        }
        let mut samples = self
            .0
            .lock()
            .map_err(|_| zerogravity_core::Error::State("quant capture lock".into()))?;
        if let std::collections::btree_map::Entry::Vacant(entry) = samples.entry(r.kind) {
            let bytes = r.kind.byte_len(r.columns as u64)? as usize * 8;
            if r.rows >= 8 {
                entry.insert(Sample {
                    kind: r.kind,
                    columns: r.columns,
                    source: r.packed_weights[..bytes].to_vec(),
                    activation: r.activation.to_vec(),
                });
            }
        }
        Ok(RegionExecution::Unsupported)
    }
}
pub fn run(device: Device, path: &Path) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model parent")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let capture = Arc::new(Capture(Mutex::new(BTreeMap::new())));
    model
        .new_session()
        .map_err(|e| e.to_string())?
        .with_kernel(capture.clone())
        .prefill_transaction(&[11, 353, 17, 42])
        .map_err(|e| e.to_string())?;
    let samples = std::mem::take(&mut *capture.0.lock().map_err(|_| "quant capture lock")?);
    let names: Vec<_> = samples
        .keys()
        .map(|k| format!("dynamic.stress.{k:?}"))
        .collect();
    let provider = CubeResidentProvider::new(
        device.clone(),
        names,
        8,
        16 * 1024 * 1024,
        1024 * 1024 * 1024,
        ResourceOwner::new("dynamic-quant-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?
    .with_device_activation_quantization(true);
    let first = samples
        .values()
        .next()
        .ok_or("missing original quant samples")?;
    let name = format!("dynamic.stress.{:?}", first.kind);
    let poor = CubeResidentProvider::new(
        device,
        [name.clone()],
        8,
        512,
        1024 * 1024 * 1024,
        ResourceOwner::new("dynamic-underbudget", "preflight").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?
    .with_device_activation_quantization(true);
    let r = MatVecRegion {
        source_identity: model.source_identity(),
        tensor_name: &name,
        kind: first.kind,
        rows: 8,
        columns: first.columns,
        packed_weights: &first.source,
        activation: &first.activation,
    };
    if !matches!(poor.try_execute(r), Err(zerogravity_core::Error::Limit(_)))
        || poor.worker_stats().map_err(|e| e.to_string())?.completed != 0
        || poor.is_fenced().map_err(|e| e.to_string())?
    {
        return Err("device quantization underbudget preflight failed".into());
    }
    let mut regions = 0;
    let mut fp32 = 0;
    let mut codes = 0;
    for sample in samples.values() {
        for cohort in 0..9 {
            let input: Vec<f32> = (0..sample.columns)
                .map(|i| match cohort {
                    0 => sample.activation[i],
                    1 => [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5][i % 8],
                    2 => [-127.0, 127.0, -0.5, 0.5, -1.5, 1.5, -126.5, 126.5][i % 8],
                    3 => {
                        if i % 2 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    4 => [f32::from_bits(1), -f32::from_bits(1), 0.0, -0.0][i % 4],
                    5 => {
                        if i % 2 == 0 {
                            f32::MIN_POSITIVE
                        } else {
                            -f32::MIN_POSITIVE
                        }
                    }
                    6 => {
                        if i % 2 == 0 {
                            127.0 * 2.0f32.powi(-24)
                        } else {
                            -127.0 * 2.0f32.powi(-24)
                        }
                    }
                    7 => {
                        if i % 2 == 0 {
                            65504.0 * 127.0
                        } else {
                            -65504.0 * 127.0
                        }
                    }
                    _ => ((i * 137 % 25401) as f32 - 12700.0) / 100.0,
                })
                .collect();
            let name = format!("dynamic.stress.{:?}", sample.kind);
            let r = MatVecRegion {
                source_identity: model.source_identity(),
                tensor_name: &name,
                kind: sample.kind,
                rows: 8,
                columns: sample.columns,
                packed_weights: &sample.source,
                activation: &input,
            };
            let counts = futures_lite::future::block_on(provider.qualify_device_activation(r))
                .map_err(|e| format!("{name} cohort{cohort} {e}"))?;
            fp32 += counts.0;
            codes += counts.1;
            let expected = source_reference_projection(r).map_err(|e| e.to_string())?;
            let RegionExecution::Completed(actual) =
                provider.try_execute(r).map_err(|e| e.to_string())?
            else {
                return Err("quant GEMV Unsupported".into());
            };
            if expected.len() != actual.len()
                || expected
                    .iter()
                    .zip(&actual)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!("quant GEMV bits {name} cohort{cohort}"));
            }
            fp32 += actual.len();
            regions += 1;
        }
    }
    if provider
        .stats()
        .map_err(|e| e.to_string())?
        .device_quantized_regions
        != regions
    {
        return Err("missing device quantization Ready".into());
    }
    println!(
        "P07_H14_SOURCE_GPU_QUANT_STRESS_PASS codecs={} cohorts=9 regions={regions} compared_fp32={fp32} compared_codes={codes} no_intermediate_model_readback=1 source_bits=1",
        samples.len()
    );
    Ok(())
}
