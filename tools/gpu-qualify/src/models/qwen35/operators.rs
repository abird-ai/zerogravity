//! Physical remaining-operator qualification. Constructed arithmetic vectors
//! are separate from genuine source-model causal token sweeps.
use std::path::Path;
use zerogravity_core::accelerator::{BinaryOperation, BinaryRegion, MatVecKernel, RegionExecution};
use zerogravity_core::resource_arbiter::ResourceOwner;
use zerogravity_cpu::source::source_reference_embedding;
use zerogravity_gpu::CubeResidentProvider;
use zerogravity_gpu::Device;
use zerogravity_model_qwen35::{LoadedQwenModel, source_binary};

pub fn run(device: Device, path: &Path) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model parent")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let provider = CubeResidentProvider::new(
        device,
        [
            "add.stress".into(),
            "sigmoid.stress".into(),
            "embedding".into(),
        ],
        300_000,
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        ResourceOwner::new("remaining-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut compared = 0;
    let mut binary_regions = 0;
    for n in [1usize, 3, 7, 8, 31, 32, 33, 1024, 4096, 65536] {
        for kind in [BinaryOperation::Add, BinaryOperation::SigmoidGate] {
            let left: Vec<f32> = (0..n)
                .map(|i| match i % 11 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::from_bits(1),
                    3 => -f32::from_bits(1),
                    4 => f32::MIN_POSITIVE,
                    5 => -f32::MIN_POSITIVE,
                    6 => f32::MAX,
                    7 => -f32::MAX,
                    _ => ((i * 17 % 12345) as f32 - 6000.0) / 1024.0,
                })
                .collect();
            let right: Vec<f32> = (0..n)
                .map(|i| {
                    if kind == BinaryOperation::Add {
                        match i % 11 {
                            0 => -0.0,
                            1 => 0.0,
                            2 => -f32::from_bits(1),
                            3 => f32::from_bits(2),
                            4 => -f32::MIN_POSITIVE,
                            5 => f32::MIN_POSITIVE,
                            6 => -f32::MAX,
                            7 => f32::MAX,
                            _ => ((i * 23 % 4789) as f32 - 2048.0) / 2048.0,
                        }
                    } else {
                        [
                            -f32::MAX,
                            -128.0,
                            -104.0,
                            -103.0,
                            -90.0,
                            -88.72284,
                            -80.0,
                            -1.0,
                            -f32::from_bits(1),
                            -0.0,
                            0.0,
                            f32::from_bits(1),
                            1.0,
                            80.0,
                            103.0,
                            128.0,
                            f32::MAX,
                        ][i % 17]
                    }
                })
                .collect();
            let r = BinaryRegion {
                source_identity: model.source_identity(),
                operation: if kind == BinaryOperation::Add {
                    "add.stress"
                } else {
                    "sigmoid.stress"
                },
                kind,
                left: &left,
                right: &right,
            };
            let expected = source_binary(r).map_err(|e| e.to_string())?;
            let RegionExecution::Completed(actual) =
                provider.try_binary(r).map_err(|e| e.to_string())?
            else {
                return Err("selected binary Unsupported".into());
            };
            compare(&expected, &actual, r.operation)?;
            compared += actual.len();
            binary_regions += 1;
        }
    }
    let mut embedding_regions = 0;
    let vocabulary = model
        .embedding_source(0)
        .map_err(|e| e.to_string())?
        .vocabulary;
    for token in [
        0,
        11,
        17,
        42,
        127,
        128,
        255,
        256,
        353,
        32767,
        65535,
        vocabulary as u32 - 1,
    ] {
        let r = model.embedding_source(token).map_err(|e| e.to_string())?;
        let expected = source_reference_embedding(r).map_err(|e| e.to_string())?;
        let RegionExecution::Completed(actual) =
            provider.try_embedding(r).map_err(|e| e.to_string())?
        else {
            return Err("selected embedding Unsupported".into());
        };
        compare(&expected, &actual, "embedding")?;
        compared += actual.len();
        embedding_regions += 1;
    }
    println!(
        "P07_H13_REMAINING_OPERATOR_STRESS_PASS binary_regions={binary_regions} embedding_regions={embedding_regions} compared_fp32={compared} source_bits=1 synthetic_arithmetic_not_model_context=1"
    );
    Ok(())
}
fn compare(expected: &[f32], actual: &[f32], name: &str) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!("{name} length"));
    }
    if let Some(i) = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!(
            "{name} bits offset={i} expected={:08x} actual={:08x}",
            expected[i].to_bits(),
            actual[i].to_bits()
        ));
    }
    Ok(())
}
