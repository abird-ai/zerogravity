//! Developer-only scalar emulation of the GGML Haswell/AVX2 FMA SiLU
//! approximation for source-output diagnosis. Not part of model execution.
use std::{collections::HashMap, env, fs, path::Path};
use zerogravity::{math::silu, weights::CpuWeights};
type Traces = HashMap<(usize, String), Vec<f32>>;
fn read(path: &str, source: bool) -> Traces {
    let mut out = Traces::new();
    for line in fs::read_to_string(path).unwrap().lines() {
        let r: serde_json::Value = serde_json::from_str(line).unwrap();
        let pos = r["position"].as_u64().unwrap() as usize;
        let stage = if source {
            r["name"].as_str().unwrap()
        } else {
            r["stage"].as_str().unwrap()
        };
        let v = r["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        out.insert((pos, stage.to_string()), v);
    }
    out
}
fn ggml_approx_exp(x: f32) -> f32 {
    if !(-80.0..80.0).contains(&x) {
        return x.exp();
    }
    let r = f32::from_bits(0x4b400000);
    let z = x.mul_add(f32::from_bits(0x3fb8aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(
        f32::from_bits(0x35bfbe8e),
        (-n).mul_add(f32::from_bits(0x3f317200), x),
    );
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(0x3f800000));
    let u = b * b;
    let j = f32::from_bits(0x3c072010)
        .mul_add(b, f32::from_bits(0x3d2b9f17))
        .mul_add(
            u,
            f32::from_bits(0x3e2aaf33).mul_add(b, f32::from_bits(0x3efffedb)),
        )
        .mul_add(u, f32::from_bits(0x3f7ffff6) * b);
    j.mul_add(k, k)
}
fn simd_silu(x: f32) -> f32 {
    x / (1.0 + ggml_approx_exp(0.0 - x))
}
fn norm(core: &[f32], weight: &[f32]) -> Vec<f32> {
    core.chunks_exact(128)
        .flat_map(|head| {
            let sum = head.iter().map(|&v| f64::from(v * v)).sum::<f64>();
            let mean = (sum / 128.0) as f32;
            let inv = (mean + 1e-6).sqrt().recip();
            head.iter().zip(weight).map(move |(&v, &w)| (v * inv) * w)
        })
        .collect()
}
fn diagnose(label: &str, candidate: &[f32], reference: &[f32]) {
    let errors = candidate
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();
    let max = errors.iter().copied().fold(0.0, f32::max);
    let rms =
        (errors.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
    let quant = |data: &[f32]| -> Vec<i8> {
        data.chunks_exact(32)
            .flat_map(|block| {
                let max = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let inv = if max == 0.0 { 0.0 } else { 127.0 / max };
                block.iter().map(move |x| (x * inv).round_ties_even() as i8)
            })
            .collect()
    };
    let ours = quant(candidate);
    let theirs = quant(reference);
    let swaps = ours
        .iter()
        .zip(theirs)
        .enumerate()
        .filter(|(_, (a, b))| **a != *b)
        .map(|(i, (_, _))| i)
        .collect::<Vec<_>>();
    println!(
        "{label:45}: MAX={max:.10} RMS={rms:.10} q8_flips={} indices={:?}",
        swaps.len(),
        &swaps[..swaps.len().min(8)]
    );
}
fn main() {
    let args = env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 4, "model.gguf source.jsonl rust.jsonl");
    let source = read(&args[2], true);
    let rust = read(&args[3], false);
    let weights = CpuWeights::open(Path::new(&args[1])).unwrap();
    let norm_weight = weights.vector("blk.0.ssm_norm.weight").unwrap();
    for pos in 0..6 {
        let core_ref = &source[&(pos, "attn_output-0".to_string())];
        let core_rust = &rust[&(pos, "gdn_core_output".to_string())];
        let z_ref = &source[&(pos, "z-0".to_string())];
        let z_rust = weights
            .matvec(
                "blk.0.attn_gate.weight",
                &rust[&(pos, "attention_input_norm".to_string())],
            )
            .unwrap();
        let expected = &source[&(pos, "final_output-0".to_string())];
        let baseline = &rust[&(pos, "gdn_gated_output".to_string())];
        println!("POS={pos}");
        diagnose("baseline runtime SiLU", baseline, expected);
        for (label, core, z, fast) in [
            (
                "source core/z accurate exp",
                core_ref.as_slice(),
                z_ref.as_slice(),
                false,
            ),
            (
                "source core/z ggml fast exp",
                core_ref.as_slice(),
                z_ref.as_slice(),
                true,
            ),
            (
                "rust core + source z ggml fast exp",
                core_rust.as_slice(),
                z_ref.as_slice(),
                true,
            ),
            (
                "rust core + rust z ggml fast exp",
                core_rust.as_slice(),
                z_rust.as_slice(),
                true,
            ),
        ] {
            let result = norm(core, &norm_weight);
            let candidate = result
                .iter()
                .zip(z)
                .map(|(&n, &g)| n * (if fast { simd_silu(g) } else { silu(g) }))
                .collect::<Vec<_>>();
            diagnose(label, &candidate, expected);
        }
    }
}
