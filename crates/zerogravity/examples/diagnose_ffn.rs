//! Developer-only numerical probe for an independently captured llama.cpp
//! FFN stage at one exact trained-model token. Never called by the runtime.
//! Usage:
//! cargo run --release --example diagnose_ffn -- <model.gguf> <external-reference.jsonl> <rust-activation.jsonl>
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use zerogravity::math::silu;
use zerogravity::weights::CpuWeights;

fn read_reference(path: &str, rust: bool, layer: usize) -> HashMap<String, Vec<f32>> {
    fs::read_to_string(path)
        .expect("independent trace file missing")
        .lines()
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("bad external JSONL");
            let key = if rust {
                if v["position"].as_u64() != Some(0) || v["layer"].as_u64() != Some(layer as u64) {
                    return None;
                }
                v["stage"].as_str().unwrap_or("").to_owned()
            } else {
                v["name"].as_str().unwrap_or("").to_owned()
            };
            if key.is_empty() {
                return None;
            }
            let values = v["values"]
                .as_array()
                .expect("stage missing float array")
                .iter()
                .map(|x| x.as_f64().expect("nonfinite stage") as f32)
                .collect();
            Some((key, values))
        })
        .collect()
}
fn report(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "stage {label} size");
    let mut diffs = Vec::new();
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        diffs.push(((a - b).abs(), i, a, b));
    }
    diffs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let rmse =
        (diffs.iter().map(|x| f64::from(x.0).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();
    println!(
        "{label}: max={:.10} rms={rmse:.10} worst={} candidate={} reference={}",
        diffs[0].0, diffs[0].1, diffs[0].2, diffs[0].3
    );
    for d in diffs.iter().take(8) {
        println!("  error={:.10} i={} cand={} ref={}", d.0, d.1, d.2, d.3);
    }
}
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: diagnose_ffn model.gguf reference.jsonl rust.jsonl layer-index"
    );
    let layer: usize = args[4].parse().expect("layer index");
    let weights = CpuWeights::open(Path::new(&args[1])).unwrap();
    let reference = read_reference(&args[2], false, layer);
    let rust = read_reference(&args[3], true, layer);
    let names = [
        (format!("attn_post_norm-{layer}"), "ffn_input_norm"),
        (format!("ffn_out-{layer}"), "ffn_output"),
    ];
    for (refname, rname) in names {
        if let (Some(ref_vec), Some(rust_vec)) = (reference.get(&refname), rust.get(rname)) {
            report(rname, rust_vec, ref_vec);
        }
    }
    let reference_input = reference.get(&format!("attn_post_norm-{layer}")).unwrap();
    let rust_input = rust.get("ffn_input_norm").unwrap();
    for projection in ["ffn_gate.weight", "ffn_up.weight"] {
        let name = format!("blk.{layer}.{projection}");
        let ref_result = reference.get(&format!("matmul_{name}")).unwrap();
        report(
            &format!("ref-input->{name}"),
            &weights.matvec(&name, reference_input).unwrap(),
            ref_result,
        );
        report(
            &format!("rust-input->{name}"),
            &weights.matvec(&name, rust_input).unwrap(),
            ref_result,
        );
    }
    if let Some(fused_reference) = reference.get(&format!("source_blk.{layer}.ffn_down_input")) {
        let gate = weights
            .matvec(&format!("blk.{layer}.ffn_gate.weight"), reference_input)
            .unwrap();
        let up = weights
            .matvec(&format!("blk.{layer}.ffn_up.weight"), reference_input)
            .unwrap();
        let fused = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect::<Vec<_>>();
        report("silu(gate)*up (reference input)", &fused, fused_reference);
        let gate_rust = weights
            .matvec(&format!("blk.{layer}.ffn_gate.weight"), rust_input)
            .unwrap();
        let up_rust = weights
            .matvec(&format!("blk.{layer}.ffn_up.weight"), rust_input)
            .unwrap();
        let fused_rust = gate_rust
            .iter()
            .zip(up_rust.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect::<Vec<_>>();
        report(
            "silu(gate)*up (Rust actual input)",
            &fused_rust,
            fused_reference,
        );
        let oracle_result = reference
            .get(&format!("matmul_blk.{layer}.ffn_down.weight"))
            .unwrap();
        report(
            "ffn_down(reference fused input)",
            &weights
                .matvec(&format!("blk.{layer}.ffn_down.weight"), fused_reference)
                .unwrap(),
            oracle_result,
        );
        report(
            "ffn_down(actual Rust fused input)",
            &weights
                .matvec(&format!("blk.{layer}.ffn_down.weight"), &fused_rust)
                .unwrap(),
            oracle_result,
        );
        report(
            "ffn_down(rust fused input)",
            &weights
                .matvec(&format!("blk.{layer}.ffn_down.weight"), &fused)
                .unwrap(),
            oracle_result,
        );
        let mut disagreements = Vec::new();
        for (b, (ours, reference)) in fused
            .chunks_exact(32)
            .zip(fused_reference.chunks_exact(32))
            .enumerate()
        {
            let choose = |chunk: &[f32]| {
                let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                chunk
                    .iter()
                    .map(|x| (x * inv).round_ties_even() as i32)
                    .collect::<Vec<_>>()
            };
            let qa = choose(ours);
            let qb = choose(reference);
            for (i, (&a, &r)) in qa.iter().zip(qb.iter()).enumerate() {
                if a != r {
                    disagreements.push((b * 32 + i, a, r, ours[i], reference[i]));
                }
            }
        }
        let mut input_mismatches = Vec::new();
        for (b, (ours, reference)) in rust_input
            .chunks_exact(32)
            .zip(reference_input.chunks_exact(32))
            .enumerate()
        {
            let choose = |chunk: &[f32]| {
                let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                chunk
                    .iter()
                    .map(|x| (x * inv).round_ties_even() as i32)
                    .collect::<Vec<_>>()
            };
            let qa = choose(ours);
            let qb = choose(reference);
            for (i, (&a, &r)) in qa.iter().zip(qb.iter()).enumerate() {
                if a != r {
                    input_mismatches.push((b * 32 + i, a, r, ours[i], reference[i]));
                }
            }
        }
        println!(
            "ffn input quant mismatches {}/{} {:?}",
            input_mismatches.len(),
            rust_input.len(),
            &input_mismatches[..input_mismatches.len().min(12)]
        );
        let mut fused_mismatches = Vec::new();
        for (b, (ours, reference)) in fused_rust
            .chunks_exact(32)
            .zip(fused_reference.chunks_exact(32))
            .enumerate()
        {
            let choose = |chunk: &[f32]| {
                let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                chunk
                    .iter()
                    .map(|x| (x * inv).round_ties_even() as i32)
                    .collect::<Vec<_>>()
            };
            let qa = choose(ours);
            let qb = choose(reference);
            for (i, (&a, &r)) in qa.iter().zip(qb.iter()).enumerate() {
                if a != r {
                    fused_mismatches.push((b * 32 + i, a, r, ours[i], reference[i]));
                }
            }
        }
        println!(
            "ffn down input quant mismatches {}/{} {:?}",
            fused_mismatches.len(),
            fused_rust.len(),
            &fused_mismatches[..fused_mismatches.len().min(12)]
        );
        println!(
            "q8 quantization mismatches {}/{}",
            disagreements.len(),
            fused.len()
        );
        for mismatch in disagreements.iter().take(18) {
            println!("  {mismatch:?}");
        }
    }
}
