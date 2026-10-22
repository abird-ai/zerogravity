//! Source-trained local check of the first Gated DeltaNet QKV projection and
//! convolution against a separately generated llama.cpp tensor callback.
//! Not linked or called by the Zerogravity runtime.
use std::{collections::HashMap, env, fs, path::Path};
use zerogravity::{math::silu, weights::CpuWeights};
type Values = HashMap<(usize, String), Vec<f32>>;
fn read_ref(path: &str, rust: bool) -> Values {
    fs::read_to_string(path)
        .expect("ref")
        .lines()
        .filter_map(|line| {
            let x: serde_json::Value = serde_json::from_str(line).unwrap();
            let pos = x["position"].as_u64()? as usize;
            let key = if rust {
                if x["layer"].as_u64()? != 0 || x["stage"] != "attention_input_norm" {
                    return None;
                }
                "attn_norm-0".to_owned()
            } else {
                x["name"].as_str()?.to_string()
            };
            let v = x["values"]
                .as_array()?
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            Some(((pos, key), v))
        })
        .collect()
}
fn quantize(x: &[f32]) -> Vec<i32> {
    x.chunks_exact(32)
        .flat_map(|b| {
            let amax = b.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
            b.iter().map(move |x| (x * inv).round_ties_even() as i32)
        })
        .collect()
}
fn max_error(a: &[f32], b: &[f32]) -> (f32, f32, usize, usize) {
    let mut worst = (0f32, 0usize);
    let mut squared = 0f64;
    let mut large = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let e = (x - y).abs();
        if e > worst.0 {
            worst = (e, i);
        }
        squared += f64::from(e).powi(2);
        if e > 1e-5 {
            large += 1;
        }
    }
    (
        worst.0,
        (squared / a.len() as f64).sqrt() as f32,
        worst.1,
        large,
    )
}
fn main() {
    let args = env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 4, "model.gguf external.jsonl rust.jsonl");
    let weights = CpuWeights::open(Path::new(&args[1])).unwrap();
    let refdata = read_ref(&args[2], false);
    let rustdata = read_ref(&args[3], true);
    let conv = weights.matrix_f32("blk.0.ssm_conv1d.weight").unwrap();
    let mut prev_ref = vec![[0f32; 3]; conv.len() / 4];
    let mut prev_rust = prev_ref.clone();
    for pos in 0..6 {
        let key = (pos, "attn_norm-0".to_owned());
        let a = refdata.get(&key).expect("source norm");
        let r = rustdata.get(&key).expect("candidate norm");
        let expected = refdata
            .get(&(pos, "conv_output_silu-0".to_owned()))
            .expect("source conv");
        let qa = quantize(a);
        let qr = quantize(r);
        let mismatches = qa.iter().zip(&qr).filter(|(x, y)| x != y).count();
        let mut candidates = Vec::new();
        for (input, prior) in [(a, &mut prev_ref), (r, &mut prev_rust)] {
            let qkv = weights.matvec("blk.0.attn_qkv.weight", input).unwrap();
            let mut out = vec![0f32; qkv.len()];
            for i in 0..qkv.len() {
                let k = &conv[4 * i..4 * i + 4];
                let v = &mut prior[i];
                out[i] = silu(v[0] * k[0] + v[1] * k[1] + v[2] * k[2] + qkv[i] * k[3]);
                v[0] = v[1];
                v[1] = v[2];
                v[2] = qkv[i];
            }
            candidates.push(out);
        }
        let (ma, ra, ia, la) = max_error(&candidates[0], expected);
        let (mr, rr, ir, lr) = max_error(&candidates[1], expected);
        let (mn, rn, _, _) = max_error(a, r);
        println!(
            "POS {pos} norm_diff_max={mn:.8} rms={rn:.9} q8_mismatch={mismatches}; conv_from_ref_norm_max={ma:.8} rms={ra:.9} large={la} worst_idx={ia}; conv_from_rust_norm_max={mr:.8} rms={rr:.9} large={lr} worst_idx={ir}"
        );
    }
}
