//! Developer-only arithmetic comparison of GGML IMROPE with exactly captured
//! independently computed CPU tensors, never used by production runtime.
use serde_json::Value;
use std::{env, fs};
fn read(path: &str, ref_stage: &str, pos: usize) -> Vec<f32> {
    for line in fs::read_to_string(path).unwrap().lines() {
        let x: Value = serde_json::from_str(line).unwrap();
        if x["position"].as_u64() == Some(pos as u64)
            && (x["name"] == ref_stage || x["stage"] == ref_stage)
        {
            return x["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|y| y.as_f64().unwrap() as f32)
                .collect();
        }
    }
    panic!("not found {} {}", ref_stage, pos);
}
fn main() {
    let a = env::args().collect::<Vec<_>>();
    assert_eq!(a.len(), 3);
    let theta_scale = 10_000_000f32.powf(-2.0 / 64.0);
    println!(
        "theta_scale = {theta_scale:.10} bits={:x}",
        theta_scale.to_bits()
    );
    for pos in 1..=5usize {
        let q = read(&a[1], "Qcur_normed-7", pos);
        let oracle = read(&a[1], "Qcur-7", pos);
        let mut predicted = [
            vec![0f32; q.len()],
            vec![0f32; q.len()],
            vec![0f32; q.len()],
            vec![0f32; q.len()],
        ];
        let mut phase = pos as f32;
        for i in 0..32 {
            let (s, c) = phase.sin_cos();
            for h in 0..q.len() / 256 {
                let index = h * 256 + i;
                let j = index + 32;
                let x = q[index];
                let y = q[j];
                let outcomes = [
                    (x * c - y * s, x * s + y * c),
                    (x.mul_add(c, -(y * s)), x.mul_add(s, y * c)),
                    ((-y).mul_add(s, x * c), y.mul_add(c, x * s)),
                    (x.mul_add(c, -(y * s)), y.mul_add(c, x * s)),
                ];
                for (v, pair) in predicted.iter_mut().zip(outcomes) {
                    v[index] = pair.0;
                    v[j] = pair.1;
                }
            }
            phase *= theta_scale;
        }
        for p in &mut predicted {
            for h in 0..q.len() / 256 {
                p[h * 256 + 64..h * 256 + 256].copy_from_slice(&q[h * 256 + 64..h * 256 + 256]);
            }
        }
        for (id, candidate) in predicted.iter().enumerate() {
            let matches = candidate
                .iter()
                .zip(&oracle)
                .filter(|(x, y)| x.to_bits() == y.to_bits())
                .count();
            let max = candidate
                .iter()
                .zip(&oracle)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            println!(
                "pos={pos} variant={id} bitmatches={matches}/{} maxdiff={max}",
                q.len()
            );
        }
    }
}
