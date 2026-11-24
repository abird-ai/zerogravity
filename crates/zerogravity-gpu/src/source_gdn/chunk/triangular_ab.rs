// Temporary device A/B against the canonical triangular solve.
use super::*;
use cubecl_common::profile::TimingMethod;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use futures_lite::future::block_on;

#[test]
#[ignore = "temporary physical HIP triangular A/B"]
fn triangular_ab() {
    let client = HipRuntime::client(&Default::default());
    assert_eq!(client.properties().timing_method, TimingMethod::Device);
    let heads = 16;
    let n = heads * 64 * 128;
    let mut seed = 79u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as i32 as f32) / 2147483648.0
    };
    for rows in [47, 64] {
        let mut system = vec![0.0f32; heads * 64 * 64];
        let mut values = vec![0.0f32; n];
        let mut beta = vec![0.0f32; n];
        for h in 0..heads {
            for row in 0..rows {
                for j in 0..row {
                    system[(h * 64 + row) * 64 + j] = next() * 0.125;
                }
                for j in 0..128 {
                    values[(h * 64 + row) * 128 + j] = next();
                    beta[(h * 64 + row) * 128 + j] = next() * 0.03125;
                }
            }
        }
        let v = client.empty(n * 4);
        let k = client.empty(n * 4);
        let status = client.create_from_slice(&[0; 4]);
        for fault in [false, true] {
            if fault {
                // Exercise tiny, signed-zero, overflow and invalid branches.
                values[0] = -0.0;
                values[1] = f32::from_bits(1);
                values[2] = f32::MIN_POSITIVE;
                beta[3] = f32::from_bits(0x807fffff);
                values[4] = f32::INFINITY;
                values[5] = f32::from_bits(0x7fc01234);
                values[6] = f32::MAX;
                system[64] = -0.0;
                system[2 * 64] = f32::from_bits(1);
                system[3 * 64] = f32::MAX;
            }
            let a = client.create_from_slice(f32::as_bytes(&system));
            let x = client.create_from_slice(f32::as_bytes(&values));
            let y = client.create_from_slice(f32::as_bytes(&beta));
            let launch = |guarded| unsafe {
                let grid = CubeCount::Static((heads * 128 / 32) as u32, 1, 1);
                let dim = CubeDim::new_1d(32);
                let args = || {
                    (
                        BufferArg::from_raw_parts(a.clone(), system.len()),
                        BufferArg::from_raw_parts(x.clone(), n),
                        BufferArg::from_raw_parts(y.clone(), n),
                        BufferArg::from_raw_parts(v.clone(), n),
                        BufferArg::from_raw_parts(k.clone(), n),
                        BufferArg::from_raw_parts(status.clone(), 1),
                    )
                };
                let (a, x, y, v, k, s) = args();
                if guarded {
                    triangular_guarded::launch_unchecked(
                        &client, grid, dim, a, x, y, v, k, s, heads,
                    );
                } else {
                    triangular::launch_unchecked(&client, grid, dim, a, x, y, v, k, s, heads);
                }
            };
            let read =
                || [v.clone(), k.clone(), status.clone()].map(|h| client.read_one(h).unwrap());
            launch(false);
            let reference = read();
            client.write(&status, cubecl::bytes::Bytes::from_bytes_vec(vec![0; 4]));
            launch(true);
            let candidate = read();
            assert_eq!(&reference[2][..], &candidate[2][..]);
            assert_eq!(
                u32::from_bytes(&candidate[2])[0],
                if fault { NONFINITE_VALUE } else { 0 }
            );
            for (ref_buf, actual_buf) in reference[..2].iter().zip(&candidate[..2]) {
                for (i, (&a, &b)) in u32::from_bytes(ref_buf)
                    .iter()
                    .zip(u32::from_bytes(actual_buf))
                    .enumerate()
                {
                    assert!(
                        a == b || fault && f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan(),
                        "rows={rows} fault={fault} index={i}: {a:08x} != {b:08x}"
                    );
                }
            }
            println!("EXACT rows={rows} heads={heads} fault={fault}");
            if fault {
                continue;
            }
            let measure = |guarded| {
                let window = client.profile_start().unwrap();
                for _ in 0..16 {
                    launch(guarded);
                }
                let duration = client.profile_end(window).unwrap();
                assert_eq!(duration.timing_method(), TimingMethod::Device);
                block_on(duration.resolve())
                    .unwrap()
                    .duration()
                    .as_secs_f64()
                    * 1000.0
            };
            for pair in 0..6 {
                let (a, b) = if pair % 2 == 0 {
                    (measure(false), measure(true))
                } else {
                    let b = measure(true);
                    (measure(false), b)
                };
                println!(
                    "DEVICE_MS rows={rows} pair={pair} launches=16 canonical={a:.6} guarded={b:.6}"
                );
            }
        }
    }
    block_on(client.sync()).unwrap();
}
