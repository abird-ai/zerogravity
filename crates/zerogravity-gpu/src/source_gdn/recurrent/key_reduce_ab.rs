// Temporary direct device A/B. Real kernel bodies, no model or state authority.
use super::*;
use cubecl_common::profile::TimingMethod;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use futures_lite::future::block_on;

#[test]
#[ignore = "temporary physical HIP A/B"]
fn key_reduce_ab() {
    let client = HipRuntime::client(&Default::default());
    assert_eq!(client.properties().timing_method, TimingMethod::Device);
    let values = 16 * 128;
    let mut seed = 79u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        match seed % 127 {
            0 => -0.0,
            1 => f32::from_bits(1),
            2 => -f32::from_bits(0x00800001),
            _ => (seed as i32 as f32) / 2147483648.0,
        }
    };
    let mut state: Vec<f32> = (0..values * 128).map(|_| next()).collect();
    let vector: Vec<f32> = (0..values).map(|_| next()).collect();
    let vx = client.create_from_slice(f32::as_bytes(&vector));
    let out = client.empty(values * 4);
    for fault in [false, true] {
        if fault {
            state[17 * 128 + 7] = f32::INFINITY;
            state[128 * 128 + 71 * 128 + 2] = f32::from_bits(0x7fc00000);
        }
        let sx = client.create_from_slice(f32::as_bytes(&state));
        for round in [false, true] {
            let status = client.create_from_slice(&[0; 4]);
            let launch = |parallel| unsafe {
                let s = BufferArg::from_raw_parts(sx.clone(), state.len());
                let v = BufferArg::from_raw_parts(vx.clone(), vector.len());
                let o = BufferArg::from_raw_parts(out.clone(), values);
                let f = BufferArg::from_raw_parts(status.clone(), 1);
                let grid = CubeCount::Static(values.div_ceil(32) as u32, 1, 1);
                if parallel {
                    key_reduce_subtotals::launch_unchecked(
                        &client,
                        grid,
                        CubeDim::new_1d(256),
                        s,
                        v,
                        o,
                        f,
                        values,
                        round,
                    );
                } else {
                    key_reduce::launch_unchecked(
                        &client,
                        grid,
                        CubeDim::new_1d(32),
                        s,
                        v,
                        o,
                        f,
                        values,
                        round,
                    );
                }
            };
            launch(false);
            let reference = client.read_one(out.clone()).unwrap();
            let reference_status = client.read_one(status.clone()).unwrap();
            client.write(&status, cubecl::bytes::Bytes::from_bytes_vec(vec![0; 4]));
            launch(true);
            let actual = client.read_one(out.clone()).unwrap();
            let actual_status = client.read_one(status.clone()).unwrap();
            assert_eq!(&reference_status[..], &actual_status[..]);
            assert_eq!(
                u32::from_bytes(&actual_status)[0],
                if fault { NONFINITE_VALUE } else { 0 }
            );
            for (i, (&a, &b)) in u32::from_bytes(&reference)
                .iter()
                .zip(u32::from_bytes(&actual))
                .enumerate()
            {
                assert!(
                    a == b || fault && f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan(),
                    "value {i}: {a:08x} != {b:08x}"
                );
            }
            if !fault {
                for (at, &bits) in u32::from_bytes(&actual).iter().enumerate() {
                    let h = at / 128;
                    let j = at % 128;
                    let mut carry = 0.0f32;
                    for block in 0..8 {
                        let mut subtotal = 0.0f32;
                        for key in block * 16..(block + 1) * 16 {
                            subtotal += state[(h * 128 + key) * 128 + j] * vector[h * 128 + key];
                        }
                        carry += subtotal;
                    }
                    let mut expected = carry.to_bits();
                    if round {
                        expected = (expected + 0x7fff + ((expected >> 16) & 1)) & 0xffff0000;
                    }
                    assert_eq!(bits, expected, "CPU value {at}");
                }
            }
            println!("EXACT values={values} round={round} fault={fault}");
            if fault {
                continue;
            }
            let measure = |parallel| {
                let window = client.profile_start().unwrap();
                for _ in 0..96 {
                    launch(parallel);
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
                    "DEVICE_MS round={round} pair={pair} launches=96 scalar={a:.6} subtotals={b:.6}"
                );
            }
        }
    }
    block_on(client.sync()).unwrap();
}
