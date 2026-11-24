//! Physical matrix geometry and native numerical/throughput qualification.
use super::*;
use cubecl::prelude::*;
use cubecl_common::profile::TimingMethod;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use futures_lite::future::block_on;

#[test]
#[ignore = "qualified physical gfx1151, isolated release process"]
fn native_bf16_wmma_projection() {
    let device = cubecl::hip::AmdDevice::default();
    let observed = cubecl::hip::observe_device(device.clone()).unwrap();
    assert_eq!(observed.arch_name, "gfx1151");
    let client = HipRuntime::client(&device);
    assert_eq!(client.properties().timing_method, TimingMethod::Device);
    // Exact integer-valued partial tiles and strided output.
    let g = Geometry {
        requests: 19,
        rows: 21,
        columns: 64,
        output_stride: 24,
    };
    let w = (0..g.rows * g.columns)
        .map(|i| ((i * 7 + i / 64 * 3) % 7) as f32 - 3.)
        .collect::<Vec<_>>();
    let x = (0..g.requests * g.columns)
        .map(|i| ((i * 11 + i / 64 * 5) % 11) as f32 - 5.)
        .collect::<Vec<_>>();
    let packed = |xs: &[f32]| {
        xs.chunks_exact(2)
            .map(|v| (v[0].to_bits() >> 16) | (v[1].to_bits() & 0xffff0000))
            .collect::<Vec<u32>>()
    };
    let (wh, xh) = (
        client.create_from_slice(u32::as_bytes(&packed(&w))),
        client.create_from_slice(u32::as_bytes(&packed(&x))),
    );
    let out_words = (g.requests - 1) * g.output_stride + g.rows;
    let output = client.create_from_slice(u32::as_bytes(&vec![0x7fc0beefu32; out_words]));
    let status = client.create_from_slice(&[0; 4]);
    unsafe {
        g.enqueue(&client, &wh, &xh, &output, &status).unwrap();
    }
    let actual = client.read_one(output).unwrap();
    let actual = f32::from_bytes(&actual);
    assert_eq!(u32::from_bytes(&client.read_one(status).unwrap()), &[0]);
    for i in 0..g.requests {
        for j in 0..g.rows {
            let sum = x[i * g.columns..(i + 1) * g.columns]
                .iter()
                .zip(&w[j * g.columns..(j + 1) * g.columns])
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let bits = sum.to_bits();
            let rounded = f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
            assert_eq!(
                actual[i * g.output_stride + j].to_bits(),
                rounded.to_bits(),
                "row={i} column={j}"
            );
        }
        if i + 1 < g.requests {
            assert!(
                actual[i * g.output_stride + g.rows..(i + 1) * g.output_stride]
                    .iter()
                    .all(|v| v.to_bits() == 0x7fc0beef)
            );
        }
    }
    println!("WMMA exact integer and strided tails pass");
    for (rows, columns) in [(3584, 1024), (6144, 1024), (1024, 3584), (5120, 2048)] {
        for requests in [1, 47, 128] {
            let mut seed = 79u32;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed & 0x807f) | 0x3d80
            };
            let weights = (0..rows * columns / 2)
                .map(|_| random() | (random() << 16))
                .collect::<Vec<_>>();
            let inputs = (0..requests * columns / 2)
                .map(|_| random() | (random() << 16))
                .collect::<Vec<_>>();
            let wh = client.create_from_slice(u32::as_bytes(&weights));
            let xh = client.create_from_slice(u32::as_bytes(&inputs));
            let z = client.create_from_slice(&[0; 4]);
            let count = rows * requests;
            let accum = client.empty(count * 4);
            let flags = client.empty(count * 4);
            let out = client.empty(count * 4);
            let status = client.create_from_slice(&[0; 4]);
            let g = Geometry {
                requests,
                rows,
                columns,
                output_stride: rows,
            };
            let launch = |native: bool| unsafe {
                if native {
                    g.enqueue(&client, &wh, &xh, &out, &status).unwrap();
                } else {
                    crate::bf16_projection_kernel::project_cpu_bf16_hoisted::launch_unchecked(
                        &client,
                        CubeCount::Static(count as u32, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(wh.clone(), weights.len()),
                        BufferArg::from_raw_parts(xh.clone(), inputs.len()),
                        BufferArg::from_raw_parts(z.clone(), 1),
                        BufferArg::from_raw_parts(accum.clone(), count),
                        BufferArg::from_raw_parts(out.clone(), count),
                        BufferArg::from_raw_parts(flags.clone(), count),
                        rows,
                        columns,
                        rows,
                    );
                    crate::resident_program::bf16_projection::projection_status::launch_unchecked(
                        &client,
                        CubeCount::Static((count as u32).div_ceil(32), 1, 1),
                        CubeDim::new_1d(32),
                        BufferArg::from_raw_parts(flags.clone(), count),
                        BufferArg::from_raw_parts(status.clone(), 1),
                    );
                }
            };
            launch(false);
            let reference = client.read_one(out.clone()).unwrap();
            launch(true);
            let native = client.read_one(out.clone()).unwrap();
            assert_eq!(
                u32::from_bytes(&client.read_one(status.clone()).unwrap()),
                &[0]
            );
            let reference = f32::from_bytes(&reference);
            let native = f32::from_bytes(&native);
            let (mut error, mut norm, mut different) = (0.0f64, 0.0f64, 0usize);
            for (&a, &b) in reference.iter().zip(native) {
                assert!(a.is_finite() && b.is_finite());
                assert_eq!(b.to_bits() & 0xffff, 0);
                error += (f64::from(a) - f64::from(b)).powi(2);
                norm += f64::from(a).powi(2);
                different += usize::from(a.to_bits() != b.to_bits());
            }
            assert!(norm > 0.0);
            let nrmse = (error / norm).sqrt();
            assert!(nrmse < 0.01, "WMMA error {nrmse}");
            let measure = |native| {
                let window = client.profile_start().unwrap();
                for _ in 0..8 {
                    launch(native);
                }
                let time = client.profile_end(window).unwrap();
                assert_eq!(time.timing_method(), TimingMethod::Device);
                block_on(time.resolve()).unwrap().duration().as_secs_f64() * 1000.0 / 8.0
            };
            let (a, b) = (measure(false), measure(true));
            let (d, c) = (measure(true), measure(false));
            println!(
                "WMMA rows={rows} columns={columns} requests={requests} reference_ms={:.6} native_ms={:.6} speedup={:.3} nrmse={nrmse:.8} different={different}/{count}",
                (a + c) / 2.,
                (b + d) / 2.,
                (a + c) / (b + d)
            );
        }
    }
    block_on(client.sync()).unwrap();
}
