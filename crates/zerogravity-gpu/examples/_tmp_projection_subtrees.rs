// Temporary same-device A/B; no model or production provider admission.
#[allow(dead_code)]
#[path = "../src/bf16_projection_kernel.rs"]
mod kernel;

use cubecl::{client::Client, prelude::*};
use cubecl_common::profile::TimingMethod;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use futures_lite::future::block_on;

fn packed(count: usize, mut seed: u32) -> Vec<u32> {
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        // Mixed signs/mantissas and finite magnitudes; include signed zero and subnormal.
        match seed % 67 {
            0 => 0x8000,
            1 => 0x0001,
            _ => (seed & 0x807f) | (((seed % 17) + 117) << 7),
        }
    };
    (0..count).map(|_| next() | (next() << 16)).collect()
}

fn run(client: &Client, rows: usize, columns: usize, requests: usize, stride: usize, fault: bool) {
    let mut w = packed(rows * columns / 2, 42);
    let mut x = packed(requests * columns / 2, 73);
    if fault {
        // Distinct odd/even lanes and K64 chunks; both operand flags must survive.
        w[columns / 2 + 33] = 0x7f803f80;
        x[columns / 2 + 63] = 0x3f807f80;
    }
    let weights = client.create_from_slice(u32::as_bytes(&w));
    let input = client.create_from_slice(u32::as_bytes(&x));
    let zero = client.create_from_slice(&[0; 4]);
    let n = rows * requests;
    let span = (requests - 1) * stride + rows;
    let acc = client.empty(n * 4);
    let output = client.create_from_slice(u32::as_bytes(&vec![0x7fc0beef; span]));
    let status = client.empty(n * 4);
    let launch = |hoisted: bool| {
        let count = CubeCount::Static(n as u32, 1, 1);
        let dim = CubeDim::new_1d(64);
        let args = || unsafe {
            (
                BufferArg::from_raw_parts(weights.clone(), rows * columns / 2),
                BufferArg::from_raw_parts(input.clone(), requests * columns / 2),
                BufferArg::from_raw_parts(zero.clone(), 1),
                BufferArg::from_raw_parts(acc.clone(), n),
                BufferArg::from_raw_parts(output.clone(), span),
                BufferArg::from_raw_parts(status.clone(), n),
            )
        };
        let (w, x, z, a, o, s) = args();
        unsafe {
            if hoisted {
                kernel::project_cpu_bf16_subtrees::launch_unchecked(
                    client, count, dim, w, x, z, a, o, s, rows, columns, stride,
                );
            } else {
                kernel::project_cpu_bf16_hoisted::launch_unchecked(
                    client, count, dim, w, x, z, a, o, s, rows, columns, stride,
                );
            }
        }
    };
    let read =
        || [acc.clone(), output.clone(), status.clone()].map(|h| client.read_one(h).unwrap());
    launch(false);
    let baseline = read();
    launch(true);
    let candidate = read();
    for (a, b) in baseline.iter().zip(&candidate) {
        assert_eq!(
            &a[..],
            &b[..],
            "bit mismatch rows={rows} columns={columns} requests={requests} fault={fault}"
        );
    }
    let flags = u32::from_bytes(&candidate[2]);
    if fault {
        assert!(flags.iter().any(|f| f & 1 != 0));
        assert!(flags.iter().any(|f| f & 2 != 0));
    } else {
        assert!(flags.iter().all(|&f| f == 0));
    }
    if stride > rows {
        let rounded = u32::from_bytes(&candidate[1]);
        for req in 0..requests - 1 {
            assert!(
                rounded[req * stride + rows..(req + 1) * stride]
                    .iter()
                    .all(|&v| v == 0x7fc0beef)
            );
        }
    }
    println!(
        "EXACT rows={rows} columns={columns} requests={requests} stride={stride} fault={fault}"
    );
    if requests != 1 || fault {
        return;
    }
    let measure = |hoisted| {
        let window = client.profile_start().unwrap();
        for _ in 0..96 {
            launch(hoisted);
        }
        let duration = client.profile_end(window).unwrap();
        assert_eq!(duration.timing_method(), TimingMethod::Device);
        block_on(duration.resolve())
            .unwrap()
            .duration()
            .as_secs_f64()
            * 1000.0
    };
    for i in 0..6 {
        let (a, b) = if i % 2 == 0 {
            (measure(false), measure(true))
        } else {
            let b = measure(true);
            (measure(false), b)
        };
        println!(
            "DEVICE_MS rows={rows} columns={columns} pair={i} launches=96 hoisted={a:.6} subtrees={b:.6}"
        );
    }
    block_on(client.sync()).unwrap();
}

fn main() {
    let client = HipRuntime::client(&Default::default());
    assert_eq!(client.properties().timing_method, TimingMethod::Device);
    for rows in [3584, 6144] {
        run(&client, rows, 1024, 1, rows, false);
    }
    run(&client, 1024, 3584, 1, 1024, false);
    run(&client, 2048, 1024, 1, 2048, false);
    run(&client, 7, 1024, 3, 11, false);
    run(&client, 7, 1024, 3, 11, true);
    block_on(client.sync()).unwrap();
}
