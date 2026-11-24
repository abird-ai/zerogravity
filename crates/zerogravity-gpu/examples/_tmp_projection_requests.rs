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
    let launch = |tile: usize| {
        let count = CubeCount::Static((rows * requests.div_ceil(tile.max(1))) as u32, 1, 1);
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
            match tile {
                1 => kernel::project_cpu_bf16::launch_unchecked(
                    client, count, dim, w, x, z, a, o, s, rows, columns, stride,
                ),
                0 => kernel::project_cpu_bf16_hoisted::launch_unchecked(
                    client, count, dim, w, x, z, a, o, s, rows, columns, stride,
                ),
                _ => kernel::project_cpu_bf16_requests::launch_unchecked(
                    client, count, dim, w, x, z, a, o, s, rows, columns, stride, tile,
                ),
            }
        }
    };
    let read =
        || [acc.clone(), output.clone(), status.clone()].map(|h| client.read_one(h).unwrap());
    launch(1);
    let baseline = read();
    for tile in [0, 2, 4] {
        launch(tile);
        let candidate = read();
        for (a, b) in baseline.iter().zip(&candidate) {
            assert_eq!(
                &a[..],
                &b[..],
                "bits rows={rows} columns={columns} requests={requests} tile={tile} fault={fault}"
            );
        }
    }
    let flags = u32::from_bytes(&baseline[2]);
    if fault {
        assert!(flags.iter().any(|f| f & 1 != 0));
        assert!(flags.iter().any(|f| f & 2 != 0));
    } else {
        assert!(flags.iter().all(|&f| f == 0));
    }
    if stride > rows {
        let rounded = u32::from_bytes(&baseline[1]);
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
    if fault || rows < 128 {
        return;
    }
    let measure = |tile| {
        let window = client.profile_start().unwrap();
        for _ in 0..24 {
            launch(tile);
        }
        let duration = client.profile_end(window).unwrap();
        assert_eq!(duration.timing_method(), TimingMethod::Device);
        block_on(duration.resolve())
            .unwrap()
            .duration()
            .as_secs_f64()
            * 1000.0
    };
    for tile in [0, 2, 4] {
        if requests == 1 && tile != 0 {
            continue;
        }
        for pair in 0..4 {
            let (a, b) = if pair % 2 == 0 {
                (measure(1), measure(tile))
            } else {
                let b = measure(tile);
                (measure(1), b)
            };
            println!(
                "DEVICE_MS rows={rows} columns={columns} requests={requests} tile={tile} pair={pair} launches=24 canonical={a:.6} candidate={b:.6}"
            );
        }
    }
    block_on(client.sync()).unwrap();
}

fn main() {
    let client = HipRuntime::client(&Default::default());
    assert_eq!(client.properties().timing_method, TimingMethod::Device);
    for (rows, columns) in [(3584, 1024), (6144, 1024), (1024, 3584), (3584, 2048)] {
        for requests in [1, 8, 47] {
            run(&client, rows, columns, requests, rows, false);
        }
    }
    run(&client, 7, 1024, 3, 11, false);
    run(&client, 7, 1024, 3, 11, true);
    block_on(client.sync()).unwrap();
}
