use super::*;

#[test]
fn geometry_and_causal_extents_are_checked() -> Result<()> {
    for (rows, base, capacity) in [(1, 0, 1), (17, 3, 32), (65, 0, 129), (1, 32_767, 32_768)] {
        let a = Bf16CausalAttention::new(rows, base, capacity, 8, 2, 256, 0.0625)?;
        assert_eq!(a.grid, [rows.div_ceil(16) as u32, 8, 1]);
        assert_eq!(
            a.kernel.binding_sizes(),
            [
                rows * 4096,
                capacity * 1024,
                capacity * 1024,
                rows * 4096,
                4
            ]
        );
    }
    for (rows, base, cap, qh, kh, width, scale) in [
        (0, 0, 1, 8, 2, 256, 0.0625),
        (1, usize::MAX, 1, 8, 2, 256, 0.0625),
        (2, 0, 1, 8, 2, 256, 0.0625),
        (1, 0, 1, 3, 2, 256, 0.0625),
        (1, 0, 1, 8, 0, 256, 0.0625),
        (1, 0, 1, 8, 2, 128, 0.0625),
        (1, 0, 1, 8, 2, 256, f32::NAN),
        (1, 0, 1, 8, 2, 256, f32::MAX),
        (1, 0, 1, 8, 2, 256, 0.0),
        (1, 0, usize::MAX, 8, 2, 256, 0.0625),
    ] {
        assert!(Bf16CausalAttention::new(rows, base, cap, qh, kh, width, scale).is_err());
    }
    Ok(())
}

#[test]
fn batch4_ragged_geometry_keeps_private_frontiers_and_aligned_convenience() -> Result<()> {
    let aligned = Bf16CausalAttentionBatch4::new(17, 128, 8, 2, 256, 0.0625)?;
    let ragged = Bf16CausalAttentionBatch4::new_ragged([0, 17, 63, 127], 128, 8, 2, 256, 0.0625)?;
    let expected = [
        4 * 8 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        128 * 2 * 256 * 2,
        4 * 8 * 256 * 2,
        4,
    ];
    assert_eq!(aligned.grid, [1, 8, 4]);
    assert_eq!(ragged.grid, [1, 8, 4]);
    assert_eq!(aligned.kernel.binding_sizes(), expected);
    assert_eq!(ragged.kernel.binding_sizes(), expected);
    let prefill =
        Bf16CausalAttentionBatch4::new_ragged_rows(17, [0, 17, 63, 100], 128, 8, 2, 256, 0.0625)?;
    let mut prefill_expected = expected;
    prefill_expected[0] *= 17;
    prefill_expected[9] *= 17;
    assert_eq!(prefill.grid, [2, 8, 4]);
    assert_eq!(prefill.kernel.binding_sizes(), prefill_expected);
    assert!(Bf16CausalAttentionBatch4::new_ragged([0, 1, 2, 128], 128, 8, 2, 256, 0.0625).is_err());
    assert!(Bf16CausalAttentionBatch4::new_ragged([0; 4], 0, 8, 2, 256, 0.0625).is_err());
    assert!(Bf16CausalAttentionBatch4::new_ragged([0; 4], 128, 3, 2, 256, 0.0625).is_err());
    Ok(())
}

#[test]
fn batch8_ragged_geometry_keeps_private_frontiers_and_aligned_convenience() -> Result<()> {
    let aligned = Bf16CausalAttentionBatch8::new(17, 128, 8, 2, 256, 0.0625)?;
    let ragged = Bf16CausalAttentionBatch8::new_ragged(
        [0, 1, 7, 17, 31, 63, 95, 127],
        128,
        8,
        2,
        256,
        0.0625,
    )?;
    let row_bytes = 8 * 256 * 2;
    let kv_bytes = 128 * 2 * 256 * 2;
    let mut expected = Vec::with_capacity(19);
    expected.push(8 * row_bytes);
    expected.extend(std::iter::repeat_n(kv_bytes, 16));
    expected.push(8 * row_bytes);
    expected.push(4);
    assert_eq!(aligned.grid, [1, 8, 8]);
    assert_eq!(ragged.grid, [1, 8, 8]);
    assert_eq!(aligned.kernel.binding_sizes(), expected);
    assert_eq!(ragged.kernel.binding_sizes(), expected);
    assert!(
        Bf16CausalAttentionBatch8::new_ragged([0, 1, 2, 3, 4, 5, 6, 128], 128, 8, 2, 256, 0.0625,)
            .is_err()
    );
    assert!(Bf16CausalAttentionBatch8::new_ragged([0; 8], 0, 8, 2, 256, 0.0625).is_err());
    assert!(Bf16CausalAttentionBatch8::new_ragged([0; 8], 128, 3, 2, 256, 0.0625).is_err());
    Ok(())
}

#[test]
fn output_and_status_aliases_refuse() -> Result<()> {
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_core::environment::stream::StreamId;
    let h = || {
        Handle::new(
            ServiceId::of::<()>(DeviceId::new(0, 0)),
            StreamId::current(),
            16,
        )
    };
    let mut handles = [h(), h(), h(), h(), h()];
    require_disjoint_writes(&handles, &[3, 4])?;
    handles[1] = handles[0].clone();
    require_disjoint_writes(&handles, &[3, 4])?;
    handles[3] = handles[0].clone();
    assert!(require_disjoint_writes(&handles, &[3, 4]).is_err());
    handles[3] = h();
    handles[4] = handles[3].clone();
    assert!(require_disjoint_writes(&handles, &[3, 4]).is_err());
    Ok(())
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
mod physical {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Read;
    use zerogravity_core::{
        device_events::{BufferRole, DeviceAccess},
        resource_arbiter::{AllocationClass, AllocationRequest},
    };

    #[test]
    #[ignore = "isolated owned attention versus independently authenticated native reference"]
    fn captured_attention_on_nvidia() -> Result<()> {
        futures_lite::future::block_on(run(true))
    }

    #[test]
    #[ignore = "isolated causal/multitile/masked-tail/nonfinite/retirement attention coverage"]
    fn causal_boundaries_on_nvidia() -> Result<()> {
        futures_lite::future::block_on(run(false))
    }

    #[test]
    #[ignore = "physical ragged B4 attention versus four accepted private B1 launches"]
    fn ragged_batch4_matches_private_b1_on_nvidia() -> Result<()> {
        futures_lite::future::block_on(ragged_batch4_matches_private_b1())
    }

    #[test]
    #[ignore = "physical ragged B8 attention versus eight accepted private B1 launches"]
    fn ragged_batch8_matches_private_b1_on_nvidia() -> Result<()> {
        futures_lite::future::block_on(ragged_batch8_matches_private_b1())
    }

    async fn ragged_batch4_matches_private_b1() -> Result<()> {
        const BASES: [usize; 4] = [0, 17, 63, 127];
        ragged_batch_matches_private_b1(
            "bf16-attention-ragged-b4",
            BASES,
            |client, query, keys, values, output, status| {
                Bf16CausalAttentionBatch4::new_ragged(BASES, 128, 8, 2, 256, 0.0625)?
                    .prepare(
                        client,
                        [
                            query,
                            keys[0].clone(),
                            keys[1].clone(),
                            keys[2].clone(),
                            keys[3].clone(),
                            values[0].clone(),
                            values[1].clone(),
                            values[2].clone(),
                            values[3].clone(),
                            output,
                            status,
                        ],
                    )?
                    .launch_on(client)?;
                Ok(())
            },
        )
        .await
    }

    async fn ragged_batch8_matches_private_b1() -> Result<()> {
        const BASES: [usize; 8] = [0, 1, 7, 17, 31, 63, 95, 127];
        ragged_batch_matches_private_b1(
            "bf16-attention-ragged-b8",
            BASES,
            |client, query, keys, values, output, status| {
                Bf16CausalAttentionBatch8::new_ragged(BASES, 128, 8, 2, 256, 0.0625)?
                    .prepare(
                        client,
                        [
                            query,
                            keys[0].clone(),
                            keys[1].clone(),
                            keys[2].clone(),
                            keys[3].clone(),
                            keys[4].clone(),
                            keys[5].clone(),
                            keys[6].clone(),
                            keys[7].clone(),
                            values[0].clone(),
                            values[1].clone(),
                            values[2].clone(),
                            values[3].clone(),
                            values[4].clone(),
                            values[5].clone(),
                            values[6].clone(),
                            values[7].clone(),
                            output,
                            status,
                        ],
                    )?
                    .launch_on(client)?;
                Ok(())
            },
        )
        .await
    }

    async fn ragged_batch_matches_private_b1<const N: usize>(
        diagnostic_name: &'static str,
        bases: [usize; N],
        launch_batch: impl FnOnce(&Client, Handle, &[Handle], &[Handle], Handle, Handle) -> Result<()>,
    ) -> Result<()> {
        const CAPACITY: usize = 128;
        const QH: usize = 8;
        const KH: usize = 2;
        const WIDTH: usize = 256;
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new(diagnostic_name)?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let queue = service.pinned_cuda_queue()?.clone();
        let singles = bases
            .into_iter()
            .map(|base| Bf16CausalAttention::new(1, base, CAPACITY, QH, KH, WIDTH, 0.0625))
            .collect::<Result<Vec<_>>>()?;
        for single in &singles {
            single.prepare_code(&queue)?;
        }

        let bf16 = |value: f32| {
            let bits = value.to_bits();
            ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let query_row_bytes = QH * WIDTH * 2;
        let kv_bytes = CAPACITY * KH * WIDTH * 2;
        let mut query_rows = Vec::with_capacity(N);
        let mut keys = Vec::with_capacity(N);
        let mut values = Vec::with_capacity(N);
        for (lane, &base) in bases.iter().enumerate() {
            let query = (0..QH * WIDTH)
                .flat_map(|i| {
                    let v = (((i * 11 + lane * 5) % 13) as f32 - 6.0) / 16.0;
                    bf16(v).to_le_bytes()
                })
                .collect::<Vec<_>>();
            let mut key = Vec::with_capacity(kv_bytes);
            let mut value = Vec::with_capacity(kv_bytes);
            for row in 0..CAPACITY {
                for i in 0..KH * WIDTH {
                    if row > base {
                        key.extend_from_slice(&0x7fc0u16.to_le_bytes());
                        value.extend_from_slice(&0x7fc0u16.to_le_bytes());
                    } else {
                        let k = (((row * 7 + i * 3 + lane) % 17) as f32 - 8.0) / 32.0;
                        let v = (((row * 5 + i * 13 + lane * 7) % 19) as f32 - 9.0) / 16.0;
                        key.extend_from_slice(&bf16(k).to_le_bytes());
                        value.extend_from_slice(&bf16(v).to_le_bytes());
                    }
                }
            }
            query_rows.push(query);
            keys.push(key);
            values.push(value);
        }
        let batch_query_bytes = query_rows.concat();
        let batch_query = client.create_from_slice(&batch_query_bytes);
        let key_handles = keys
            .iter()
            .map(|bytes| client.create_from_slice(bytes))
            .collect::<Vec<_>>();
        let value_handles = values
            .iter()
            .map(|bytes| client.create_from_slice(bytes))
            .collect::<Vec<_>>();
        let batch_output = client.create_from_slice(&vec![0xa5; N * query_row_bytes]);
        let batch_status = client.create_from_slice(&0u32.to_ne_bytes());
        launch_batch(
            client,
            batch_query,
            &key_handles,
            &value_handles,
            batch_output.clone(),
            batch_status.clone(),
        )?;

        let mut single_outputs = Vec::with_capacity(N);
        let mut single_statuses = Vec::with_capacity(N);
        for lane in 0..N {
            let query = client.create_from_slice(&query_rows[lane]);
            let output = client.create_from_slice(&vec![0xa5; query_row_bytes]);
            let status = client.create_from_slice(&0u32.to_ne_bytes());
            singles[lane]
                .prepare(
                    client,
                    [
                        query,
                        key_handles[lane].clone(),
                        value_handles[lane].clone(),
                        output.clone(),
                        status.clone(),
                    ],
                )?
                .launch_on(client)?;
            single_outputs.push(output);
            single_statuses.push(status);
        }
        client
            .sync()
            .await
            .map_err(|e| Error::State(format!("ragged B{N} attention sync: {e}")))?;
        let actual = client
            .read_one(batch_output)
            .map_err(|e| Error::State(format!("ragged B{N} attention output: {e}")))?;
        let status = client
            .read_one(batch_status)
            .map_err(|e| Error::State(format!("ragged B{N} attention status: {e}")))?;
        if &status[..] != 0u32.to_ne_bytes().as_slice() {
            return Err(Error::State(format!(
                "ragged B{N} attention status differs: {status:?}"
            )));
        }
        let mut mismatches = 0usize;
        for lane in 0..N {
            let expected = client
                .read_one(single_outputs[lane].clone())
                .map_err(|e| Error::State(format!("private B1 attention output: {e}")))?;
            let one_status = client
                .read_one(single_statuses[lane].clone())
                .map_err(|e| Error::State(format!("private B1 attention status: {e}")))?;
            if &one_status[..] != 0u32.to_ne_bytes().as_slice() {
                return Err(Error::State(format!(
                    "private B1 attention lane {lane} status differs: {one_status:?}"
                )));
            }
            mismatches += actual[lane * query_row_bytes..(lane + 1) * query_row_bytes]
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
        }
        if mismatches != 0 {
            return Err(Error::State(format!(
                "ragged B{N} attention differs from private B1: {mismatches} BF16 words"
            )));
        }
        println!(
            "RAGGED_B{N}_ATTENTION_PASS bases={bases:?} mismatches=0 status=0 private_b1=true"
        );
        drop((
            singles,
            key_handles,
            value_handles,
            single_outputs,
            single_statuses,
        ));
        client
            .memory_cleanup()
            .map_err(|e| Error::State(format!("ragged B{N} attention cleanup: {e}")))?;
        service.close_model_namespace()?;
        Ok(())
    }

    type Case = (Bf16CausalAttention, [Vec<u8>; 3], Vec<u8>, u32);

    fn synthetic() -> Result<Vec<Case>> {
        let mut cases = Vec::new();
        for (rows, base, capacity, qh, kh) in [
            (1, 0, 1, 1, 1),
            (17, 0, 32, 4, 2),
            (47, 0, 64, 8, 2),
            (65, 0, 96, 8, 2),
            (3, 63, 80, 4, 1),
            (17, 112, 192, 8, 2),
            (1, 511, 520, 8, 2),
            (1, 1023, 1032, 8, 2),
            (1, 32_767, 32_768, 2, 1),
        ] {
            let attention = Bf16CausalAttention::new(rows, base, capacity, qh, kh, 256, 0.0625)?;
            let sizes = attention.kernel.binding_sizes();
            let query = vec![0; sizes[0]];
            let mut keys = vec![0; sizes[1]];
            let mut values = vec![0; sizes[2]];
            let mut expected = Vec::with_capacity(sizes[3]);
            let pattern = |head: usize, lane: usize| [0.25f32, -0.5, 1.0, -2.0][(head + lane) % 4];
            for row in 0..capacity {
                for h in 0..kh {
                    for d in 0..256 {
                        let i = (row * kh + h) * 256 + d;
                        let (k, v) = if row >= base + rows {
                            (0x7fc0u16, 0x7fc0u16)
                        } else {
                            (
                                0,
                                ((if row % 2 == 0 { 0.0 } else { pattern(h, d) }).to_bits() >> 16)
                                    as u16,
                            )
                        };
                        keys[2 * i..2 * i + 2].copy_from_slice(&k.to_le_bytes());
                        values[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            for row in 0..rows {
                for h in 0..qh {
                    for d in 0..256 {
                        let active = base + row + 1;
                        let value = if active / 2 == 0 {
                            0.0
                        } else {
                            (active / 2) as f32 / active as f32 * pattern(h / (qh / kh), d)
                        };
                        let bits = value.to_bits();
                        let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                        expected.extend_from_slice(&rounded.to_le_bytes());
                    }
                }
            }
            cases.push((attention, [query, keys, values], expected, 0));
        }
        for operand in 0..3 {
            let attention = Bf16CausalAttention::new(1, 0, 1, 1, 1, 256, 0.0625)?;
            let mut inputs = [vec![0; 512], vec![0; 512], vec![0; 512]];
            inputs[operand][..2].copy_from_slice(&0x7fc0u16.to_le_bytes());
            cases.push((
                attention,
                inputs,
                vec![0; 512],
                crate::program_kernels::NONFINITE_VALUE,
            ));
        }
        Ok(cases)
    }

    async fn run(captured: bool) -> Result<()> {
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("bf16-attention")?;
        let service = diagnostic.service.clone();
        let compiler = diagnostic.compiler.clone();
        let queue = service.pinned_cuda_queue()?.clone();
        let cases = if captured {
            vec![reference()?]
        } else {
            synthetic()?
        };
        let mut failed = 0;
        for (attention, inputs, expected, expected_status) in cases {
            let guarded = |b: &[u8]| [vec![0xa5; 16], b.to_vec(), vec![0xa5; 16]].concat();
            let data = [
                guarded(&inputs[0]),
                guarded(&inputs[1]),
                guarded(&inputs[2]),
                guarded(&vec![0xa5; expected.len()]),
                guarded(&[0; 4]),
            ];
            let ticket = service.queue().submit(
                service.budget(),
                service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: data.iter().map(|b| b.len() as u64).sum(),
                }],
                None,
            )?;
            let mut slot = ticket.wait().await?;
            slot.begin_submission()?;
            let mut init = service.begin_initialization(0)?;
            init.retain_completion_owner(compiler.clone())?;
            attention.prepare_code(&queue)?;
            let mut leases = Vec::new();
            for b in &data {
                leases.push(init.upload_bytes(b, BufferRole::Scratch)?);
            }
            drop(init.complete_sync().await?);
            slot.finish()?;
            let handles =
                std::array::from_fn(|i| leases[i].handle().offset_start(16).offset_end(16));
            let launch = attention.prepare(service.client(), handles)?;
            let ticket = service.queue().submit(
                service.budget(),
                service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: data.iter().map(|b| b.len() as u64).sum(),
                }],
                None,
            )?;
            let mut slot = ticket.wait().await?;
            slot.begin_submission()?;
            let accesses = leases
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    (
                        l.clone(),
                        if i < 3 {
                            DeviceAccess::Read
                        } else {
                            DeviceAccess::Write
                        },
                    )
                })
                .collect::<Vec<_>>();
            let mut flight = service.submit(0, &accesses)?;
            flight.retain_completion_owner(compiler.clone())?;
            launch.launch_on(service.client())?;
            let reads = leases
                .iter()
                .map(|l| l.readback(0, l.bytes()))
                .collect::<Result<Vec<_>>>()?;
            let (ready, actual) = flight
                .complete_with_readback(&leases[3..], &reads, |a| {
                    if a.len() != 5 {
                        return Err(Error::State("attention readback count".into()));
                    }
                    for i in 0..5 {
                        if a[i].len() != data[i].len()
                            || a[i][..16] != data[i][..16]
                            || a[i][data[i].len() - 16..] != data[i][data[i].len() - 16..]
                            || (i < 3 && a[i].as_ref() != data[i])
                        {
                            return Err(Error::State(
                                "attention guards or immutable inputs changed".into(),
                            ));
                        }
                    }
                    Ok(())
                })
                .await?;
            let status = u32::from_le_bytes(actual[4][16..20].try_into().unwrap());
            let mismatch = actual[3][16..16 + expected.len()]
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| {
                    (
                        i,
                        u16::from_le_bytes(a.try_into().unwrap()),
                        u16::from_le_bytes(b.try_into().unwrap()),
                    )
                })
                .collect::<Vec<_>>();
            drop(ready);
            slot.finish()?;
            drop((launch, accesses, reads, leases));
            if service.resident_bytes()? != 0 || service.model_events()? != 0 {
                return Err(Error::State("attention retirement differs".into()));
            }
            let bad = status != expected_status || (expected_status == 0 && !mismatch.is_empty());
            failed += usize::from(bad);
            println!(
                "owned-attention grid={:?} mismatches={} first={:?} status={status} expected_status={expected_status} guards=true retired=true",
                attention.grid,
                mismatch.len(),
                &mismatch[..mismatch.len().min(12)]
            );
        }
        service
            .client()
            .memory_cleanup()
            .map_err(|e| Error::State(format!("attention cleanup: {e}")))?;
        service.close_model_namespace()?;
        if failed != 0 {
            return Err(Error::State(format!(
                "attention numerical/status mismatch in {failed} cases; all retired"
            )));
        }
        Ok(())
    }

    fn reference() -> Result<Case> {
        let env = |name| std::env::var(name).map_err(|_| Error::State(format!("missing {name}")));
        let path = env("ZG_BF16_ATTENTION_REFERENCE")?;
        let expected_digest = env("ZG_BF16_ATTENTION_REFERENCE_SHA256")?;
        let file = std::fs::File::open(path)
            .map_err(|e| Error::State(format!("attention fixture: {e}")))?;
        let m = file
            .metadata()
            .map_err(|e| Error::State(format!("attention fixture metadata: {e}")))?;
        if !m.is_file() || !(60..=32 * 1024 * 1024).contains(&m.len()) {
            return Err(Error::Limit("attention bounded regular fixture".into()));
        }
        let mut bytes = Vec::new();
        file.take(m.len() + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::State(format!("attention fixture read: {e}")))?;
        if bytes.len() != m.len() as usize
            || bytes[..8] != *b"ZGAT0001"
            || format!("{:x}", Sha256::digest(&bytes)) != expected_digest
        {
            return Err(Error::State(
                "attention independent digest/header/extent mismatch".into(),
            ));
        }
        let mut dims = [0usize; 6];
        for (i, d) in dims.iter_mut().enumerate() {
            *d = usize::try_from(u64::from_le_bytes(
                bytes[8 + i * 8..16 + i * 8].try_into().unwrap(),
            ))
            .map_err(|_| Error::Limit("attention fixture dimension".into()))?;
        }
        let [rows, base, capacity, qh, kh, width] = dims;
        let a = Bf16CausalAttention::new(
            rows,
            base,
            capacity,
            qh,
            kh,
            width,
            f32::from_le_bytes(bytes[56..60].try_into().unwrap()),
        )?;
        let sizes = a.kernel.binding_sizes();
        let mut offset = 60usize;
        let mut parts = Vec::new();
        for size in &sizes[..4] {
            let end = offset
                .checked_add(*size)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| Error::State("attention fixture payload extent".into()))?;
            parts.push(bytes[offset..end].to_vec());
            offset = end;
        }
        if offset != bytes.len() {
            return Err(Error::State("attention fixture trailing bytes".into()));
        }
        let output = parts.pop().unwrap();
        Ok((a, parts.try_into().unwrap(), output, 0))
    }
}
