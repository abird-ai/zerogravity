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
