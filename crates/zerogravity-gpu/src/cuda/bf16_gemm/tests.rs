use super::*;
use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_core::environment::stream::StreamId;

#[test]
fn output_stride_preserves_operands_and_checks_the_complete_view() -> Result<()> {
    let compact = Bf16Gemm::new(17, 3, 49)?;
    let strided = Bf16Gemm::with_output_stride(17, 3, 49, Bf16GemmReduction::Ordered, 83)?;
    assert_eq!(strided.grid, compact.grid);
    assert_eq!(
        strided.kernel.binding_sizes(),
        [17 * 49 * 2, 3 * 49 * 2, (2 * 83 + 17) * 2]
    );
    assert_eq!(
        Bf16Gemm::with_output_stride_certified(17, 3, 49, Bf16GemmReduction::Ordered, 83)?
            .kernel
            .binding_sizes(),
        [17 * 49 * 2, 3 * 49 * 2, (2 * 83 + 17) * 2, 4]
    );
    assert_eq!(output_span(17, 1, 83)?, 34);
    assert!(output_span(17, 3, 16).is_err());
    assert!(output_span(17, 3, usize::MAX).is_err());
    Ok(())
}

#[test]
fn geometry_checks_extents_and_tiles_without_case_whitelists() -> Result<()> {
    for (columns, rows, inner, grid) in [
        (1, 1, 1, [1, 1, 1]),
        (65, 8, 17, [2, 1, 1]),
        (65, 9, 31, [2, 1, 1]),
        (128, 65, 33, [2, 3, 1]),
        (248_320, 1, 1024, [3880, 1, 1]),
    ] {
        let gemm = Bf16Gemm::new(columns, rows, inner)?;
        assert_eq!(gemm.grid, grid);
        assert_eq!(
            gemm.kernel.binding_sizes(),
            [columns * inner * 2, rows * inner * 2, rows * columns * 2]
        );
    }
    for shape in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (usize::MAX, 1, 2),
        (1, 1, usize::MAX),
    ] {
        assert!(Bf16Gemm::new(shape.0, shape.1, shape.2).is_err());
    }
    Ok(())
}

#[test]
fn reduction_geometry_and_spans_are_checked() -> Result<()> {
    use Bf16GemmReduction::*;
    for reduction in [
        InterleavedPair { terms: 0 },
        RoundedSegments { terms: 17 },
        LaneStrided {
            lanes: 3,
            terms: 48,
        },
        LaneStrided {
            lanes: 64,
            terms: 128,
        },
        LaneStrided {
            lanes: 16,
            terms: 33,
        },
        LaneBlocked { lanes: 0, terms: 1 },
        LaneBlocked {
            lanes: 32,
            terms: usize::MAX / 16,
        },
    ] {
        assert!(Bf16Gemm::with_reduction(17, 3, 49, reduction).is_err());
    }
    for lanes in [1, 2, 4, 8, 16, 32] {
        let gemm = Bf16Gemm::with_reduction(17, 3, 49, LaneBlocked { lanes, terms: 7 })?;
        assert_eq!(gemm.grid, [17usize.div_ceil(128 / lanes) as u32, 3, 1]);
    }
    assert!(
        Bf16Gemm::with_reduction(
            1,
            65_536,
            1,
            LaneBlocked {
                lanes: 32,
                terms: 7
            }
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn output_alias_is_refused_but_disjoint_views_and_shared_reads_are_valid() -> Result<()> {
    let handle = Handle::new(
        ServiceId::of::<()>(DeviceId::new(0, 0)),
        StreamId::current(),
        32,
    );
    let left = handle.clone().offset_end(16);
    let right = handle.clone().offset_start(16);
    require_disjoint_output(&left, &left, &right)?;
    assert!(require_disjoint_output(&left, &right, &right).is_err());
    assert!(require_disjoint_output(&left, &left, &handle).is_err());
    assert!(require_disjoint_output(&handle.clone().offset_start(33), &left, &right).is_err());
    let other = Handle::new(handle.service, StreamId::current(), 32);
    assert!(require_disjoint_output(&other.clone().offset_end(33), &left, &right).is_err());
    assert!(require_disjoint_output(&left, &right, &other.offset_start(33)).is_err());
    Ok(())
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
#[test]
#[ignore = "isolated SM86 driver-only GEMM numerical/guard/retirement qualification"]
fn tensor_core_gemm_on_nvidia() -> Result<()> {
    futures_lite::future::block_on(physical_gemm(false))
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
async fn physical_gemm(reference: bool) -> Result<()> {
    use zerogravity_core::{
        device_events::{BufferRole, DeviceAccess},
        resource_arbiter::{AllocationClass, AllocationRequest},
    };
    let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("bf16-gemm")?;
    let service = diagnostic.service.clone();
    let compiler = diagnostic.compiler.clone();
    let queue = service.pinned_cuda_queue()?.clone();
    let cases = if reference {
        reference_cases()?
    } else {
        synthetic_cases()
    };
    let mut failed = 0usize;
    for TestCase {
        columns,
        rows,
        inner,
        offset,
        wb,
        xb,
        expected,
        reduction,
    } in cases
    {
        let gemm = Bf16Gemm::with_reduction(columns, rows, inner, reduction)?;
        let total = wb.len() + xb.len() + expected.len();
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: total as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(compiler.clone())?;
        gemm.prepare_code(&queue)?;
        let weights = init.upload_bytes(&wb, BufferRole::ImmutableWeights)?;
        let input = init.upload_bytes(&xb, BufferRole::Activation)?;
        let output = init.upload_bytes(&vec![0xa5; expected.len()], BufferRole::Output)?;
        drop(init.complete_sync().await?);
        slot.finish()?;

        let view = |h: Handle| h.offset_start(offset as u64).offset_end(4);
        let launch = gemm.prepare(
            service.client(),
            view(weights.handle()),
            view(input.handle()),
            view(output.handle()),
        )?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: expected.len() as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(
            0,
            &[
                (weights.clone(), DeviceAccess::Read),
                (input.clone(), DeviceAccess::Read),
                (output.clone(), DeviceAccess::Write),
            ],
        )?;
        flight.retain_completion_owner(compiler.clone())?;
        launch.launch_on(service.client())?;
        let (ready, actual) = flight
            .complete_with_readback(
                std::slice::from_ref(&output),
                &[output.readback(0, expected.len())?],
                |actual| {
                    if actual.len() != 1
                        || actual[0].len() != expected.len()
                        || actual[0][..offset] != expected[..offset]
                        || actual[0][expected.len() - 4..] != expected[expected.len() - 4..]
                    {
                        return Err(Error::State("GEMM output extent/guards differ".into()));
                    }
                    Ok(())
                },
            )
            .await?;
        let mismatches = actual[0][offset..expected.len() - 4]
            .chunks_exact(2)
            .zip(expected[offset..expected.len() - 4].chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        let first = actual[0]
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| {
                (
                    i,
                    u16::from_le_bytes([a[0], a[1]]),
                    u16::from_le_bytes([b[0], b[1]]),
                )
            });
        failed += usize::from(mismatches != 0);
        drop(ready);
        slot.finish()?;
        drop((launch, weights, input, output));
        if service.resident_bytes()? != 0 || service.model_events()? != 0 {
            return Err(Error::State("GEMM case left resident bytes/events".into()));
        }
        println!(
            "owned-tensor-core-gemm M={columns} N={rows} K={inner} offset={offset} reduction={reduction:?} mismatches={mismatches} first={first:?} guards=true retired=true"
        );
    }
    service
        .client()
        .memory_cleanup()
        .map_err(|e| Error::State(format!("GEMM cleanup: {e}")))?;
    service.close_model_namespace()?;
    if failed != 0 {
        return Err(Error::State(format!(
            "GEMM numerical mismatch in {failed} cases; all cases completed and retired"
        )));
    }
    Ok(())
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
struct TestCase {
    columns: usize,
    rows: usize,
    inner: usize,
    offset: usize,
    wb: Vec<u8>,
    xb: Vec<u8>,
    expected: Vec<u8>,
    reduction: Bf16GemmReduction,
}
#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
fn synthetic_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();
    let shapes = [
        (1, 1, 1),
        (17, 2, 7),
        (65, 8, 17),
        (65, 9, 31),
        (128, 17, 32),
        (129, 33, 33),
        (64, 64, 64),
        (96, 65, 80),
        (128, 47, 1024),
        (256, 1, 3584),
    ];
    for (columns, rows, inner) in shapes {
        for reduction in [
            Bf16GemmReduction::Ordered,
            Bf16GemmReduction::InterleavedPair { terms: 32 },
            Bf16GemmReduction::InterleavedPair { terms: 96 },
            Bf16GemmReduction::RoundedSegments { terms: 64 },
            Bf16GemmReduction::RoundedSegments { terms: 96 },
            Bf16GemmReduction::LaneStrided {
                lanes: 16,
                terms: 512,
            },
            Bf16GemmReduction::LaneStrided {
                lanes: 8,
                terms: 48,
            },
            Bf16GemmReduction::LaneBlocked {
                lanes: 32,
                terms: 32,
            },
            Bf16GemmReduction::LaneBlocked { lanes: 4, terms: 7 },
        ] {
            for offset in [0usize, 4] {
                // Signed binary fractions: every multiply and FP32 sum is exact.
                // Non-identity matrices exercise all MMA lanes, tiles and K stages.
                let value = |i: usize, seed: usize| ((i * 13 + seed * 7) % 17) as f32 / 8.0 - 1.0;
                let w = (0..columns * inner)
                    .map(|i| {
                        value(i, 3)
                            + if (i / inner).is_multiple_of(3) {
                                1.0
                            } else {
                                0.0
                            }
                    })
                    .collect::<Vec<_>>();
                let x = (0..rows * inner)
                    .map(|i| {
                        value(i, 5)
                            + if (i / inner).is_multiple_of(2) {
                                1.0
                            } else {
                                0.0
                            }
                    })
                    .collect::<Vec<_>>();
                let bf16 = |v: f32| {
                    let bits = v.to_bits();
                    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
                };
                let encode = |data: &[f32]| {
                    data.iter()
                        .flat_map(|v| bf16(*v).to_le_bytes())
                        .collect::<Vec<_>>()
                };
                let mut wb = vec![0x55; offset];
                wb.reserve_exact(w.len() * 2 + 4);
                wb.extend(encode(&w));
                wb.extend([0x55; 4]);
                let mut xb = vec![0x55; offset];
                xb.reserve_exact(x.len() * 2 + 4);
                xb.extend(encode(&x));
                xb.extend([0x55; 4]);
                let mut expected = vec![0xa5; offset];
                expected.reserve_exact(rows * columns * 2 + 4);
                for row in 0..rows {
                    for col in 0..columns {
                        let sum = match reduction {
                            Bf16GemmReduction::RoundedSegments { terms } => (0..inner)
                                .step_by(terms)
                                .map(|start| {
                                    let partial = (start..(start + terms).min(inner))
                                        .map(|k| w[col * inner + k] * x[row * inner + k])
                                        .sum::<f32>();
                                    f32::from_bits(u32::from(bf16(partial)) << 16)
                                })
                                .sum::<f32>(),
                            _ => (0..inner)
                                .map(|k| w[col * inner + k] * x[row * inner + k])
                                .sum::<f32>(),
                        };
                        expected.extend(bf16(sum).to_le_bytes());
                    }
                }
                expected.extend([0xa5; 4]);
                cases.push(TestCase {
                    columns,
                    rows,
                    inner,
                    offset,
                    wb,
                    xb,
                    expected,
                    reduction,
                });
            }
        }
    }
    cases
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
#[test]
#[ignore = "isolated driver-only GEMM versus independently captured authenticated operands"]
fn captured_projection_gemm_on_nvidia() -> Result<()> {
    futures_lite::future::block_on(physical_gemm(true))
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
fn reference_cases() -> Result<Vec<TestCase>> {
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        io::{Cursor, Read},
    };
    let path = std::env::var_os("ZG_BF16_GEMM_REFERENCE")
        .ok_or_else(|| Error::State("ZG_BF16_GEMM_REFERENCE required".into()))?;
    let metadata = fs::symlink_metadata(&path)
        .map_err(|e| Error::State(format!("GEMM fixture metadata: {e}")))?;
    if !metadata.is_file() || metadata.len() > 32 * 1024 * 1024 {
        return Err(Error::Limit(
            "GEMM fixture requires bounded regular file".into(),
        ));
    }
    let data = fs::read(path).map_err(|e| Error::State(format!("GEMM fixture read: {e}")))?;
    let digest = Sha256::digest(&data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if std::env::var("ZG_BF16_GEMM_REFERENCE_SHA256")
        .ok()
        .as_deref()
        != Some(digest.as_str())
    {
        return Err(Error::State(
            "GEMM fixture independent reference digest differs".into(),
        ));
    }
    let mut input = Cursor::new(data.as_slice());
    let mut magic = [0u8; 8];
    let io = |e| Error::State(format!("GEMM fixture decode: {e}"));
    input.read_exact(&mut magic).map_err(io)?;
    if &magic != b"ZGGM0001" {
        return Err(Error::State("GEMM fixture format".into()));
    }
    let word = |input: &mut Cursor<&[u8]>| -> Result<usize> {
        let mut b = [0u8; 8];
        input.read_exact(&mut b).map_err(io)?;
        usize::try_from(u64::from_le_bytes(b))
            .map_err(|_| Error::Limit("GEMM fixture dimension".into()))
    };
    let count = word(&mut input)?;
    if !(1..=32).contains(&count) {
        return Err(Error::Limit("GEMM fixture case count".into()));
    }
    let mut cases = Vec::new();
    for _ in 0..count {
        let columns = word(&mut input)?;
        let rows = word(&mut input)?;
        let inner = word(&mut input)?;
        let shape = Bf16GemmShape::new(columns, rows, inner)?;
        let mut payload = |size: usize, guard| -> Result<Vec<u8>> {
            if size > 32 * 1024 * 1024
                || input
                    .position()
                    .checked_add(size as u64)
                    .is_none_or(|end| end > data.len() as u64)
            {
                return Err(Error::Limit("GEMM fixture payload extent".into()));
            }
            let mut bytes = vec![guard; size + 4];
            input.read_exact(&mut bytes[..size]).map_err(io)?;
            Ok(bytes)
        };
        cases.push(TestCase {
            columns,
            rows,
            inner,
            offset: 0,
            wb: payload(shape.weight_bytes(), 0x55)?,
            xb: payload(shape.input_bytes(), 0x55)?,
            expected: payload(shape.output_bytes(), 0xa5)?,
            // Actual reference arithmetic under investigation, never a runtime selector.
            reduction: match (columns, rows, inner) {
                (8192, 47, 1024) => Bf16GemmReduction::Ordered,
                (8192, 1, 1024) => Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 512,
                },
                (32, 1, 1024) => Bf16GemmReduction::LaneBlocked {
                    lanes: 32,
                    terms: 32,
                },
                (32, 47 | 48, 1024) => Bf16GemmReduction::RoundedSegments { terms: 64 },
                (1024, 47, 2048) => Bf16GemmReduction::InterleavedPair { terms: 32 },
                (1024, 48, 2048) => Bf16GemmReduction::Ordered,
                (7168, 47 | 48, 1024) => Bf16GemmReduction::Ordered,
                (1024, 47 | 48, 3584) => Bf16GemmReduction::RoundedSegments { terms: 1792 },
                (1024, 1, 2048) => Bf16GemmReduction::RoundedSegments { terms: 704 },
                (5120, 47, 1024) => Bf16GemmReduction::InterleavedPair { terms: 512 },
                (5120, 1, 1024) => Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 512,
                },
                _ => {
                    return Err(Error::State(
                        "unexpected captured projection fixture geometry".into(),
                    ));
                }
            },
        });
    }
    if input.position() != data.len() as u64 {
        return Err(Error::State("GEMM fixture trailing bytes".into()));
    }
    Ok(cases)
}
