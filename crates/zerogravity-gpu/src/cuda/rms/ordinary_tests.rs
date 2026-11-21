//! Ordinary CUDA versus retained staged CUDA/Torch RMS, not the CPU cascade.
use super::*;
use crate::cuda::{
    ordinary_text::{Access, BufferRole as Role, Geometry, LaunchBinding, Program},
    test_support::CudaDiagnostic,
};
use zerogravity_core::{
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest},
};

#[test]
#[ignore = "isolated CUDA ordinary row/residual RMS versus independent staged Torch profile"]
fn ordinary_rms_matches_staged_cuda_on_nvidia() -> Result<()> {
    futures_lite::future::block_on(run())
}

async fn run() -> Result<()> {
    let diagnostic = CudaDiagnostic::new("cuda-rms")?;
    let service = &diagnostic.service;
    let queue = service.pinned_cuda_queue()?;
    let mut failed = 0;
    for columns in [Columns::Qwen800M, Columns::Qwen2B] {
        let width = columns.count();
        let program = Program::new(if width == 1024 {
            Geometry::qwen35_08()
        } else {
            Geometry::qwen35_2b()
        })?;
        for rows in [
            Rows::One,
            Rows::Two,
            Rows::Text { prompt_tokens: 17 },
            Rows::SixtyFour,
        ] {
            for residual in [false, true] {
                let n = rows.count() * width;
                let samples = |count: usize, patterns: &[u16], salt: usize| {
                    (0..count)
                        .flat_map(|i| patterns[(i * 23 + salt) % patterns.len()].to_le_bytes())
                        .collect::<Vec<_>>()
                };
                let input = samples(
                    n,
                    &[0x3f80, 0xbf81, 0x3f7f, 0xc001, 0x4081, 0x3800, 0xb880],
                    0,
                );
                let prior = samples(
                    n,
                    &[0x3b00, 0xbb80, 0x3c00, 0xbc80, 0x3d00, 0xbd01, 0x4001],
                    3,
                );
                let gamma = samples(width, &[0, 0x3c80, 0xbc80, 0x3e00, 0xbe00, 0x3d81], 1);
                require_non_bf16_sum(&input, &prior)?;
                let sizes = [
                    width * 2,
                    width * 4,
                    width * 4,
                    n * 2,
                    n * 2,
                    n * 4,
                    n * 4,
                    rows.count() * 4,
                    rows.count() * 4,
                    rows.count() * 4,
                    n * 4,
                    n * 4,
                    n * 2,
                    n * 2,
                    n * 2,
                    n * 2,
                    4,
                ];
                let mut data = sizes
                    .into_iter()
                    .map(|size| vec![0xa5; size + 32])
                    .collect::<Vec<_>>();
                for (i, bytes) in [(0, &gamma), (3, &input), (4, &prior), (14, &prior)] {
                    data[i][16..16 + bytes.len()].copy_from_slice(bytes);
                }
                data[16][16..20].fill(0);
                let total = data.iter().map(|b| b.len() as u64).sum();
                let ticket = service.queue().submit(
                    service.budget(),
                    service.owner().clone(),
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: total,
                    }],
                    None,
                )?;
                let mut slot = ticket.wait().await?;
                slot.begin_submission()?;
                let mut init = service.begin_initialization(0)?;
                init.retain_completion_owner(diagnostic.compiler.clone())?;
                let leases = data
                    .iter()
                    .map(|b| init.upload_bytes(b, BufferRole::Scratch))
                    .collect::<Result<Vec<_>>>()?;
                let view = |i: usize| leases[i].handle().offset_start(16).offset_end(16);
                let mut launches = Vec::new();
                for stage in Stage::ORDER {
                    let spec = if residual {
                        fused::fused_specification(stage, rows, columns)?
                    } else {
                        specification_for(stage, rows, columns)?
                    };
                    spec.kernel.prepare_code(queue)?;
                    let indices: &[usize] = match stage {
                        Stage::GammaWiden => &[0, 1],
                        Stage::GammaShift => &[1, 2],
                        Stage::InputWiden if residual => &[3, 4, 5, 13],
                        Stage::InputWiden => &[3, 5],
                        Stage::Square => &[5, 6],
                        Stage::Mean => &[6, 7],
                        Stage::Epsilon => &[7, 8],
                        Stage::Inverse => &[8, 9],
                        Stage::Normalize => &[5, 9, 10],
                        Stage::Weight => &[10, 2, 11],
                        Stage::Bf16Store => &[11, 12],
                    };
                    launches.push(spec.kernel.prepare(
                        service.client(),
                        spec.grid,
                        indices.iter().map(|&i| view(i)).collect(),
                    )?);
                }
                let definition = if residual {
                    program.residual_rms(rows.count(), 1e-6)?
                } else {
                    program.rms(rows.count(), 1e-6)?
                };
                let mut bindings = vec![LaunchBinding::new(Role::Input, Access::Read, view(3))];
                if residual {
                    bindings.push(LaunchBinding::new(
                        Role::Residual,
                        Access::ReadWrite,
                        view(14),
                    ));
                }
                bindings.extend([
                    LaunchBinding::new(Role::RawGamma, Access::Read, view(0)),
                    LaunchBinding::new(Role::Output, Access::Write, view(15)),
                ]);
                let candidate = definition.prepare_owned(
                    service.client(),
                    bindings,
                    LaunchBinding::new(Role::Status, Access::ReadWrite, view(16)),
                )?;
                drop(init.complete_sync().await?);
                slot.finish()?;
                let ticket = service.queue().submit(
                    service.budget(),
                    service.owner().clone(),
                    &[AllocationRequest {
                        kind: AllocationClass::Output,
                        bytes: total,
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
                            if [0, 3, 4].contains(&i) {
                                DeviceAccess::Read
                            } else {
                                DeviceAccess::Write
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let mut flight = service.submit(0, &accesses)?;
                flight.retain_completion_owner(diagnostic.compiler.clone())?;
                for launch in &launches {
                    launch.launch_on(service.client())?;
                }
                candidate.enqueue_owned(service.client(), &view(16))?;
                let reads = leases
                    .iter()
                    .map(|l| l.readback(0, l.bytes()))
                    .collect::<Result<Vec<_>>>()?;
                let (ready, actual) = flight
                    .complete_with_readback(&leases[12..], &reads, |a| {
                        if a.len() != data.len() {
                            return Err(Error::State("RMS cohort extent".into()));
                        }
                        for i in 0..data.len() {
                            if a[i].len() != data[i].len()
                                || a[i][..16] != data[i][..16]
                                || a[i][data[i].len() - 16..] != data[i][data[i].len() - 16..]
                                || ([0, 3, 4].contains(&i) && a[i].as_ref() != data[i])
                            {
                                return Err(Error::State(
                                    "RMS guard or immutable input changed".into(),
                                ));
                            }
                        }
                        Ok(())
                    })
                    .await?;
                let output_diff = actual[12][16..16 + n * 2]
                    .chunks_exact(2)
                    .zip(actual[15][16..16 + n * 2].chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count();
                let residual_diff = if residual {
                    actual[13][16..16 + n * 2]
                        .chunks_exact(2)
                        .zip(actual[14][16..16 + n * 2].chunks_exact(2))
                        .filter(|(a, b)| a != b)
                        .count()
                } else {
                    0
                };
                let status = u32::from_le_bytes(actual[16][16..20].try_into().unwrap());
                failed += usize::from(output_diff != 0 || residual_diff != 0 || status != 0);
                drop(ready);
                slot.finish()?;
                drop((launches, candidate, accesses, reads, leases));
                if service.resident_bytes()? != 0 || service.model_events()? != 0 {
                    return Err(Error::State("RMS residual lease/event".into()));
                }
                println!(
                    "ordinary-cuda-rms width={width} rows={} residual={residual} output_diff={output_diff} residual_diff={residual_diff} status={status} guards=true retired=true",
                    rows.count()
                );
            }
        }
    }
    service
        .client()
        .memory_cleanup()
        .map_err(|e| Error::State(format!("RMS cleanup: {e}")))?;
    service.close_model_namespace()?;
    if failed != 0 {
        return Err(Error::State(format!(
            "CUDA RMS mismatch in {failed} cases; all retired"
        )));
    }
    Ok(())
}
fn require_non_bf16_sum(input: &[u8], residual: &[u8]) -> Result<()> {
    let value =
        |b: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16);
    if !input
        .chunks_exact(2)
        .zip(residual.chunks_exact(2))
        .any(|(a, b)| (value(a) + value(b)).to_bits() & 0xffff != 0)
    {
        return Err(Error::State(
            "RMS fixture cannot distinguish premature residual rounding".into(),
        ));
    }
    Ok(())
}
