//! ROOT-only real CUDA differential: original copy +original finite certificate.
//! Synthetic state words only; trained all42/fullV/worker gates are separate.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_state_transport::{PreparedCertifiedScatter, geometry};
use cubecl::{CubeElement, Device, bytes::Bytes};
use std::{path::Path, sync::Arc};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::LoadedQwenModel;

const PAD: usize = 4;
const CANARY: u32 = 0xa5a55a5a;
const FINITE: [u32; 8] = [
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
];
const NONFINITE: [u32; 4] = [0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345];

fn word(
    row: usize,
    channel: usize,
    at: usize,
    words: usize,
    pattern: usize,
    channels: usize,
) -> u32 {
    if pattern >= 2 && row * channels + channel == pattern - 2 && at + 1 == words {
        NONFINITE[(pattern - 2) % NONFINITE.len()]
    } else if at % 16 < FINITE.len() {
        FINITE[(at + row + channel) % FINITE.len()]
    } else {
        0x3f000000 | ((at as u32).wrapping_mul(73) & 0x007fffff)
    }
}
fn visible(lease: &ResidentLease, words: usize) -> Handle {
    lease
        .handle()
        .offset_start((PAD * 4) as u64)
        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
}
fn rows(handles: &[Handle], channels: usize) -> Vec<Row<'_>> {
    handles
        .chunks_exact(channels)
        .map(|h| Row {
            first: &h[0],
            second: (channels == 2).then(|| &h[1]),
        })
        .collect()
}
fn refuses(value: Result<PreparedCertifiedScatter>, text: &str) -> Result<()> {
    if !matches!(value, Err(ref e) if e.to_string().contains(text)) {
        return Err(Error::State(format!(
            "certified scatter expected cold refusal: {text}"
        )));
    }
    Ok(())
}
fn check_words(read: &Bytes, words: usize, value: impl Fn(usize) -> u32) -> Result<usize> {
    if read.len() != (words + 2 * PAD) * 4 {
        return Err(Error::State(
            "certified scatter complete canary byte count".into(),
        ));
    }
    for (at, raw) in read.chunks_exact(4).enumerate() {
        let expected = if at < PAD || at >= PAD + words {
            CANARY
        } else {
            value(at - PAD)
        };
        if raw != expected.to_ne_bytes() {
            return Err(Error::State(format!(
                "certified scatter raw bit/canary mismatch at{at}"
            )));
        }
    }
    Ok(words + 2 * PAD)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stop {
    None,
    Abandon,
    RejectStatus,
}

async fn case(
    service: &Arc<ResidentService>,
    batch: usize,
    shape: Shape,
    pattern: usize,
    stop: Stop,
) -> Result<(usize, usize, usize)> {
    let (words, total) = geometry(shape, batch)?;
    let channels = 1 + usize::from(matches!(shape, Shape::Kv { .. }));
    if pattern >= batch * channels + 2 {
        return Err(Error::Malformed("certified scatter fixture pattern".into()));
    }
    let seed = if pattern == 0 {
        0
    } else {
        crate::program_kernels::VIEW_OUT_OF_BOUNDS | crate::program_kernels::INVALID_REDUCTION
    };
    let expected_status = seed
        | if pattern >= 2 {
            crate::program_kernels::NONFINITE_VALUE
        } else {
            0
        };
    let bytes = (words + 2 * PAD) * 4 * batch * channels * 3
        + (total + 2 * PAD) * 4 * channels
        + (1 + 2 * PAD) * 4 * 2;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    if bytes > service.remaining_capacity()? {
        return Err(Error::Limit(
            "certified scatter fixture physical envelope".into(),
        ));
    }
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let mut input = Vec::new();
    for row in 0..batch {
        for channel in 0..channels {
            let mut data = vec![CANARY; words + 2 * PAD];
            for (at, value) in data[PAD..PAD + words].iter_mut().enumerate() {
                *value = word(row, channel, at, words, pattern, channels);
            }
            input.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
        }
    }
    let mut output = Vec::new();
    for _ in 0..2 * batch * channels {
        output.push(init.upload_bytes(
            u32::as_bytes(&vec![CANARY; words + 2 * PAD]),
            BufferRole::Scratch,
        )?);
    }
    for _ in 0..channels {
        output.push(init.upload_bytes(
            u32::as_bytes(&vec![CANARY; total + 2 * PAD]),
            BufferRole::Scratch,
        )?);
    }
    let mut status = Vec::new();
    for _ in 0..2 {
        let mut data = [CANARY; 1 + 2 * PAD];
        data[PAD] = seed;
        status.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
    }
    let initialized = init.complete_sync().await?;
    if initialized.outputs().len() != input.len() + output.len() + status.len() {
        return Err(Error::State(
            "certified scatter initialized lease coverage".into(),
        ));
    }
    drop(initialized);
    let client = service.client();
    let inputs = input.iter().map(|l| visible(l, words)).collect::<Vec<_>>();
    let old_private = output[..batch * channels]
        .iter()
        .map(|l| visible(l, words))
        .collect::<Vec<_>>();
    let new_private = output[batch * channels..2 * batch * channels]
        .iter()
        .map(|l| visible(l, words))
        .collect::<Vec<_>>();
    let dense = output[2 * batch * channels..]
        .iter()
        .map(|l| visible(l, total))
        .collect::<Vec<_>>();
    let statuses = status.iter().map(|l| visible(l, 1)).collect::<Vec<_>>();
    let input_rows = rows(&inputs, channels);
    let old_rows = rows(&old_private, channels);
    let new_rows = rows(&new_private, channels);
    let dense_v = (channels == 2).then(|| &dense[1]);
    // BOTH complete old/new span+status bindings pass before the original flight.
    PreparedCertifiedScatter::prepare(client, &old_rows, &dense[0], dense_v, shape, &statuses[0])?;
    let prepared = PreparedCertifiedScatter::prepare(
        client,
        &new_rows,
        &dense[0],
        dense_v,
        shape,
        &statuses[1],
    )?;
    PreparedTransport::prepare(
        client,
        &input_rows,
        &dense[0],
        dense_v,
        shape,
        Direction::Gather,
    )?;
    let bytes_before = service.resident_bytes()?;
    let uploads_before = service.uploaded_value_bytes();
    let mut foreign = statuses[1].clone();
    foreign.service.service = std::any::TypeId::of::<()>();
    refuses(
        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &foreign),
        "service",
    )?;
    refuses(
        PreparedCertifiedScatter::prepare(
            client,
            &new_rows,
            &dense[0],
            dense_v,
            shape,
            &new_private[0],
        ),
        "aliases",
    )?;
    refuses(
        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &dense[0]),
        "aliases",
    )?;
    let mut short = statuses[1].clone();
    short.offset_end = Some(short.size() - (PAD * 4) as u64);
    refuses(
        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &short),
        "alignment/size",
    )?;
    let mut misaligned = statuses[1].clone();
    misaligned.offset_start = Some((PAD * 4 + 1) as u64);
    refuses(
        PreparedCertifiedScatter::prepare(
            client,
            &new_rows,
            &dense[0],
            dense_v,
            shape,
            &misaligned,
        ),
        "alignment/size",
    )?;
    let mut aliased = rows(&new_private, channels);
    aliased[batch - 1].first = aliased[0].first;
    refuses(
        PreparedCertifiedScatter::prepare(
            client,
            &aliased,
            &dense[0],
            dense_v,
            shape,
            &statuses[1],
        ),
        "aliased",
    )?;
    if service.resident_bytes()? != bytes_before
        || service.uploaded_value_bytes() != uploads_before
        || service.pending_events()? != 0
    {
        return Err(Error::State(
            "certified scatter cold checks allocated/uploaded/submitted".into(),
        ));
    }
    // Initialization has already completed. The actual work uses a fresh original
    // selected slot; no status is cleared or synthesized after its seed upload.
    slot.finish()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 4,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    let mut accesses = input
        .iter()
        .cloned()
        .map(|l| (l, DeviceAccess::Read))
        .collect::<Vec<_>>();
    accesses.extend(
        output
            .iter()
            .chain(&status)
            .cloned()
            .map(|l| (l, DeviceAccess::Write)),
    );
    let promoted = output.iter().chain(&status).cloned().collect::<Vec<_>>();
    let order = input
        .iter()
        .chain(&output)
        .chain(&status)
        .collect::<Vec<_>>();
    let reads = order
        .iter()
        .map(|l| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
    slot.begin_submission()?;
    let flight = service.submit(0, &accesses)?;
    let gathered = super::launch_original(
        client,
        &input_rows,
        &dense[0],
        dense_v,
        shape,
        Direction::Gather,
        &statuses[0],
    )?;
    let mut original = super::launch_original(
        client,
        &old_rows,
        &dense[0],
        dense_v,
        shape,
        Direction::Scatter,
        &statuses[0],
    )?;
    for handle in &old_private {
        for offset in (0..words).step_by(crate::program_kernels::MAX_WORD_OPS) {
            let count = (words - offset).min(crate::program_kernels::MAX_WORD_OPS);
            // SAFETY: independent original preflight covers these complete
            // original views; original flight retains every accessed lease.
            unsafe {
                crate::program_kernels::certify_finite_strided::launch_unchecked(
                    client,
                    grid(count),
                    threads(),
                    arg(handle, words),
                    arg(&statuses[0], 1),
                    offset as u32,
                    count as u32,
                    1,
                    count,
                );
            }
            original += 1;
        }
    }
    let native = prepared.launch(client)?;
    if stop == Stop::Abandon {
        // Real original commands are enqueued; omit ANY trusted terminal
        // readback. Unknown completion must retain original handles and charge.
        drop(flight);
        drop(slot);
        let state = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
        if !state.fenced
            || state.reserved == 0
            || service.pending_events()? == 0
            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
            || service.ensure_usable().is_ok()
        {
            return Err(Error::State(
                "certified scatter unknown completion not quarantined/fenced".into(),
            ));
        }
        return Ok((0, original + gathered, native));
    }
    if stop == Stop::RejectStatus {
        let mut observed = false;
        let result = flight
            .complete_with_readback(&promoted, &reads, |read| {
                if expected_status & crate::program_kernels::NONFINITE_VALUE == 0
                    || read.len() != order.len()
                {
                    return Err(Error::State(
                        "certified scatter finite-status fixture coverage".into(),
                    ));
                }
                for bytes in &read[read.len() - 2..] {
                    check_words(bytes, 1, |_| expected_status)?;
                }
                observed = true;
                // SAME original owner policy: no Ready follows a nonzero final
                // certificate. Its armed flight conservatively quarantines/fences.
                Err(Error::State(
                    "certified scatter original finite-status rejected".into(),
                ))
            })
            .await;
        drop(slot);
        if !observed
            || !matches!(result, Err(ref e) if e.to_string().contains("original finite-status rejected"))
            || service.ensure_usable().is_ok()
            || service.pending_events()? == 0
            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
        {
            return Err(Error::State(
                "certified scatter finite-status produced Ready or escaped quarantine".into(),
            ));
        }
        return Ok((0, original + gathered, native));
    }
    let mut checked = 0;
    let (ready, read) = flight
        .complete_with_readback(&promoted, &reads, |read| {
            if read.len() != input.len() + output.len() + 2 {
                return Err(Error::State(
                    "certified scatter all readback bindings".into(),
                ));
            }
            let mut index = 0;
            for _set in 0..3 {
                for row in 0..batch {
                    for channel in 0..channels {
                        checked += check_words(&read[index], words, |at| {
                            word(row, channel, at, words, pattern, channels)
                        })?;
                        index += 1;
                    }
                }
            }
            for channel in 0..channels {
                checked += check_words(&read[index], total, |at| {
                    let (row, local) = match shape {
                        Shape::Vector { .. } => (at / words, at % words),
                        Shape::Kv { slab, .. } => (
                            (at / slab) % batch,
                            (at / (slab * batch)) * slab + at % slab,
                        ),
                    };
                    word(row, channel, local, words, pattern, channels)
                })?;
                index += 1;
            }
            for _ in 0..2 {
                checked += check_words(&read[index], 1, |_| expected_status)?;
                index += 1;
            }
            if index != read.len() {
                return Err(Error::State("certified scatter omitted readback".into()));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    if original != batch * (1 + channels)
        || gathered != batch
        || native != channels
        || ready.outputs().len() != promoted.len()
        || read.len() != order.len()
        || service.uploaded_value_bytes() != uploads_before
    {
        return Err(Error::State(
            "certified scatter exact launch/Ready/upload accounting".into(),
        ));
    }
    drop(ready);
    Ok((checked, original, native))
}

async fn qualify(device: Device, path: &Path, stop: Stop) -> Result<()> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "certified scatter qualification requires release CUDA".into(),
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("approved source directory".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new(
            "certified-state-scatter",
            "synthetic-original-certificate-differential",
        )?,
        256 * 1024 * 1024,
        5120 * 1024 * 1024,
    )?;
    let key = service.physical_key();
    if stop != Stop::None {
        let pattern = if stop == Stop::RejectStatus { 2 } else { 0 };
        let shape = if stop == Stop::Abandon {
            Shape::Kv {
                slab: 512,
                capacity: 512,
            }
        } else {
            Shape::Vector { words: 262_144 }
        };
        case(&service, 8, shape, pattern, stop).await?;
        drop(service);
        let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
        if !final_state.fenced || final_state.reserved == 0 {
            return Err(Error::State(
                "certified scatter unknown original envelope released".into(),
            ));
        }
        if stop == Stop::Abandon {
            println!(
                "CUDA_CERTIFIED_SCATTER_UNKNOWN_PASS B8 KV_fields=2 original_commands_enqueued=1 terminal_readback=omitted pending_event=retained handles=quarantined service=fenced original_envelope=retained sync_or_recovery=not_claimed roots_published=0"
            );
        } else {
            println!(
                "CUDA_CERTIFIED_SCATTER_STATUS_REFUSAL_PASS B8 original_nonfinite_status=observed old_and_new_status_bits=exact trusted_numerical_Ready=refused pending_event=retained handles=quarantined service=fenced original_envelope=retained roots_published=0"
            );
        }
        return Ok(());
    }
    let mut counts = (0usize, 0usize, 0usize, 0usize);
    for batch in [2, 4, 8] {
        for shape in [
            Shape::Vector { words: 13 },
            Shape::Vector { words: 18_432 },
            Shape::Vector { words: 262_144 },
            Shape::Kv {
                slab: 5,
                capacity: 3,
            },
            Shape::Kv {
                slab: 512,
                capacity: 512,
            },
        ] {
            let channels = 1 + usize::from(matches!(shape, Shape::Kv { .. }));
            for pattern in 0..batch * channels + 2 {
                let got = case(&service, batch, shape, pattern, Stop::None).await?;
                counts.0 += 1;
                counts.1 += got.0;
                counts.2 += got.1;
                counts.3 += got.2;
                if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
                    return Err(Error::State(
                        "certified scatter private lease/event not reclaimed".into(),
                    ));
                }
            }
        }
    }
    drop(service);
    let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
    if final_state.reserved != 0 || final_state.fenced {
        return Err(Error::State(
            "certified scatter final physical original envelope".into(),
        ));
    }
    println!(
        "CUDA_CERTIFIED_SCATTER_PASS B=2,4,8 cases={} words={} original_scatter_plus_certificate_launchsites={} fused_field_launchsites={} status=byte_exact_OR finite_signedzero_subnormal_Inf_NaN=preserved all_rows_fields=checked prefix_suffix_status_canaries=unchanged original_queue_leases_Ready=1 warm_uploads=0 final_reserved=0 roots_published=0 all42_fullmodel_worker_graph_speed=separate_gates",
        counts.0, counts.1, counts.2, counts.3
    );
    Ok(())
}

#[test]
#[ignore = "release original CUDA operator only; isolated GPU coordinator"]
fn original_qwen35_08b_certified_state_scatter() -> Result<()> {
    futures_lite::future::block_on(qualify(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
        Stop::None,
    ))
}
#[test]
#[ignore = "release original CUDA operator only; isolated GPU coordinator"]
fn original_qwen35_2b_certified_state_scatter() -> Result<()> {
    futures_lite::future::block_on(qualify(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
        Stop::None,
    ))
}
#[cfg(feature = "hardware-abort-probe")]
#[test]
#[ignore = "release unknown completion fences device; ONE fresh process only"]
fn original_qwen35_08b_certified_state_scatter_abandon() -> Result<()> {
    futures_lite::future::block_on(qualify(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
        Stop::Abandon,
    ))
}
#[cfg(feature = "hardware-abort-probe")]
#[test]
#[ignore = "release unknown completion fences device; ONE fresh process only"]
fn original_qwen35_2b_certified_state_scatter_abandon() -> Result<()> {
    futures_lite::future::block_on(qualify(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
        Stop::Abandon,
    ))
}

#[test]
#[ignore = "release nonfinite status refuses Ready and fences device; ONE fresh process"]
fn original_qwen35_08b_certified_state_scatter_status_refusal() -> Result<()> {
    futures_lite::future::block_on(qualify(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
        Stop::RejectStatus,
    ))
}
#[test]
#[ignore = "release nonfinite status refuses Ready and fences device; ONE fresh process"]
fn original_qwen35_2b_certified_state_scatter_status_refusal() -> Result<()> {
    futures_lite::future::block_on(qualify(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
        Stop::RejectStatus,
    ))
}
