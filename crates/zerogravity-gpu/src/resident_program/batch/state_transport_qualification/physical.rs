//! Physical original-per-row vs native bit copy, with original leases/Ready.
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_state_transport::*;
use cubecl::{CubeElement, Device, bytes::Bytes, server::Handle};
use std::{path::Path, sync::Arc};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::LoadedQwenModel;

const PAD: usize = 4;
const SENTINEL: u32 = 0xa5a55a5a;
const SPECIAL: [u32; 10] = [
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000,
    0x7fc12345,
];

fn value(row: usize, channel: usize, at: usize) -> u32 {
    if at % 16 < SPECIAL.len() {
        SPECIAL[(at + row + channel) % SPECIAL.len()]
    } else {
        0x3f000000
            | (((at as u32).wrapping_mul(73)
                ^ ((row as u32 + 1) * 0x10001)
                ^ ((channel as u32 + 1) * 0x7001))
                & 0x007fffff)
    }
}

fn view(lease: &ResidentLease, words: usize) -> Handle {
    lease
        .handle()
        .offset_start((PAD * 4) as u64)
        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
}

fn rows(handles: &[Handle], channels: usize) -> Vec<Row<'_>> {
    handles
        .chunks_exact(channels)
        .map(|group| Row {
            first: &group[0],
            second: (channels == 2).then(|| &group[1]),
        })
        .collect()
}

fn refusal(result: Result<PreparedTransport>, text: &str) -> Result<()> {
    match result {
        Err(error) if error.to_string().contains(text) => Ok(()),
        _ => Err(Error::State(format!(
            "state transport expected prelaunch refusal: {text}"
        ))),
    }
}

fn compare(read: &[Bytes], batch: usize, channels: usize, shape: Shape) -> Result<usize> {
    let (words, total) = geometry(shape, batch)?;
    let expected_count = 1 + batch * channels * 3 + channels * 2;
    if read.len() != expected_count || read[0].len() != 4 || read[0][..] != [0, 0, 0, 0] {
        return Err(Error::State(
            "state transport complete status/read coverage".into(),
        ));
    }
    let mut checked = 0;
    let mut check = |bytes: &Bytes, count: usize, expected: &dyn Fn(usize) -> u32| -> Result<()> {
        if bytes.len() != (count + 2 * PAD) * 4 {
            return Err(Error::State(
                "state transport complete canary readback".into(),
            ));
        }
        for (at, raw) in bytes.chunks_exact(4).enumerate() {
            let want = if at < PAD || at >= PAD + count {
                SENTINEL
            } else {
                expected(at - PAD)
            };
            if raw != want.to_ne_bytes() {
                return Err(Error::State(format!(
                    "state transport physical bit/canary difference at{at}"
                )));
            }
            checked += 1;
        }
        Ok(())
    };
    let mut index = 1;
    for _private_set in 0..3 {
        for row in 0..batch {
            for channel in 0..channels {
                check(&read[index], words, &|at| value(row, channel, at))?;
                index += 1;
            }
        }
    }
    for _packed_set in 0..2 {
        for channel in 0..channels {
            check(&read[index], total, &|at| {
                let (row, local) = match shape {
                    Shape::Vector { .. } => (at / words, at % words),
                    Shape::Kv { slab, .. } => (
                        (at / slab) % batch,
                        (at / (slab * batch)) * slab + at % slab,
                    ),
                };
                value(row, channel, local)
            })?;
            index += 1;
        }
    }
    if index != read.len() {
        return Err(Error::State(
            "state transport omitted physical row/channel".into(),
        ));
    }
    Ok(checked)
}

async fn case(
    service: &Arc<ResidentService>,
    batch: usize,
    shape: Shape,
) -> Result<(usize, usize, usize)> {
    let (words, total) = geometry(shape, batch)?;
    let channels = if matches!(shape, Shape::Kv { .. }) {
        2
    } else {
        1
    };
    let private_bytes = (words + 2 * PAD) * 4;
    let packed_bytes = (total + 2 * PAD) * 4;
    let bytes = private_bytes * batch * channels * 3 + packed_bytes * channels * 2 + 4;
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
        return Err(Error::Limit("state transport fixture envelope".into()));
    }
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let status = init.upload_bytes(u32::as_bytes(&[0u32]), BufferRole::Scratch)?;
    let mut input = Vec::with_capacity(batch * channels);
    for row in 0..batch {
        for channel in 0..channels {
            let mut data = vec![SENTINEL; words + 2 * PAD];
            for at in 0..words {
                data[PAD + at] = value(row, channel, at);
            }
            input.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
        }
    }
    let mut output = Vec::with_capacity(2 * batch * channels + 2 * channels);
    for _ in 0..2 * batch * channels {
        output.push(init.upload_bytes(
            u32::as_bytes(&vec![SENTINEL; words + 2 * PAD]),
            BufferRole::Scratch,
        )?);
    }
    for _ in 0..2 * channels {
        output.push(init.upload_bytes(
            u32::as_bytes(&vec![SENTINEL; total + 2 * PAD]),
            BufferRole::Scratch,
        )?);
    }
    let initialized = init.complete_sync().await?;
    if initialized.outputs().len() != input.len() + output.len() + 1 {
        return Err(Error::State(
            "state transport initialization coverage".into(),
        ));
    }
    drop(initialized);
    let client = service.client();
    let input_handles = input
        .iter()
        .map(|lease| view(lease, words))
        .collect::<Vec<_>>();
    let old_private = output[..batch * channels]
        .iter()
        .map(|lease| view(lease, words))
        .collect::<Vec<_>>();
    let new_private = output[batch * channels..2 * batch * channels]
        .iter()
        .map(|lease| view(lease, words))
        .collect::<Vec<_>>();
    let packed = output[2 * batch * channels..]
        .iter()
        .map(|lease| view(lease, total))
        .collect::<Vec<_>>();
    let input_rows = rows(&input_handles, channels);
    let old_rows = rows(&old_private, channels);
    let new_rows = rows(&new_private, channels);
    let old_k = &packed[0];
    let old_v = (channels == 2).then(|| &packed[1]);
    let new_k = &packed[channels];
    let new_v = (channels == 2).then(|| &packed[channels + 1]);
    let gather =
        PreparedTransport::prepare(client, &input_rows, new_k, new_v, shape, Direction::Gather)?;
    let scatter =
        PreparedTransport::prepare(client, &new_rows, new_k, new_v, shape, Direction::Scatter)?;
    let stable_bytes = service.resident_bytes()?;
    let stable_uploads = service.uploaded_value_bytes();
    let mut foreign = input_handles[0].clone();
    foreign.service.service = std::any::TypeId::of::<()>();
    let mut invalid_rows = rows(&input_handles, channels);
    invalid_rows[0].first = &foreign;
    refusal(
        PreparedTransport::prepare(
            client,
            &invalid_rows,
            new_k,
            new_v,
            shape,
            Direction::Gather,
        ),
        "service",
    )?;
    invalid_rows = rows(&input_handles, channels);
    invalid_rows[batch - 1].first = invalid_rows[0].first;
    refusal(
        PreparedTransport::prepare(
            client,
            &invalid_rows,
            new_k,
            new_v,
            shape,
            Direction::Gather,
        ),
        "aliased",
    )?;
    invalid_rows = rows(&input_handles, channels);
    invalid_rows[0].first = new_k; // Geometry is large enough: exercise the alias guard.
    refusal(
        PreparedTransport::prepare(
            client,
            &invalid_rows,
            new_k,
            new_v,
            shape,
            Direction::Gather,
        ),
        "aliased",
    )?;
    let mut bad_end = input_handles[0].clone();
    bad_end.offset_end = Some(bad_end.size() + 1);
    invalid_rows = rows(&input_handles, channels);
    invalid_rows[0].first = &bad_end;
    refusal(
        PreparedTransport::prepare(
            client,
            &invalid_rows,
            new_k,
            new_v,
            shape,
            Direction::Gather,
        ),
        "end exceeds",
    )?;
    let mut bad_start = input_handles[0].clone();
    bad_start.offset_start = Some(bad_start.size());
    invalid_rows[0].first = &bad_start;
    refusal(
        PreparedTransport::prepare(
            client,
            &invalid_rows,
            new_k,
            new_v,
            shape,
            Direction::Gather,
        ),
        "start exceeds",
    )?;
    let mut short = input_handles[0].clone();
    short.offset_end = Some(short.size() - (PAD + words - 1) as u64 * 4);
    invalid_rows[0].first = &short;
    refusal(
        PreparedTransport::prepare(
            client,
            &invalid_rows,
            new_k,
            new_v,
            shape,
            Direction::Gather,
        ),
        "alignment/size",
    )?;
    if service.resident_bytes()? != stable_bytes
        || service.uploaded_value_bytes() != stable_uploads
        || service.pending_events()? != 0
    {
        return Err(Error::State(
            "state transport prelaunch refusal allocated/uploaded/submitted".into(),
        ));
    }
    let mut accesses = input
        .iter()
        .cloned()
        .map(|lease| (lease, DeviceAccess::Read))
        .collect::<Vec<_>>();
    accesses.extend(
        output
            .iter()
            .cloned()
            .map(|lease| (lease, DeviceAccess::Write)),
    );
    accesses.push((status.clone(), DeviceAccess::Write));
    let mut promoted = output.clone();
    promoted.push(status.clone());
    let ordered = std::iter::once(&status)
        .chain(&input)
        .chain(&output)
        .collect::<Vec<_>>();
    let reads = ordered
        .iter()
        .map(|lease| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let flight = service.submit(0, &accesses)?;
    let original = super::launch_original(
        client,
        &input_rows,
        old_k,
        old_v,
        shape,
        Direction::Gather,
        &status.handle(),
    )?;
    let native = gather.launch(client)?;
    let original = original
        + super::launch_original(
            client,
            &old_rows,
            old_k,
            old_v,
            shape,
            Direction::Scatter,
            &status.handle(),
        )?;
    let native = native + scatter.launch(client)?;
    let mut checked = 0;
    let (ready, data) = flight
        .complete_with_readback(&promoted, &reads, |read| {
            checked = compare(read, batch, channels, shape)?;
            Ok(())
        })
        .await?;
    slot.finish()?;
    if original != 2 * batch
        || native != 2
        || service.uploaded_value_bytes() != stable_uploads
        || ready.outputs().len() != promoted.len()
        || data.len() != ordered.len()
    {
        return Err(Error::State(
            "state transport physical count/completion/upload coverage".into(),
        ));
    }
    drop(ready);
    Ok((checked, original, native))
}

pub(crate) async fn qualify_words(device: Device, path: &Path, device_cap: u64) -> Result<()> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "state transport physical qualification requires release CUDA".into(),
        ));
    }
    let directory = path
        .parent()
        .ok_or_else(|| Error::Malformed("state transport model directory".into()))?;
    let source = LoadedQwenModel::open(path, &directory.join("config.json"))?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("state-bit-transport", "original-per-row-differential")?,
        256 * 1024 * 1024,
        device_cap,
    )?;
    let key = service.physical_key();
    let mut totals = (0, 0, 0);
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
            let actual = case(&service, batch, shape).await?;
            totals.0 += actual.0;
            totals.1 += actual.1;
            totals.2 += actual.2;
            if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
                return Err(Error::State(
                    "state transport fixture retained private leases/events".into(),
                ));
            }
        }
    }
    drop(service);
    let final_budget = ResourceArbiter::process_shared().snapshot(&key)?;
    if final_budget.reserved != 0 || final_budget.fenced {
        return Err(Error::State(
            "state transport final original physical envelope".into(),
        ));
    }
    println!(
        "CUDA_STATE_BIT_TRANSPORT_PASS B=2,4,8 compared_words={} original_per_row_dispatches={} native_domain_dispatches={} exact_U32_signedzero_subnormal_nonfinite_payload=1 prefix_suffix_canaries=unchanged old_physical_kernels=independent original_lease_Ready_queue=1 warm_uploads=0 final_reserved=0 published_roots=0 fullmodel_graph_worker_HTTP_pressure_abort_speed=separate_gates",
        totals.0, totals.1, totals.2
    );
    Ok(())
}

#[test]
#[ignore = "release CUDA original0.8B physical U32 transport; isolated GPU coordinator only"]
fn original_qwen35_08b_native_state_word_transport() -> Result<()> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf");
    futures_lite::future::block_on(qualify_words(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &path,
        5120 * 1024 * 1024,
    ))
}

#[test]
#[ignore = "release CUDA original2B physical U32 transport; isolated GPU coordinator only"]
fn original_qwen35_2b_native_state_word_transport() -> Result<()> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf");
    futures_lite::future::block_on(qualify_words(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        &path,
        5120 * 1024 * 1024,
    ))
}
