//! Source-real imported prefixes; unequal native positions, packet feedback and refill.
//! Prefix hydration is accepted CPU math, not continuous GPU prefill qualification.
use super::*;
use zerogravity_core::resource_arbiter::ResourceArbiter;
use zerogravity_model_qwen35::{QwenCheckpoint, QwenDenseCpu};

const SEEDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];
const POSITIONS: [usize; 8] = [255, 256, 0, 31, 32, 1, 510, 511];
const PROVIDER_BYTES: usize = 4 * 1024 * 1024 * 1024;
const DEVICE_BYTES: u64 = 5 * 1024 * 1024 * 1024;

struct Lane {
    cpu: QwenDenseCpu,
    single: ResidentSession,
    native: ResidentSession,
}

fn selected(logits: &[f32]) -> (u32, u32) {
    assert_eq!(logits.len(), 248_320);
    assert!(logits.iter().all(|value| value.is_finite()));
    let id = (0..logits.len())
        .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
        .expect("complete source vocabulary");
    (id as u32, logits[id].to_bits())
}

async fn hydrated(
    model: &Arc<ResidentModel>,
    source: &Arc<LoadedQwenModel>,
    checkpoint: &QwenCheckpoint,
) -> Result<Lane> {
    let mut cpu = QwenDenseCpu::from_shared(Arc::clone(source))?;
    cpu.restore_checkpoint(checkpoint)?;
    let expected = cpu.snapshot().domain_values();
    assert_eq!(expected.len(), 42);
    let rebuilt = source.checkpoint_from_domains(checkpoint.position(), &expected)?;
    compare_domains(&expected, &rebuilt.domain_values())?;
    let mut single = model.session().await?;
    let mut native = model.session().await?;
    single.import_checkpoint(checkpoint).await?;
    native.import_checkpoint(&rebuilt).await?;
    compare_domains(&expected, &single.snapshot().read_domains().await?)?;
    compare_domains(&expected, &native.snapshot().read_domains().await?)?;
    assert_eq!(single.binding().frontier(), checkpoint.position() as u64);
    assert_eq!(native.binding().frontier(), checkpoint.position() as u64);
    assert_ne!(single.binding().session(), native.binding().session());
    Ok(Lane {
        cpu,
        single,
        native,
    })
}

fn roots(lanes: &[Lane]) -> Vec<RootBinding> {
    lanes.iter().map(|lane| lane.native.binding()).collect()
}

async fn same_state(lanes: &[Lane]) -> Result<()> {
    for lane in lanes {
        let expected = lane.cpu.snapshot().domain_values();
        assert_eq!(expected.len(), 42);
        compare_domains(&expected, &lane.single.snapshot().read_domains().await?)?;
        // The original reader also rejects every noncanonical future KV word.
        compare_domains(&expected, &lane.native.snapshot().read_domains().await?)?;
    }
    Ok(())
}

async fn reference(lanes: &mut [Lane], tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    assert_eq!(lanes.len(), tokens.len());
    let mut expected = Vec::with_capacity(lanes.len());
    for (lane, &token) in lanes.iter_mut().zip(tokens) {
        let logits = lane.cpu.forward(token)?;
        let single = lane.single.forward(&[token], false).await?;
        assert!(single.dispatches > 0);
        selected(&logits);
        exact(
            &logits,
            &single.logits,
            "ragged independent full248320 CPU/B1",
        )?;
        let domains = lane.cpu.snapshot().domain_values();
        assert_eq!(domains.len(), 42);
        compare_domains(&domains, &lane.single.snapshot().read_domains().await?)?;
        expected.push(logits);
    }
    Ok(expected)
}

fn receipt(candidate: &PreparedResidentCohort, before: &[RootBinding], greedy: bool) {
    assert_eq!(candidate.len(), before.len());
    assert_eq!(candidate.prior_frontier(), None);
    assert_eq!(
        candidate.prior_frontiers().collect::<Vec<_>>(),
        before
            .iter()
            .map(|binding| binding.frontier())
            .collect::<Vec<_>>()
    );
    assert!(candidate.fused_dispatches() > 0);
    assert!(candidate.fused_dispatches() < candidate.dispatches());
    assert_eq!(
        candidate.native_greedy_dispatches(),
        if greedy { 4 } else { 0 }
    );
}

fn published(lanes: &[Lane], before: &[RootBinding]) {
    assert_eq!(lanes.len(), before.len());
    for (lane, prior) in lanes.iter().zip(before) {
        let after = lane.native.binding();
        assert_eq!(after.source(), prior.source());
        assert_eq!(after.session(), prior.session());
        assert_eq!(after.session_epoch(), prior.session_epoch());
        assert_eq!(after.root(), prior.root() + 1);
        assert_eq!(after.frontier(), prior.frontier() + 1);
    }
}

fn packet(read: &mut InferenceReadback, lane: &Lane, expected: &[f32]) -> ResidentToken {
    assert!(read.logits.is_empty());
    let packet = read
        .device_token
        .take()
        .expect("original private GPU packet");
    assert_eq!((packet.id, packet.max_bits), selected(expected));
    assert_eq!(packet.producer, Some(lane.native.binding()));
    assert_eq!(packet.lease.bytes(), 12);
    assert!(packet.lease.is_initialized());
    assert!(Arc::ptr_eq(
        packet.lease.service(),
        &lane.native.model.service
    ));
    packet
}

async fn feedback(
    model: &Arc<ResidentModel>,
    lanes: &mut [Lane],
    packets: &[ResidentToken],
    abandon_once: bool,
) -> Result<Vec<ResidentToken>> {
    let before = roots(lanes);
    let uploads = model.service.uploaded_value_bytes();
    let queue = model.service.queue().stats()?;
    let mut swapped = packets.to_vec();
    swapped.swap(0, 1);
    let invalid = lanes
        .iter()
        .zip(&swapped)
        .map(|(lane, token)| (&lane.native, token))
        .collect::<Vec<_>>();
    assert!(
        model
            .prepare_device_tokens_fused_cohort_admitted(&invalid)
            .await
            .is_err()
    );
    assert_eq!(model.service.queue().stats()?, queue);
    assert_eq!(roots(lanes), before);
    drop(invalid);
    drop(swapped);
    if abandon_once {
        let rows = lanes
            .iter()
            .zip(packets)
            .map(|(lane, token)| (&lane.native, token))
            .collect::<Vec<_>>();
        let completed = model.prepare_device_tokens_fused_cohort(&rows).await?;
        receipt(&completed, &before, true);
        assert_eq!(
            roots(lanes),
            before,
            "Ready is private before atomic publication"
        );
        drop(completed);
        same_state(lanes).await?;
        assert_eq!(model.service.pending_events()?, 0);
    }
    let ids = packets.iter().map(|token| token.id).collect::<Vec<_>>();
    let expected = reference(lanes, &ids).await?;
    let rows = lanes
        .iter()
        .zip(packets)
        .map(|(lane, token)| (&lane.native, token))
        .collect::<Vec<_>>();
    let completed = model.prepare_device_tokens_fused_cohort(&rows).await?;
    receipt(&completed, &before, true);
    assert_eq!(roots(lanes), before);
    let mut reads = completed.commit(
        &mut lanes
            .iter_mut()
            .map(|lane| &mut lane.native)
            .collect::<Vec<_>>(),
    )?;
    published(lanes, &before);
    same_state(lanes).await?;
    assert_eq!(model.service.uploaded_value_bytes(), uploads);
    assert_eq!(model.service.pending_events()?, 0);
    let packets = reads
        .iter_mut()
        .zip(lanes.iter())
        .zip(&expected)
        .map(|((read, lane), logits)| packet(read, lane, logits))
        .collect::<Vec<_>>();
    assert_eq!(
        packets
            .iter()
            .map(|token| token.lease.id())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        lanes.len()
    );
    Ok(packets)
}

/// A poison input makes accidental arithmetic on a finished row observable.
/// Inputs/scratch/readback use the original physical owner and fair queue.
async fn inactive_av_bits(model: &ResidentModel) -> Result<()> {
    use cubecl::prelude::{BufferArg, CubeCount, CubeDim};
    fn arg(handle: &Handle, elements: usize) -> BufferArg {
        // Each exact allocation is retained by the original submission guard.
        unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
    }
    let service = &model.service;
    let bytes = service.resident_bytes()?;
    let mut input = vec![1.0f32.to_bits(); 2048];
    input[..512].fill(f32::NAN.to_bits());
    let previous = (0..64)
        .map(|at| {
            if at < 32 {
                [0x80000000u32, 1, 0x80000001, 0xbf800000][at % 4]
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let previous_bytes = previous
        .iter()
        .flat_map(|bits| bits.to_ne_bytes())
        .collect::<Vec<_>>();
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: 16 * 1024,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut initialization = service.begin_initialization(0)?;
    let input = initialization.upload_bytes(
        &input
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>(),
        BufferRole::Scratch,
    )?;
    let previous_gpu = initialization.upload_bytes(&previous_bytes, BufferRole::Scratch)?;
    let positions = initialization.upload_bytes(
        &[255u32, 256]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>(),
        BufferRole::Scratch,
    )?;
    let lanes = initialization.allocate_empty(64 * 4, BufferRole::Scratch)?;
    let sums = initialization.upload_bytes(
        &[1.0f64; 64]
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>(),
        BufferRole::Scratch,
    )?;
    let subtracted = initialization.upload_bytes(
        &[2.0f64; 64]
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>(),
        BufferRole::Scratch,
    )?;
    let ready = initialization.complete_sync().await?;
    slot.finish()?;
    drop(ready);
    let accesses = [
        (input.clone(), DeviceAccess::Read),
        (previous_gpu.clone(), DeviceAccess::Read),
        (positions.clone(), DeviceAccess::Read),
        (lanes.clone(), DeviceAccess::Write),
        (sums.clone(), DeviceAccess::Write),
        (subtracted.clone(), DeviceAccess::Write),
    ];
    let outputs = [lanes.clone(), sums.clone(), subtracted.clone()];
    let readback = outputs
        .iter()
        .map(|lease| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: 64 * 20,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let flight = service.submit(0, &accesses)?;
    for stage in 0..3 {
        unsafe {
            crate::attention_value::accumulate_rows::launch_unchecked(
                service.client(),
                CubeCount::Static(2, 1, 1),
                CubeDim::new_1d(32),
                arg(&input.handle(), 2048),
                arg(&previous_gpu.handle(), 64),
                arg(&lanes.handle(), 64),
                arg(&sums.handle(), 64),
                arg(&subtracted.handle(), 64),
                arg(&positions.handle(), 2),
                8,
                1,
                512,
                2,
                2,
                2,
                256,
                stage,
            );
        }
    }
    let (ready, _) = flight
        .complete_with_readback(&outputs, &readback, |data| {
            if data.len() != 3
                || data[0].len() != 64 * 4
                || data[1].len() != 64 * 8
                || data[2].len() != 64 * 8
            {
                return Err(Error::State(
                    "ragged AV bit-transport readback coverage".into(),
                ));
            }
            for at in 0..64 {
                let got = u32::from_ne_bytes(data[0][at * 4..at * 4 + 4].try_into().unwrap());
                let expected = if at < 32 {
                    previous[at]
                } else {
                    1.0f32.to_bits()
                };
                if got != expected {
                    return Err(Error::State(format!(
                        "ragged AV inactive/active lane {at} {got:08x} != {expected:08x}"
                    )));
                }
                if at < 32 {
                    let sum = u64::from_ne_bytes(data[1][at * 8..at * 8 + 8].try_into().unwrap());
                    let sub = u64::from_ne_bytes(data[2][at * 8..at * 8 + 8].try_into().unwrap());
                    if sum != 1.0f64.to_bits() || sub != 2.0f64.to_bits() {
                        return Err(Error::State(
                            "ragged AV finished row executed floating arithmetic".into(),
                        ));
                    }
                }
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop(ready);
    drop(readback);
    drop(outputs);
    drop(accesses);
    drop((input, previous_gpu, positions, lanes, sums, subtracted));
    assert_eq!(service.resident_bytes()?, bytes);
    assert_eq!(service.pending_events()?, 0);
    println!(
        "P12_RAGGED_AV_BITS_PASS valid_tokens=256,257 stage=8 finished_row_signedzero_subnormal_bits=exact poison_input=not_evaluated active_original_FMA=exact dispatches=3"
    );
    Ok(())
}

async fn qualify(name: &str) -> Result<()> {
    if cfg!(debug_assertions) {
        panic!("physical unequal-frontier qualification requires --release");
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let source = LoadedQwenModel::open(
        &directory.join("model.gguf"),
        &directory.join("config.json"),
    )?;
    assert_eq!(source.geometry().vocab_size, 248_320);
    let mut checkpoints = Vec::with_capacity(8);
    for (row, position) in POSITIONS.into_iter().enumerate() {
        let mut cpu = QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let stream = (0..position)
            .map(|at| SEEDS[(at + row * 3) % SEEDS.len()])
            .collect::<Vec<_>>();
        if !stream.is_empty() {
            cpu.prefill_chunk_without_logits(&stream)?;
        }
        checkpoints.push(cpu.checkpoint()?);
    }
    let mut fresh = QwenDenseCpu::from_shared(Arc::clone(&source))?;
    fresh.prefill_chunk_without_logits(&SEEDS[..3])?;
    let refill = fresh.checkpoint()?;
    let empty = QwenDenseCpu::from_shared(Arc::clone(&source))?.checkpoint()?;
    let model = load_qualified(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        Arc::clone(&source),
        512,
        PROVIDER_BYTES,
        DEVICE_BYTES,
    )
    .await?;
    inactive_av_bits(&model).await?;
    let baseline = model.service.resident_bytes()?;
    let key = model.service.physical_key();
    for width in [2, 4, 8] {
        let mut lanes = Vec::with_capacity(width);
        for checkpoint in &checkpoints[..width] {
            lanes.push(hydrated(&model, &source, checkpoint).await?);
        }
        let before = roots(&lanes);
        let ids = SEEDS[..width].to_vec();
        let expected = reference(&mut lanes, &ids).await?;
        let uploads = model.service.uploaded_value_bytes();
        let rows = lanes
            .iter()
            .zip(&ids)
            .map(|(lane, &id)| (&lane.native, id))
            .collect::<Vec<_>>();
        let completed = model.prepare_logits_fused_cohort(&rows).await?;
        receipt(&completed, &before, false);
        assert_eq!(roots(&lanes), before);
        let reads = completed.commit(
            &mut lanes
                .iter_mut()
                .map(|lane| &mut lane.native)
                .collect::<Vec<_>>(),
        )?;
        for (read, logits) in reads.iter().zip(&expected) {
            exact(logits, &read.logits, "ragged native full248320 logits")?;
        }
        published(&lanes, &before);
        same_state(&lanes).await?;
        assert_eq!(model.service.uploaded_value_bytes(), uploads);
        if width == 8 {
            let exhausted_roots = roots(&lanes);
            let queue = model.service.queue().stats()?;
            let exhausted = lanes
                .iter()
                .zip(SEEDS)
                .map(|(lane, id)| (&lane.native, id))
                .collect::<Vec<_>>();
            assert!(
                model
                    .prepare_greedy_fused_cohort_admitted(&exhausted)
                    .await
                    .is_err()
            );
            assert_eq!(model.service.queue().stats()?, queue);
            assert_eq!(roots(&lanes), exhausted_roots);
            // Retire the near-capacity row too before repeated native feedback.
            lanes[6] = hydrated(&model, &source, &refill).await?;
        }
        let retired = lanes[width - 1].native.binding();
        lanes[width - 1] = hydrated(&model, &source, &refill).await?;
        assert_ne!(
            lanes[width - 1].native.binding().session(),
            retired.session()
        );
        let expected = reference(&mut lanes, &ids).await?;
        let mut packets = Vec::with_capacity(width);
        for ((lane, &id), logits) in lanes.iter_mut().zip(&ids).zip(&expected) {
            let mut read = lane.native.forward_greedy(&[id]).await?;
            packets.push(packet(&mut read, lane, logits));
        }
        same_state(&lanes).await?;
        packets = feedback(&model, &mut lanes, &packets, true).await?;
        // A native-produced packet returns to the original B1 engine, while
        // siblings keep their native-produced packets and independent roots.
        let fallback_id = packets[0].id;
        let fallback_expected = reference(&mut lanes[..1], &[fallback_id]).await?;
        let stale = packets[0].clone();
        let mut read = lanes[0].native.forward_device_token(&packets[0]).await?;
        packets[0] = packet(&mut read, &lanes[0], &fallback_expected[0]);
        let before = roots(&lanes);
        let queue = model.service.queue().stats()?;
        let stale_rows = lanes
            .iter()
            .enumerate()
            .map(|(row, lane)| (&lane.native, if row == 0 { &stale } else { &packets[row] }))
            .collect::<Vec<_>>();
        assert!(
            model
                .prepare_device_tokens_fused_cohort_admitted(&stale_rows)
                .await
                .is_err()
        );
        assert_eq!(model.service.queue().stats()?, queue);
        assert_eq!(roots(&lanes), before);
        drop(stale_rows);
        drop(stale);
        let retired = lanes[width - 1].native.binding();
        lanes[width - 1] = hydrated(&model, &source, &empty).await?;
        assert_ne!(
            lanes[width - 1].native.binding().session(),
            retired.session()
        );
        let refill_expected = reference(&mut lanes[width - 1..], &[SEEDS[width - 1]]).await?;
        let mut read = lanes[width - 1]
            .native
            .forward_greedy(&[SEEDS[width - 1]])
            .await?;
        packets[width - 1] = packet(&mut read, &lanes[width - 1], &refill_expected[0]);
        same_state(&lanes).await?;
        let next = feedback(&model, &mut lanes, &packets, false).await?;
        drop(next);
        drop(packets);
        drop(lanes);
        let workspace_bytes = model
            .workspace
            .lock()
            .await
            .batch
            .as_ref()
            .expect("native workspace")
            .program
            .budget()
            .owned_bytes()?;
        assert_eq!(model.service.resident_bytes()?, baseline + workspace_bytes);
        assert_eq!(model.service.pending_events()?, 0);
        println!(
            "P12_RAGGED_SOURCE_PASS source={name} B={width} initial_positions={:?} full248320_CPU_B1_native=bit_exact all42=bit_exact valid256_257=true independent_frontiers=true B1_to_native=true native_to_B1=true refill=true Ready_abandon=true foreign_stale_packet_prelaunch=refused warm_uploads=0",
            &POSITIONS[..width]
        );
    }
    drop(model);
    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
    assert_eq!(budget.reserved, 0);
    assert!(!budget.fenced);
    println!("P12_RAGGED_RECLAIM_PASS source={name} final_reserved_bytes=0");
    Ok(())
}

#[test]
#[ignore = "release physical CUDA; approved original .8B source, CPU prefixes and all42/fullV references"]
fn original_qwen35_08b_native_unequal_frontiers() -> Result<()> {
    futures_lite::future::block_on(qualify("qwen35-0.8b"))
}

#[test]
#[ignore = "release physical CUDA; approved original 2B source, CPU prefixes and all42/fullV references"]
fn original_qwen35_2b_native_unequal_frontiers() -> Result<()> {
    futures_lite::future::block_on(qualify("qwen35-2b"))
}
