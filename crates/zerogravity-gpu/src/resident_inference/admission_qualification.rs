//! Real lease pressure: prelaunch refusal, smaller native work and reclamation.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};
use zerogravity_model_qwen35::{QwenDenseCpu, tokenize::QwenTokenizer};

const PROVIDER_BYTES: usize = 4 * 1024 * 1024 * 1024;
const DEVICE_BYTES: u64 = 5 * 1024 * 1024 * 1024;
const SEEDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];
type Domains = Vec<(StateSlot, Vec<f32>)>;

fn selected(logits: &[f32]) -> (u32, u32) {
    assert_eq!(logits.len(), 248_320);
    assert!(logits.iter().all(|value| value.is_finite()));
    let id = (0..logits.len())
        .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
        .expect("complete original vocabulary");
    (id as u32, logits[id].to_bits())
}

fn same_domains(expected: &Domains, actual: &Domains) -> Result<()> {
    assert_eq!(expected.len(), 42);
    assert_eq!(actual.len(), 42);
    let expected = expected
        .iter()
        .map(|(slot, values)| (*slot, values))
        .collect::<BTreeMap<_, _>>();
    let actual = actual
        .iter()
        .map(|(slot, values)| (*slot, values))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(expected.len(), 42, "duplicate CPU/prior state slot");
    assert_eq!(actual.len(), 42, "duplicate physical state slot");
    for (slot, expected) in expected {
        exact(
            expected,
            actual.get(&slot).expect("missing state slot"),
            "admission all42",
        )?;
    }
    Ok(())
}

async fn domains(sessions: &[ResidentSession]) -> Result<Vec<Domains>> {
    let mut output = Vec::with_capacity(sessions.len());
    for session in sessions {
        // The accepted reader also rejects every noncanonical future KV word.
        output.push(session.snapshot().read_domains().await?);
    }
    Ok(output)
}

fn packet(
    read: &mut InferenceReadback,
    session: &ResidentSession,
    expected: (u32, u32),
) -> ResidentToken {
    let token = read
        .device_token
        .take()
        .expect("completed original GPU packet");
    assert!(read.logits.is_empty());
    assert_eq!((token.id, token.max_bits), expected);
    assert_eq!(token.producer, Some(session.binding()));
    assert_eq!(token.lease.bytes(), 12);
    assert!(token.lease.is_initialized());
    assert!(Arc::ptr_eq(token.lease.service(), &session.model.service));
    token
}

#[derive(Debug, PartialEq, Eq)]
struct WorkspaceIdentity {
    rows: Option<usize>,
    leases: Vec<String>,
    handles: Vec<String>,
}

async fn workspace_identity(model: &ResidentModel) -> WorkspaceIdentity {
    let workspace = model.workspace.lock().await;
    let Some(batch) = &workspace.batch else {
        return WorkspaceIdentity {
            rows: None,
            leases: Vec::new(),
            handles: Vec::new(),
        };
    };
    // Keep identities only, never unleased raw handles across a band change.
    WorkspaceIdentity {
        rows: Some(batch.rows),
        leases: batch
            .leases
            .iter()
            .map(|(lease, _)| {
                format!(
                    "{:?}:{}:{}",
                    lease.id(),
                    lease.allocation_generation(),
                    lease.bytes()
                )
            })
            .collect(),
        handles: batch
            .program
            .owned_handles()
            .into_iter()
            .map(|handle| {
                let binding = handle.binding();
                format!(
                    "{:?}:{:?}:{:?}",
                    binding.service,
                    binding.memory.id(),
                    binding.range()
                )
            })
            .collect(),
    }
}

async fn refused_without_effects(
    model: &Arc<ResidentModel>,
    sessions: &[ResidentSession],
    packets: &[ResidentToken],
    idle_rows: Option<usize>,
) -> Result<()> {
    let prior_domains = domains(sessions).await?;
    let roots = sessions
        .iter()
        .map(ResidentSession::binding)
        .collect::<Vec<_>>();
    let workspace = workspace_identity(model).await;
    assert_eq!(workspace.rows, idle_rows);
    let canary = model
        .service
        .allocate_uninitialized(4, BufferRole::Scratch)?;
    let next_generation = canary.allocation_generation().checked_add(1).unwrap();
    drop(canary);
    let bytes = model.service.resident_bytes()?;
    let uploads = model.service.uploaded_value_bytes();
    let queue = model.service.queue().stats()?;
    assert_eq!(queue.queued, 0);
    assert!(!queue.active);
    assert_eq!(model.service.pending_events()?, 0);

    let host_rows = sessions.iter().zip(SEEDS).collect::<Vec<_>>();
    assert!(
        model
            .prepare_greedy_fused_cohort_admitted(&host_rows)
            .await?
            .is_none()
    );
    let device_rows = sessions.iter().zip(packets).collect::<Vec<_>>();
    assert!(
        model
            .prepare_device_tokens_fused_cohort_admitted(&device_rows)
            .await?
            .is_none()
    );
    assert!(matches!(
        model.prepare_greedy_fused_cohort(&host_rows).await,
        Err(Error::Limit(_))
    ));

    let mut invalid = host_rows;
    invalid.last_mut().unwrap().1 = u32::MAX;
    assert!(
        model
            .prepare_greedy_fused_cohort_admitted(&invalid)
            .await
            .is_err()
    );
    let mut swapped = packets.to_vec();
    swapped.swap(0, 1);
    let foreign_rows = sessions.iter().zip(&swapped).collect::<Vec<_>>();
    assert!(
        model
            .prepare_device_tokens_fused_cohort_admitted(&foreign_rows)
            .await
            .is_err()
    );
    drop(foreign_rows);
    drop(swapped);

    assert_eq!(
        sessions
            .iter()
            .map(ResidentSession::binding)
            .collect::<Vec<_>>(),
        roots
    );
    assert_eq!(
        model.service.resident_bytes()?,
        bytes,
        "refusal allocated candidate/packet"
    );
    assert_eq!(
        model.service.uploaded_value_bytes(),
        uploads,
        "refusal initialized a GPU program"
    );
    assert_eq!(
        model.service.queue().stats()?,
        queue,
        "refusal entered physical queue"
    );
    assert_eq!(model.service.pending_events()?, 0);
    assert_eq!(
        workspace_identity(model).await,
        workspace,
        "refusal retired/replaced idle workspace"
    );
    model.service.ensure_usable()?;
    let canary = model
        .service
        .allocate_uninitialized(4, BufferRole::Scratch)?;
    assert_eq!(
        canary.allocation_generation(),
        next_generation,
        "hidden allocations during refusal"
    );
    drop(canary);
    for (session, expected) in sessions.iter().zip(&prior_domains) {
        same_domains(expected, &session.snapshot().read_domains().await?)?;
    }
    assert_eq!(model.service.resident_bytes()?, bytes);
    println!(
        "P12_NATIVE_ADMISSION_REFUSAL_PASS idle_rows={idle_rows:?} B8=None invalid_host_foreign_producer=Err legacy=Limit allocation_queue_upload_root_all42=unchanged unfenced=true"
    );
    Ok(())
}

struct References {
    cpu: Vec<QwenDenseCpu>,
    b1: Vec<ResidentSession>,
    selected: Vec<(u32, u32)>,
}

async fn advance(
    model: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    packets: &mut [ResidentToken],
    reference: &mut References,
    start: usize,
    count: usize,
) -> Result<()> {
    let original = sessions
        .iter()
        .map(ResidentSession::binding)
        .collect::<Vec<_>>();
    for row in start..start + count {
        let input = reference.selected[row].0;
        let logits = reference.cpu[row].forward(input)?;
        let single = reference.b1[row].forward(&[input], false).await?;
        exact(
            &logits,
            &single.logits,
            "independent CPU/B1 full vocabulary under admission pressure",
        )?;
        reference.selected[row] = selected(&logits);
    }
    let candidate = {
        let inputs = sessions[start..start + count]
            .iter()
            .zip(&packets[start..start + count])
            .collect::<Vec<_>>();
        model
            .prepare_device_tokens_fused_cohort_admitted(&inputs)
            .await?
            .expect("qualified native band must fit actual remaining bytes")
    };
    assert_eq!(candidate.len(), count);
    assert_ne!(candidate.fused_dispatches(), 0);
    assert_eq!(candidate.native_greedy_dispatches(), 4);
    assert_eq!(
        sessions
            .iter()
            .map(ResidentSession::binding)
            .collect::<Vec<_>>(),
        original
    );
    let dispatches = candidate.dispatches();
    let mut live = sessions[start..start + count]
        .iter_mut()
        .collect::<Vec<_>>();
    let reads = candidate.commit(&mut live)?;
    drop(live);
    assert_eq!(reads.len(), count);
    assert_eq!(
        reads.iter().map(|read| read.dispatches).sum::<usize>(),
        dispatches
    );
    assert_eq!(
        reads.iter().map(|read| read.bytes).sum::<usize>(),
        4 + 12 * count
    );
    for (offset, mut read) in reads.into_iter().enumerate() {
        let row = start + offset;
        packets[row] = packet(&mut read, &sessions[row], reference.selected[row]);
    }
    let mut unique = BTreeSet::new();
    for (row, session) in sessions.iter().enumerate() {
        let after = session.binding();
        if (start..start + count).contains(&row) {
            let before = original[row];
            assert_eq!(after.source(), before.source());
            assert_eq!(after.session(), before.session());
            assert_eq!(after.session_epoch(), before.session_epoch());
            assert_eq!(after.root(), before.root() + 1);
            assert_eq!(after.frontier(), before.frontier() + 1);
        } else {
            assert_eq!(after, original[row]);
        }
        assert!(
            unique.insert(packets[row].lease.id()),
            "shared output packet lease"
        );
        let expected = reference.cpu[row].snapshot().domain_values();
        same_domains(&expected, &session.snapshot().read_domains().await?)?;
        same_domains(
            &expected,
            &reference.b1[row].snapshot().read_domains().await?,
        )?;
    }
    assert_eq!(model.service.pending_events()?, 0);
    model.service.ensure_usable()?;
    println!(
        "P12_NATIVE_ADMISSION_PROGRESS_PASS B={count} row_start={start} actual_dispatches={dispatches} CPU_B1_full_logits_packets_all42=exact independent_roots=true"
    );
    Ok(())
}

async fn qualify(name: &str) -> Result<()> {
    if cfg!(debug_assertions) {
        panic!("physical admission qualification requires --release");
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let source = LoadedQwenModel::open(
        &directory.join("model.gguf"),
        &directory.join("config.json"),
    )?;
    let tokenizer = QwenTokenizer::from_file(&directory.join("tokenizer.json"))?;
    assert_eq!(source.geometry().vocab_size, 248_320);
    assert!(u64::from(tokenizer.eos()) < source.geometry().vocab_size);
    let device = Device::Cuda(cubecl::device::CudaDevice::new(0));
    let client = device.client();
    let properties = client.properties();
    let pci = properties
        .identity
        .physical
        .as_ref()
        .and_then(|identity| identity.pci_address)
        .expect("qualified CUDA physical PCI identity");
    let key = PhysicalDeviceKey::new("pci", &pci.to_string())?;
    let arbiter = ResourceArbiter::process_shared();
    if let Ok(prior) = arbiter.snapshot(&key) {
        assert_eq!(prior.reserved, 0, "physical tests must run serially");
        assert!(!prior.fenced);
    }
    let model = load_qualified(
        device,
        Arc::clone(&source),
        512,
        PROVIDER_BYTES,
        DEVICE_BYTES,
    )
    .await?;
    let mut sessions = Vec::with_capacity(8);
    let mut packets = Vec::with_capacity(8);
    let mut reference = References {
        cpu: Vec::with_capacity(8),
        b1: Vec::with_capacity(8),
        selected: Vec::with_capacity(8),
    };
    for input in SEEDS {
        let mut cpu = QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let logits = cpu.prefill_transaction(&[input])?;
        let expected = selected(&logits);
        let mut single = model.session().await?;
        exact(
            &logits,
            &single.forward(&[input], false).await?.logits,
            "admission independent initial CPU/B1 logits",
        )?;
        let mut session = model.session().await?;
        let mut read = session.forward_greedy(&[input]).await?;
        packets.push(packet(&mut read, &session, expected));
        let state = cpu.snapshot().domain_values();
        same_domains(&state, &single.snapshot().read_domains().await?)?;
        same_domains(&state, &session.snapshot().read_domains().await?)?;
        sessions.push(session);
        reference.cpu.push(cpu);
        reference.b1.push(single);
        reference.selected.push(expected);
    }
    let (need4, need8) = {
        let workspace = model.workspace.lock().await;
        assert!(workspace.batch.is_none());
        let required = |rows: usize| -> Result<usize> {
            let owned = BatchProgram::preflight(&workspace.program, rows)?.owned_bytes()?;
            model
                .state_bytes
                .checked_mul(rows)
                .and_then(|bytes| bytes.checked_add(rows * 12))
                .and_then(|bytes| bytes.checked_add(owned))
                .ok_or_else(|| Error::Limit("admission qualification checked byte sum".into()))
        };
        (required(4)?, required(8)?)
    };
    assert!(need8 > need4);
    let available = model.service.remaining_capacity()?;
    assert!(
        need8 <= available,
        "B8 must fit again after real pressure drops"
    );
    let pressure_bytes = available
        .checked_sub(need4)
        .filter(|bytes| *bytes > 0)
        .expect("real BufferLease pressure extent");
    let pressure = model
        .service
        .allocate_uninitialized(pressure_bytes, BufferRole::Scratch)?;
    assert_eq!(pressure.bytes(), pressure_bytes);
    assert!(!pressure.is_initialized());
    assert_eq!(model.service.remaining_capacity()?, need4);
    assert!(model.service.resident_bytes()? <= PROVIDER_BYTES);
    println!(
        "P12_NATIVE_ADMISSION_PRESSURE source={name} actual_scratch_lease_bytes={pressure_bytes} cold_required_B4={need4} cold_required_B8={need8} provider_cap={PROVIDER_BYTES} device_cap={DEVICE_BYTES}"
    );
    refused_without_effects(&model, &sessions, &packets, None).await?;
    advance(&model, &mut sessions, &mut packets, &mut reference, 0, 4).await?;
    advance(&model, &mut sessions, &mut packets, &mut reference, 4, 4).await?;
    refused_without_effects(&model, &sessions, &packets, Some(4)).await?;
    drop(pressure);
    advance(&model, &mut sessions, &mut packets, &mut reference, 0, 8).await?;
    assert_eq!(workspace_identity(&model).await.rows, Some(8));
    drop(packets);
    drop(sessions);
    drop(reference);
    let service = Arc::clone(&model.service);
    assert_eq!(service.pending_events()?, 0);
    service.ensure_usable()?;
    drop(model);
    assert_eq!(
        service.resident_bytes()?,
        0,
        "original GPU lease reclamation"
    );
    assert_eq!(arbiter.snapshot(&key)?.reserved, PROVIDER_BYTES as u64);
    drop(service);
    let final_state = arbiter.snapshot(&key)?;
    assert_eq!(final_state.reserved, 0);
    assert!(!final_state.fenced);
    println!(
        "P12_NATIVE_ADMISSION_PHYSICAL_PASS source={name} actual_pressure=true cold_warm_refusal=prelaunch B4_under_pressure_B8_after_release=CPU_B1_exact all42=true idle_handles_preserved=true final_reserved=0 unfenced=true"
    );
    Ok(())
}

#[test]
#[ignore = "release physical CUDA; approved original .8B; run serially"]
fn original_qwen35_08b_native_prelaunch_admission_pressure() -> Result<()> {
    futures_lite::future::block_on(qualify("qwen35-0.8b"))
}

#[test]
#[ignore = "release physical CUDA; approved original 2B; run serially"]
fn original_qwen35_2b_native_prelaunch_admission_pressure() -> Result<()> {
    futures_lite::future::block_on(qualify("qwen35-2b"))
}
