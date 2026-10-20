//! NEXT source-only qualification: typed StateOnly/Logits bodies, B2/B4/B8.
//! ROOT must first pass the frozen original B2/logits prototype physically.
use super::qualification::{idle, loaded, roots, states, unchanged};
use super::*;
use zerogravity_model_qwen35::{QwenDenseCpu, tokenize::QwenTokenizer};

const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

fn selected(logits: &[f32]) -> Result<(u32, u32)> {
    if logits.len() != 248_320 || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::State("wide graph CPU fullV selection".into()));
    }
    let mut best = 0;
    for (at, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = at;
        }
    }
    Ok((best as u32, logits[best].to_bits()))
}
#[cfg(feature = "hardware-abort-probe")]
fn body(readout: Readout) -> ResidentOutput {
    match readout {
        Readout::None => ResidentOutput::StateOnly,
        Readout::Logits | Readout::Greedy => ResidentOutput::Logits,
    }
}
#[cfg(feature = "hardware-abort-probe")]
fn mode_name(readout: Readout) -> &'static str {
    match readout {
        Readout::None => "state",
        Readout::Logits => "logits",
        Readout::Greedy => "greedy",
    }
}
#[cfg(feature = "hardware-abort-probe")]
fn parse_mode(mode: &str) -> Result<Readout> {
    match mode {
        "state" => Ok(Readout::None),
        "logits" => Ok(Readout::Logits),
        "greedy" => Ok(Readout::Greedy),
        _ => Err(Error::Malformed(
            "wide graph mode must be state/logits/greedy".into(),
        )),
    }
}
fn rows<'a>(
    sessions: &'a [ResidentSession],
    tokens: &[u32],
    order: &[usize],
) -> Result<Vec<(&'a ResidentSession, u32)>> {
    if sessions.len() != tokens.len() || order.len() != tokens.len() {
        return Err(Error::State("wide graph exact row coverage".into()));
    }
    let mut seen = std::collections::BTreeSet::new();
    order
        .iter()
        .map(|&row| {
            if row >= sessions.len() || !seen.insert(row) {
                return Err(Error::State("wide graph row permutation".into()));
            }
            Ok((&sessions[row], tokens[row]))
        })
        .collect()
}
fn ordered_sessions<'a>(
    sessions: &'a mut [ResidentSession],
    order: &[usize],
) -> Result<Vec<&'a mut ResidentSession>> {
    let mut remaining = sessions.iter_mut().map(Some).collect::<Vec<_>>();
    order
        .iter()
        .map(|&row| {
            remaining
                .get_mut(row)
                .and_then(Option::take)
                .ok_or_else(|| Error::State("wide graph mutable row permutation".into()))
        })
        .collect()
}
fn packet(read: &InferenceReadback, session: &ResidentSession, expected: (u32, u32)) -> Result<()> {
    let token = read
        .device_token
        .as_ref()
        .ok_or_else(|| Error::State("wide graph missing real GPU packet".into()))?;
    if token.producer != Some(session.binding())
        || (token.id, token.max_bits) != expected
        || token.lease.bytes() != 12
        || !token.lease.is_initialized()
        || !Arc::ptr_eq(token.lease.service(), &session.model.service)
    {
        return Err(Error::State(
            "wide graph packet/source/root/physical owner".into(),
        ));
    }
    Ok(())
}
fn candidate_shape(
    candidate: &PreparedResidentCohort,
    count: usize,
    readout: Readout,
) -> Result<()> {
    if candidate.len() != count
        || candidate.dispatches() == 0
        || candidate.fused_dispatches() == 0
        || candidate.native_greedy_dispatches()
            != if matches!(readout, Readout::Greedy) {
                4
            } else {
                0
            }
    {
        return Err(Error::State(
            "wide graph real cohort/body/sampler receipt".into(),
        ));
    }
    for (row, prepared) in candidate.prepared.iter().enumerate() {
        let read = &prepared.readback;
        let row_bytes = match readout {
            Readout::None => 0,
            Readout::Logits => 248_320 * 4,
            Readout::Greedy => 12,
        };
        if read.bytes != row_bytes + usize::from(row == 0) * 4
            || read.chunk_bytes != read.bytes
            || read.state_only_steps != usize::from(matches!(readout, Readout::None))
            || read.logit_steps != usize::from(!matches!(readout, Readout::None))
            || read.logits.len()
                != if matches!(readout, Readout::Logits) {
                    248_320
                } else {
                    0
                }
            || read.device_token.is_some() != matches!(readout, Readout::Greedy)
        {
            return Err(Error::State(
                "wide graph state-only/logit/packet readback geometry".into(),
            ));
        }
    }
    Ok(())
}

/// Explicit qualifier read of the already Ready, source-owned full logits.
/// Greedy API still returns only12B packets; diagnostic transfer is separate.
async fn completed_logits(model: &Arc<ResidentModel>, root: u64, count: usize) -> Result<Vec<f32>> {
    let bytes = count
        .checked_mul(248_320 * 4)
        .ok_or_else(|| Error::Limit("wide graph diagnostic logits".into()))?;
    let ticket = model.service.queue().submit(
        model.service.budget(),
        model.service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: (bytes + 4) as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    let workspace = model.workspace.lock().await;
    let batch = workspace
        .batch
        .as_ref()
        .ok_or_else(|| Error::State("wide graph diagnostic batch absent".into()))?;
    if batch.rows != count {
        return Err(Error::State("wide graph diagnostic row band".into()));
    }
    let logits = batch
        .program
        .logits()
        .ok_or_else(|| Error::State("wide graph diagnostic planned logits absent".into()))?;
    if logits.elements != bytes / 4 {
        return Err(Error::State("wide graph diagnostic fullV geometry".into()));
    }
    let status = lease_for(&batch.leases, &batch.program.status())?.clone();
    let logits = lease_for(&batch.leases, &logits.handle)?.clone();
    let accesses = [
        (status.clone(), DeviceAccess::Read),
        (logits.clone(), DeviceAccess::Read),
    ];
    slot.begin_submission()?;
    let flight = model.service.submit(root, &accesses)?;
    let read = flight
        .complete_read_only_readback(
            &[status.readback(0, 4)?, logits.readback(0, bytes)?],
            |values| {
                if values.len() != 2
                    || !all_zero(&values[0])
                    || words(&values[1])?.iter().any(|x| !x.is_finite())
                {
                    return Err(Error::State(
                        "wide graph diagnostic fullV status/finite".into(),
                    ));
                }
                Ok(())
            },
        )
        .await?;
    slot.finish()?;
    words(&read[1])
}

async fn pressure_and_abandon(
    model: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    graph: &Arc<CapturedResidentGraph>,
    readout: Readout,
) -> Result<()> {
    let count = sessions.len();
    let tokens = (0..count).map(|r| IDS[(r + 3) % 8]).collect::<Vec<_>>();
    let order = (0..count).collect::<Vec<_>>();
    let rows = rows(sessions, &tokens, &order)?;
    let bindings = roots(sessions);
    let prior = states(sessions).await?;
    let bytes = model.service.resident_bytes()?;
    let enqueues = graph.replay_enqueues();
    let required = model
        .state_bytes
        .checked_mul(count)
        .and_then(|n| {
            n.checked_add(if matches!(readout, Readout::Greedy) {
                12 * count
            } else {
                0
            })
        })
        .ok_or_else(|| Error::Limit("wide graph pressure footprint".into()))?;
    let fill = model
        .service
        .remaining_capacity()?
        .checked_sub(required)
        .and_then(|n| n.checked_add(1))
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::Limit("wide graph qualifier pressure headroom".into()))?;
    let filler = model
        .service
        .allocate_uninitialized(fill, BufferRole::Scratch)?;
    let queue = model.service.queue().stats()?;
    let refused = model
        .prepare_fused_execution(
            &rows,
            None,
            readout,
            super::super::fused::FusedExecution::Graph(graph.clone()),
        )
        .await?;
    if refused.is_some()
        || graph.replay_enqueues() != enqueues
        || model.service.queue().stats()? != queue
        || model.service.resident_bytes()? != bytes + fill
        || model.service.pending_events()? != 0
    {
        return Err(Error::State(
            "wide graph capacity refusal was not prelaunch None".into(),
        ));
    }
    drop(filler);
    idle(model, bytes)?;
    unchanged(sessions, &bindings, &prior).await?;
    let discarded = model.prepare_graph_readout(&rows, graph, readout).await?;
    candidate_shape(&discarded, count, readout)?;
    if graph.replay_enqueues() != enqueues + 1 {
        return Err(Error::State(
            "wide graph Ready cancellation actual enqueue".into(),
        ));
    }
    drop(discarded);
    unchanged(sessions, &bindings, &prior).await?;
    idle(model, bytes)?;
    // Actual Ready proposal loses every sibling's publication when one row
    // restores its own original snapshot (same frontier/state, new epoch).
    let stale = model.prepare_graph_readout(&rows, graph, readout).await?;
    drop(rows);
    candidate_shape(&stale, count, readout)?;
    let snapshot = sessions[count - 1].snapshot();
    sessions[count - 1].restore(&snapshot)?;
    let changed = roots(sessions);
    if changed[..count - 1] != bindings[..count - 1]
        || changed[count - 1] == bindings[count - 1]
        || changed[count - 1].frontier() != bindings[count - 1].frontier()
        || stale
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
    {
        return Err(Error::State(
            "wide graph stale per-row epoch published cohort".into(),
        ));
    }
    drop(snapshot);
    unchanged(sessions, &changed, &prior).await?;
    if graph.replay_enqueues() != enqueues + 2 {
        return Err(Error::State(
            "wide graph abandoned/stale actual enqueue count".into(),
        ));
    }
    idle(model, bytes)
}

async fn reference_rows(
    model: &Arc<ResidentModel>,
    cpus: &mut [QwenDenseCpu],
    histories: &[Vec<u32>],
    tokens: &[u32],
    readout: Readout,
) -> Result<Vec<Vec<f32>>> {
    if cpus.len() != histories.len() || cpus.len() != tokens.len() {
        return Err(Error::State("wide graph oracle coverage".into()));
    }
    let mut expected = Vec::with_capacity(tokens.len());
    for ((cpu, history), &token) in cpus.iter_mut().zip(histories).zip(tokens) {
        let logits = cpu.forward(token)?;
        let uploads = model.service.uploaded_value_bytes();
        let mut b1 = model.session().await?;
        if model.service.uploaded_value_bytes().checked_sub(uploads)
            != Some(model.state_bytes as u64)
        {
            return Err(Error::State(
                "wide graph independent B1 state upload".into(),
            ));
        }
        b1.forward(history, false).await?;
        let before = b1.snapshot();
        let got = b1.forward(&[token], false).await?;
        exact(
            &logits,
            &got.logits,
            "wide graph accepted independent B1 fullV",
        )?;
        let state = cpu.snapshot().domain_values();
        if state.len() != 42 {
            return Err(Error::State("wide graph CPU all42 coverage".into()));
        }
        compare_domains(&state, &b1.snapshot().read_domains().await?)?;
        if matches!(readout, Readout::Greedy) {
            b1.restore(&before)?;
            let greedy = b1.forward_greedy(&[token]).await?;
            if !greedy.logits.is_empty() || greedy.bytes != 16 {
                return Err(Error::State(
                    "wide graph independent B1 greedy packet geometry".into(),
                ));
            }
            packet(&greedy, &b1, selected(&logits)?)?;
            compare_domains(&state, &b1.snapshot().read_domains().await?)?;
            drop(greedy);
        }
        drop(before);
        if model.service.uploaded_value_bytes() != uploads + model.state_bytes as u64 {
            return Err(Error::State("wide graph B1 warm value upload".into()));
        }
        expected.push(logits);
    }
    Ok(expected)
}

struct TeacherPolicy {
    readout: Readout,
    reverse: bool,
}

async fn teacher_step(
    model: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    cpus: &mut [QwenDenseCpu],
    histories: &mut [Vec<u32>],
    tokens: &[u32],
    graph: &Arc<CapturedResidentGraph>,
    policy: TeacherPolicy,
) -> Result<usize> {
    let TeacherPolicy { readout, reverse } = policy;
    let count = sessions.len();
    let expected = reference_rows(model, cpus, histories, tokens, readout).await?;
    let uploads = model.service.uploaded_value_bytes();
    let prior = states(sessions).await?;
    let snapshots = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let order = if reverse {
        (0..count).rev().collect::<Vec<_>>()
    } else {
        (0..count).collect::<Vec<_>>()
    };
    let before_frontiers = sessions
        .iter()
        .map(|s| s.binding().frontier())
        .collect::<Vec<_>>();
    // SAME original native cohort/publisher, followed by checked authority
    // restore. No second B-shaped GPU lane or invented checkpoint/root.
    let native = match readout {
        Readout::None => {
            model
                .prepare_state_fused_cohort(&rows(sessions, tokens, &order)?)
                .await?
        }
        Readout::Logits => {
            model
                .prepare_logits_fused_cohort(&rows(sessions, tokens, &order)?)
                .await?
        }
        Readout::Greedy => {
            model
                .prepare_greedy_fused_cohort(&rows(sessions, tokens, &order)?)
                .await?
        }
    };
    candidate_shape(&native, count, readout)?;
    let native_dispatches = native.dispatches();
    let native_dots = native.fused_dispatches();
    let reads = native.commit(&mut ordered_sessions(sessions, &order)?)?;
    if reads.len() != count {
        return Err(Error::State(
            "wide graph native committed row coverage".into(),
        ));
    }
    for (index, &row) in order.iter().enumerate() {
        match readout {
            Readout::Logits => exact(
                &expected[row],
                &reads[index].logits,
                "wide graph ordinary native fullV",
            )?,
            Readout::Greedy => packet(&reads[index], &sessions[row], selected(&expected[row])?)?,
            Readout::None => {}
        }
        compare_domains(
            &cpus[row].snapshot().domain_values(),
            &sessions[row].snapshot().read_domains().await?,
        )?;
    }
    if matches!(readout, Readout::Greedy) {
        let raw = completed_logits(model, sessions[0].binding().root(), count).await?;
        for (index, &row) in order.iter().enumerate() {
            exact(
                &expected[row],
                &raw[index * 248_320..(index + 1) * 248_320],
                "wide graph ordinary native greedy diagnostic fullV",
            )?;
        }
    }
    let native_states = states(sessions).await?;
    drop(reads);
    for ((session, snapshot), expected_prior) in sessions.iter_mut().zip(&snapshots).zip(&prior) {
        session.restore(snapshot)?;
        compare_domains(expected_prior, &session.snapshot().read_domains().await?)?;
    }
    drop(snapshots);
    if sessions
        .iter()
        .map(|s| s.binding().frontier())
        .ne(before_frontiers.iter().copied())
    {
        return Err(Error::State("wide graph restore frontier changed".into()));
    }
    let bytes = model.service.resident_bytes()?;
    let candidate = model
        .prepare_graph_readout(&rows(sessions, tokens, &order)?, graph, readout)
        .await?;
    candidate_shape(&candidate, count, readout)?;
    if candidate.dispatches() != native_dispatches
        || candidate.fused_dispatches() != native_dots
        || candidate
            .prior_frontiers()
            .ne(order.iter().map(|&row| sessions[row].binding().frontier()))
    {
        return Err(Error::State(
            "wide graph exact body/outside/independent frontier receipt".into(),
        ));
    }
    let kernels = candidate.dispatches();
    let reads = candidate.commit(&mut ordered_sessions(sessions, &order)?)?;
    if reads.len() != count {
        return Err(Error::State("wide graph committed row omission".into()));
    }
    for (index, &row) in order.iter().enumerate() {
        match readout {
            Readout::Logits => exact(
                &expected[row],
                &reads[index].logits,
                "wide graph actual graph fullV",
            )?,
            Readout::Greedy => packet(&reads[index], &sessions[row], selected(&expected[row])?)?,
            Readout::None => {}
        }
        compare_domains(
            &cpus[row].snapshot().domain_values(),
            &sessions[row].snapshot().read_domains().await?,
        )?;
        compare_domains(
            &native_states[row],
            &sessions[row].snapshot().read_domains().await?,
        )?;
    }
    if matches!(readout, Readout::Greedy) {
        let raw = completed_logits(model, sessions[0].binding().root(), count).await?;
        for (index, &row) in order.iter().enumerate() {
            exact(
                &expected[row],
                &raw[index * 248_320..(index + 1) * 248_320],
                "wide graph actual greedy diagnostic fullV",
            )?;
        }
    }
    drop(reads);
    for (history, &token) in histories.iter_mut().zip(tokens) {
        history.push(token);
    }
    if model.service.uploaded_value_bytes() != uploads {
        return Err(Error::State(
            "wide graph warm native/graph uploaded values".into(),
        ));
    }
    idle(model, bytes)?;
    Ok(kernels)
}

pub(crate) async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
) -> Result<()> {
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Unsupported("wide graph rows must be 2/4/8".into()));
    }
    let (source, model) = loaded(device, path, provider_cap, device_cap).await?;
    let key = model.service.physical_key();
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("wide graph tokenizer path".into()))?;
    let eos = QwenTokenizer::from_file(&parent.join("tokenizer.json"))?.eos();
    if u64::from(eos) >= model.vocab {
        return Err(Error::State("wide graph original trained EOS input".into()));
    }
    let mut cpus = Vec::with_capacity(count);
    let mut sessions = Vec::with_capacity(count);
    let mut histories = Vec::with_capacity(count);
    for row in 0..count {
        let history = (0..2 + row % 3)
            .map(|at| IDS[(row + at) % 8])
            .collect::<Vec<_>>();
        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
        let expected = cpu.prefill_transaction(&history)?;
        let mut session = model.session().await?;
        exact(
            &expected,
            &session.forward(&history, false).await?.logits,
            "wide graph distinct real histories",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        cpus.push(cpu);
        sessions.push(session);
        histories.push(history);
    }
    let tokens = (0..count).map(|row| IDS[(row + 3) % 8]).collect::<Vec<_>>();
    let order = (0..count).collect::<Vec<_>>();
    let prefix = model
        .capture_body(
            &rows(&sessions, &tokens, &order)?,
            ResidentOutput::StateOnly,
        )
        .await?;
    if prefix.profile.output != ResidentOutput::StateOnly || prefix.profile.rows != count {
        return Err(Error::State("wide graph typed StateOnly body key".into()));
    }
    let before = roots(&sessions);
    let original = states(&sessions).await?;
    let bytes = model.service.resident_bytes()?;
    if model
        .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &prefix, Readout::Logits)
        .await
        .is_ok()
        || model
            .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &prefix, Readout::Greedy)
            .await
            .is_ok()
        || model
            .prepare_graph_readout(
                &rows(
                    &sessions[..count / 2],
                    &tokens[..count / 2],
                    &(0..count / 2).collect::<Vec<_>>(),
                )?,
                &prefix,
                Readout::None,
            )
            .await
            .is_ok()
        || prefix.replay_enqueues() != 0
    {
        return Err(Error::State(
            "wide graph typed StateOnly/readout/band refusal".into(),
        ));
    }
    unchanged(&sessions, &before, &original).await?;
    idle(&model, bytes)?;
    pressure_and_abandon(&model, &mut sessions, &prefix, Readout::None).await?;
    let prefix_receipt = prefix.receipt();
    let mut kernel_total = 0;
    for step in 0..2 {
        let mut tokens = (0..count)
            .map(|row| IDS[(row + step + 4) % 8])
            .collect::<Vec<_>>();
        if step == 1 {
            tokens[count - 1] = eos;
        } // Genuine tokenizer EOS is a teacher, not generated output.
        kernel_total += teacher_step(
            &model,
            &mut sessions,
            &mut cpus,
            &mut histories,
            &tokens,
            &prefix,
            TeacherPolicy {
                readout: Readout::None,
                reverse: step % 2 != 0,
            },
        )
        .await?;
    }
    if prefix.replay_enqueues() != 4 {
        return Err(Error::State(
            "wide graph StateOnly host enqueue count".into(),
        ));
    }
    let prefix_enqueues = prefix.replay_enqueues();
    prefix.retire(sessions[0].binding().root()).await?;
    let graph = model
        .capture_body(&rows(&sessions, &tokens, &order)?, ResidentOutput::Logits)
        .await?;
    let logits_receipt = graph.receipt();
    let bytes = model.service.resident_bytes()?;
    let before = roots(&sessions);
    let prior = states(&sessions).await?;
    if model
        .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &graph, Readout::None)
        .await
        .is_ok()
        || graph.replay_enqueues() != 0
    {
        return Err(Error::State(
            "wide graph Logits body accepted StateOnly key".into(),
        ));
    }
    unchanged(&sessions, &before, &prior).await?;
    idle(&model, bytes)?;
    for (step, readout) in [Readout::Logits, Readout::Greedy].into_iter().enumerate() {
        pressure_and_abandon(&model, &mut sessions, &graph, readout).await?;
        let tokens = (0..count)
            .map(|row| IDS[(row + step + 6) % 8])
            .collect::<Vec<_>>();
        kernel_total += teacher_step(
            &model,
            &mut sessions,
            &mut cpus,
            &mut histories,
            &tokens,
            &graph,
            TeacherPolicy {
                readout,
                reverse: step % 2 == 0,
            },
        )
        .await?;
    }
    if graph.replay_enqueues() != 6 {
        return Err(Error::State(
            "wide graph Logits/Greedy host enqueue count".into(),
        ));
    }
    let logit_enqueues = graph.replay_enqueues();
    graph.retire(sessions[0].binding().root()).await?;
    {
        let _allocation = model.allocation.lock().await;
        let mut workspace = model.workspace.lock().await;
        if workspace
            .batch
            .as_ref()
            .is_none_or(|b| b.program.graph_pinned())
        {
            return Err(Error::State("wide graph retired band still pinned".into()));
        }
        workspace.batch = None;
    }
    drop(cpus);
    drop(sessions);
    drop(histories);
    drop(model);
    drop(source);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State(
            "wide graph final original resource ownership".into(),
        ));
    }
    println!(
        "P12_GRAPH_WIDE_PASS B={count} original=GGUF capacity=512 AV_padding=256 independent_CPU_B1_native=true all_vocab=248320 all_domains=42 state_only_teacher_steps=2 genuine_EOS_teacher=true prefix_generated_output=0 future_logits_steps=1 outside_greedy_steps=1 sampler_kernels=4 prefix_body_K={} logits_body_K={} private_warm_K={:?}/{:?} actual_graph_host_enqueues={prefix_enqueues}/{logit_enqueues} committed_kernel_sites={kernel_total} greedy_fullV=separate_readonly_diagnostic capture_global_barrier=UNQUALIFIED metadata_admission=UNQUALIFIED final_reserved=0",
        prefix_receipt.captured_kernels,
        logits_receipt.captured_kernels,
        prefix_receipt.warm_dispatches,
        logits_receipt.warm_dispatches
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
pub(crate) async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
    mode: &str,
) -> Result<()> {
    let readout = parse_mode(mode)?;
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Unsupported("wide graph abort rows2/4/8".into()));
    }
    let (source, model) = loaded(device.clone(), path, provider_cap, device_cap).await?;
    let key = model.service.physical_key();
    let cap = model.service.capacity() as u64;
    let mut sessions = Vec::with_capacity(count);
    for row in 0..count {
        let history = (0..2 + row % 3)
            .map(|at| IDS[(row + at) % 8])
            .collect::<Vec<_>>();
        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
        let mut session = model.session().await?;
        exact(
            &cpu.prefill_transaction(&history)?,
            &session.forward(&history, false).await?.logits,
            "wide graph abort CPU/B1 prefix",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        sessions.push(session);
    }
    let tokens = (0..count).map(|row| IDS[(row + 4) % 8]).collect::<Vec<_>>();
    let order = (0..count).collect::<Vec<_>>();
    let graph = model
        .capture_body(&rows(&sessions, &tokens, &order)?, body(readout))
        .await?;
    let bindings = roots(&sessions);
    let prior = states(&sessions).await?;
    let old_ready = match readout {
        Readout::None => {
            model
                .prepare_state_fused_cohort(&rows(&sessions, &tokens, &order)?)
                .await?
        }
        Readout::Logits => {
            model
                .prepare_logits_fused_cohort(&rows(&sessions, &tokens, &order)?)
                .await?
        }
        Readout::Greedy => {
            model
                .prepare_greedy_fused_cohort(&rows(&sessions, &tokens, &order)?)
                .await?
        }
    };
    let client = model.service.client().clone();
    let before = crate::physical_quarantine::quarantined_handles_for(&client)?;
    let owners = crate::physical_quarantine::quarantined_completion_owners_for(&client)?;
    let weak_graph = Arc::downgrade(&graph);
    let weak_service = Arc::downgrade(&model.service);
    model.launch_seen.store(0, Ordering::Release);
    model.abort_after_launch.store(1, Ordering::Release);
    let pending_rows = rows(&sessions, &tokens, &order)?;
    let mut future = Box::pin(model.prepare_graph_readout(&pending_rows, &graph, readout));
    loop {
        if let Some(result) = futures_lite::future::poll_once(future.as_mut()).await {
            return Err(Error::State(format!(
                "wide graph abort returned before actual pending: success={}",
                result.is_ok()
            )));
        }
        if model.launch_seen.load(Ordering::Acquire) > 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    if graph.replay_enqueues() != 1 {
        return Err(Error::State(
            "wide graph abort actual executable enqueue".into(),
        ));
    }
    drop(future);
    drop(pending_rows);
    let handles = model.launch_seen.load(Ordering::Acquire);
    let retained = crate::physical_quarantine::quarantined_handles_for(&client)?;
    if handles == 0
        || retained.saturating_sub(before) < handles
        || roots(&sessions) != bindings
        || model.service.ensure_usable().is_ok()
        || prior.len() != count
        || prior.iter().any(|d| d.len() != 42)
        || old_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
        || crate::physical_quarantine::quarantined_completion_owners_for(&client)? <= owners
    {
        return Err(Error::State(
            "wide graph unknown completion ownership/publication".into(),
        ));
    }
    if sessions[0].prepare(&[11], false).await.is_ok()
        || model
            .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &graph, readout)
            .await
            .is_ok()
        || ResidentService::new(
            device,
            source.source_identity(),
            ResourceOwner::new("wide-graph-abort", "refused")?,
            1024 * 1024,
            device_cap,
        )
        .is_ok()
    {
        return Err(Error::State(
            "wide graph unknown service admitted work/model".into(),
        ));
    }
    drop(graph);
    drop(sessions);
    drop(model);
    drop(source);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if weak_graph.upgrade().is_none()
        || weak_service.upgrade().is_none()
        || budget.reserved != cap
        || !budget.fenced
        || crate::physical_quarantine::quarantined_handles_for(&client)? < retained
    {
        return Err(Error::State(
            "wide graph unknown executable/source/envelope dropped".into(),
        ));
    }
    println!(
        "P12_GRAPH_WIDE_ABORT_PASS B={count} mode={} actual_graph_enqueue=1 roots_unchanged=true prior_all42=observed_before_abort post_abort_readback=NOT_ATTEMPTED completion=unknown executable=retained driver_destroy=not_requested subsequent_model=refused retained_envelope={cap} trusted_recovery=false",
        mode_name(readout)
    );
    Ok(())
}
