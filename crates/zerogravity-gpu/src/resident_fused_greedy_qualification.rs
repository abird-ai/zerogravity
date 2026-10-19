//! Native greedy/feedback proof over original source math and independent roots.
use super::*;

const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];
const VOCABULARY: usize = 248_320;
const SAMPLER_DISPATCHES: usize = 4;
type Domains = Vec<(StateSlot, Vec<f32>)>;

fn greedy(values: &[f32]) -> Result<(u32, u32)> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::State(
            "greedy source oracle requires finite logits".into(),
        ));
    }
    let best = (0..values.len())
        .max_by(|a, b| values[*a].total_cmp(&values[*b]).then_with(|| b.cmp(a)))
        .expect("nonempty logits");
    Ok((best as u32, values[best].to_bits()))
}

fn roots(sessions: &[ResidentSession]) -> Vec<RootBinding> {
    sessions.iter().map(ResidentSession::binding).collect()
}

fn unchanged(sessions: &[ResidentSession], expected: &[RootBinding]) -> Result<()> {
    if roots(sessions) != expected {
        return Err(Error::State(
            "private/refused greedy cohort changed roots".into(),
        ));
    }
    Ok(())
}

async fn domains(sessions: &[ResidentSession]) -> Result<Vec<Domains>> {
    let mut values = Vec::with_capacity(sessions.len());
    for session in sessions {
        values.push(session.snapshot().read_domains().await?);
    }
    Ok(values)
}

async fn same_domains(sessions: &[ResidentSession], expected: &[Domains]) -> Result<()> {
    if sessions.len() != expected.len() {
        return Err(Error::State("greedy state row coverage".into()));
    }
    for (session, expected) in sessions.iter().zip(expected) {
        compare_domains(expected, &session.snapshot().read_domains().await?)?;
    }
    Ok(())
}

fn quiet(resident: &ResidentModel, bytes: usize) -> Result<()> {
    if resident.service.pending_events()? != 0 || resident.service.resident_bytes()? != bytes {
        return Err(Error::State(
            "greedy candidate/packet/event lease not reclaimed".into(),
        ));
    }
    Ok(())
}

async fn host(
    resident: &Arc<ResidentModel>,
    sessions: &[ResidentSession],
    ids: &[u32],
) -> Result<PreparedResidentCohort> {
    if ids.len() != sessions.len() {
        return Err(Error::State("greedy host row coverage".into()));
    }
    let rows = sessions
        .iter()
        .zip(ids)
        .map(|(s, &id)| (s, id))
        .collect::<Vec<_>>();
    resident.prepare_greedy_fused_cohort(&rows).await
}

async fn feedback(
    resident: &Arc<ResidentModel>,
    sessions: &[ResidentSession],
    packets: &[ResidentToken],
) -> Result<PreparedResidentCohort> {
    if packets.len() != sessions.len() {
        return Err(Error::State("greedy device row coverage".into()));
    }
    let rows = sessions.iter().zip(packets).collect::<Vec<_>>();
    resident.prepare_device_tokens_fused_cohort(&rows).await
}

fn private_receipt(candidate: &PreparedResidentCohort, count: usize, frontier: u64) -> Result<()> {
    if candidate.len() != count
        || candidate.prior_frontier() != frontier
        || candidate.fused_dispatches() == 0
        || candidate.native_greedy_dispatches() != SAMPLER_DISPATCHES
        || candidate.dispatches() <= candidate.native_greedy_dispatches()
        || candidate.prepared.iter().any(|row| {
            !row.readback.logits.is_empty()
                || row
                    .readback
                    .device_token
                    .as_ref()
                    .is_none_or(|token| token.producer.is_some() || token.lease.bytes() != 12)
        })
    {
        return Err(Error::State(
            "native greedy private model/sampler/packet receipt".into(),
        ));
    }
    let bytes = candidate
        .prepared
        .iter()
        .map(|row| row.readback.bytes)
        .sum::<usize>();
    let chunk_bytes = candidate
        .prepared
        .iter()
        .map(|row| row.readback.chunk_bytes)
        .sum::<usize>();
    let dispatches = candidate
        .prepared
        .iter()
        .map(|row| row.readback.dispatches)
        .sum::<usize>();
    if bytes != 4 + 12 * count || chunk_bytes != bytes || dispatches != candidate.dispatches() {
        return Err(Error::State(
            "native greedy shared physical readback/dispatch accounting".into(),
        ));
    }
    Ok(())
}

fn packets(
    mut readback: Vec<InferenceReadback>,
    sessions: &[ResidentSession],
    expected: &[(u32, u32)],
) -> Result<Vec<ResidentToken>> {
    if readback.len() != sessions.len() || expected.len() != sessions.len() {
        return Err(Error::State("native greedy committed row coverage".into()));
    }
    let mut output = Vec::with_capacity(sessions.len());
    let mut allocations = std::collections::BTreeSet::new();
    for ((read, session), &(id, bits)) in readback.iter_mut().zip(sessions).zip(expected) {
        let token = read
            .device_token
            .take()
            .ok_or_else(|| Error::State("native greedy missing private packet".into()))?;
        if !read.logits.is_empty()
            || token.id != id
            || token.max_bits != bits
            || token.producer != Some(session.binding())
            || token.lease.bytes() != 12
            || !Arc::ptr_eq(token.lease.service(), &session.model.service)
            || !allocations.insert(token.lease.id())
        {
            return Err(Error::State(
                "native greedy original ID/max/producer/packet lease".into(),
            ));
        }
        output.push(token);
    }
    Ok(output)
}

fn publication(before: &[RootBinding], sessions: &[ResidentSession]) -> Result<()> {
    for (&prior, session) in before.iter().zip(sessions) {
        let after = session.binding();
        if after.source() != prior.source()
            || after.session() != prior.session()
            || after.session_epoch() != prior.session_epoch()
            || after.root() != prior.root() + 1
            || after.frontier() != prior.frontier() + 1
        {
            return Err(Error::State("native greedy independent publication".into()));
        }
    }
    Ok(())
}

async fn refusals(
    resident: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    tokens: &[ResidentToken],
) -> Result<()> {
    let before = roots(sessions);
    let state = domains(sessions).await?;
    let bytes = resident.service.resident_bytes()?;
    let mut swapped = tokens.to_vec();
    swapped.swap(0, 1);
    let mut duplicate = tokens.to_vec();
    duplicate[1] = duplicate[0].clone();
    let mut bad = IDS[..sessions.len()].to_vec();
    bad[sessions.len() - 1] = u32::MAX;
    if feedback(resident, sessions, &swapped).await.is_ok()
        || feedback(resident, sessions, &duplicate).await.is_ok()
        || host(resident, sessions, &bad).await.is_ok()
    {
        return Err(Error::State(
            "greedy swapped/duplicate/bad row accepted".into(),
        ));
    }
    let dropped = feedback(resident, sessions, tokens).await?;
    private_receipt(&dropped, sessions.len(), before[0].frontier())?;
    unchanged(sessions, &before)?;
    drop(dropped);
    let reordered = feedback(resident, sessions, tokens).await?;
    let mut live = sessions.iter_mut().collect::<Vec<_>>();
    live.swap(0, 1);
    if reordered.commit(&mut live).is_ok() {
        return Err(Error::State("greedy reordered publication accepted".into()));
    }
    drop(live);
    unchanged(sessions, &before)?;
    same_domains(sessions, &state).await?;
    quiet(resident, bytes)?;
    Ok(())
}

pub(crate) async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
) -> Result<()> {
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Limit("native greedy requires B2/B4/B8".into()));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("greedy source path".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    if source.geometry().vocab_size != VOCABULARY as u64 {
        return Err(Error::Unsupported(
            "native greedy original vocabulary".into(),
        ));
    }
    let tokenizer = zerogravity_model_qwen35::tokenize::QwenTokenizer::from_file(
        &parent.join("tokenizer.json"),
    )?;
    let eos = tokenizer.eos();
    if eos as usize >= VOCABULARY {
        return Err(Error::State(
            "greedy tokenizer EOS exceeds source vocabulary".into(),
        ));
    }
    let resident =
        load_qualified(device, Arc::clone(&source), 512, provider_cap, device_cap).await?;
    let original_bytes = resident.service.resident_bytes()?;
    let mut cpus = Vec::with_capacity(count);
    let mut sessions = Vec::with_capacity(count);
    let mut shadows = Vec::with_capacity(count);
    let mut b1 = Vec::with_capacity(count);
    for (row, &first) in IDS.iter().take(count).enumerate() {
        let history = [first, IDS[(row + 3) % IDS.len()]];
        let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let expected = cpu.prefill_transaction(&history)?;
        let mut lane = resident.session().await?;
        let mut shadow = resident.session().await?;
        let mut reference = resident.session().await?;
        lane.prefill_without_logits(&history, false).await?;
        shadow.prefill_without_logits(&history, false).await?;
        exact(
            &expected,
            &reference.forward(&history, false).await?.logits,
            "greedy independent CPU/B1 nonzero prefix",
        )?;
        let state = cpu.snapshot().domain_values();
        for physical in [&lane, &shadow, &reference] {
            compare_domains(&state, &physical.snapshot().read_domains().await?)?;
        }
        cpus.push(cpu);
        sessions.push(lane);
        shadows.push(shadow);
        b1.push(reference);
    }
    let uploads = resident.service.uploaded_value_bytes();
    let mut device_packets = Vec::new();
    let mut next_ids = IDS[..count].to_vec();
    next_ids[count - 1] = eos; // Real EOS input, not a fabricated trained winner.
    let mut stable_bytes = None;
    let mut total_dispatches = 0usize;
    let mut total_projections = 0usize;
    let mut total_sampler = 0usize;
    let mut total_readback = 0usize;
    for step in 0..6 {
        if step == 1 {
            refusals(&resident, &mut sessions, &device_packets).await?;
        }
        if step == 5 {
            let before = roots(&sessions);
            let state = domains(&sessions).await?;
            let bytes = resident.service.resident_bytes()?;
            let candidate = feedback(&resident, &sessions, &device_packets).await?;
            let saved = sessions[0].snapshot();
            sessions[0].restore(&saved)?;
            drop(saved);
            let restored = roots(&sessions);
            if restored[0].session_epoch() != before[0].session_epoch() + 1
                || candidate
                    .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
                    .is_ok()
                || feedback(&resident, &sessions, &device_packets)
                    .await
                    .is_ok()
            {
                return Err(Error::State(
                    "greedy restore epoch accepted stale packet/proposal".into(),
                ));
            }
            unchanged(&sessions, &restored)?;
            same_domains(&sessions, &state).await?;
            quiet(&resident, bytes)?;
        }
        let before = roots(&sessions);
        let mut expected_logits = Vec::with_capacity(count);
        let mut expected_states = Vec::with_capacity(count);
        let mut selected = Vec::with_capacity(count);
        for ((cpu, reference), &id) in cpus.iter_mut().zip(&mut b1).zip(&next_ids) {
            let logits = cpu.forward(id)?;
            let prior = reference.snapshot();
            let actual = reference.forward(&[id], false).await?;
            if logits.len() != VOCABULARY {
                return Err(Error::State("greedy full source vocabulary".into()));
            }
            exact(
                &logits,
                &actual.logits,
                "greedy CPU/B1 complete source logits",
            )?;
            let state = cpu.snapshot().domain_values();
            if state.len() != 42 {
                return Err(Error::State("greedy original state coverage".into()));
            }
            compare_domains(&state, &reference.snapshot().read_domains().await?)?;
            let want = greedy(&logits)?;
            reference.restore(&prior)?;
            drop(prior);
            let mut b1_greedy = reference.forward_greedy(&[id]).await?;
            let b1_packet = b1_greedy
                .device_token
                .take()
                .ok_or_else(|| Error::State("accepted B1 greedy packet absent".into()))?;
            if !b1_greedy.logits.is_empty()
                || b1_greedy.bytes != 16
                || b1_greedy.chunk_bytes != 16
                || (b1_packet.id, b1_packet.max_bits) != want
                || b1_packet.producer != Some(reference.binding())
            {
                return Err(Error::State(
                    "accepted B1 greedy packet differs from full logits".into(),
                ));
            }
            compare_domains(&state, &reference.snapshot().read_domains().await?)?;
            drop(b1_packet);
            drop(b1_greedy);
            selected.push(want);
            expected_logits.push(logits);
            expected_states.push(state);
        }
        let shadow_rows = shadows
            .iter()
            .zip(&next_ids)
            .map(|(s, &id)| (s, id))
            .collect::<Vec<_>>();
        let full = resident.prepare_logits_fused_cohort(&shadow_rows).await?;
        if full.fused_dispatches() == 0 {
            return Err(Error::State(
                "greedy native logits shadow was serial".into(),
            ));
        }
        let full = full.commit(&mut shadows.iter_mut().collect::<Vec<_>>())?;
        for (row, actual) in full.iter().enumerate() {
            exact(
                &expected_logits[row],
                &actual.logits,
                "greedy native full-vocabulary shadow",
            )?;
        }
        same_domains(&shadows, &expected_states).await?;
        let candidate = if step == 0 || step == 5 {
            host(&resident, &sessions, &next_ids).await?
        } else if step == 2 {
            let mut tampered = device_packets.clone();
            tampered[0].id = u32::MAX;
            tampered[0].max_bits = f32::NAN.to_bits();
            feedback(&resident, &sessions, &tampered).await?
        } else {
            feedback(&resident, &sessions, &device_packets).await?
        };
        private_receipt(&candidate, count, before[0].frontier())?;
        unchanged(&sessions, &before)?;
        total_dispatches += candidate.dispatches();
        total_projections += candidate.fused_dispatches();
        total_sampler += candidate.native_greedy_dispatches();
        let result = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        total_readback += result.iter().map(|read| read.bytes).sum::<usize>();
        publication(&before, &sessions)?;
        let next = packets(result, &sessions, &selected)?;
        same_domains(&sessions, &expected_states).await?;
        if step == 1 {
            let current = roots(&sessions);
            let last = count - 1;
            let previous = device_packets[last]
                .producer
                .ok_or_else(|| Error::State("previous greedy packet has no producer".into()))?;
            if previous.session_epoch() != current[last].session_epoch()
                || previous.root() >= current[last].root()
                || previous.session() != current[last].session()
                || previous.source() != current[last].source()
            {
                return Err(Error::State(
                    "greedy stale-root fixture is not same-epoch".into(),
                ));
            }
            let bytes = resident.service.resident_bytes()?;
            let pending = resident.service.pending_events()?;
            let mut mixed = next.clone();
            mixed[last] = device_packets[last].clone();
            if feedback(&resident, &sessions, &mixed).await.is_ok() {
                return Err(Error::State(
                    "same-epoch previous-root greedy packet accepted".into(),
                ));
            }
            unchanged(&sessions, &current)?;
            same_domains(&sessions, &expected_states).await?;
            quiet(&resident, bytes)?;
            if resident.service.pending_events()? != pending {
                return Err(Error::State(
                    "same-epoch packet refusal changed pending events".into(),
                ));
            }
        }
        if step == 5
            && feedback(&resident, &sessions, &device_packets)
                .await
                .is_ok()
        {
            return Err(Error::State(
                "greedy root advance accepted previous packets".into(),
            ));
        }
        device_packets = next;
        next_ids = selected.iter().map(|&(id, _)| id).collect();
        if resident.service.uploaded_value_bytes() != uploads {
            return Err(Error::State(
                "native greedy/feedback uploaded warm values".into(),
            ));
        }
        let bytes = *stable_bytes.get_or_insert(resident.service.resident_bytes()?);
        quiet(&resident, bytes)?;
    }
    let bytes_with_packets =
        stable_bytes.ok_or_else(|| Error::State("missing greedy live admission".into()))?;
    drop(device_packets);
    drop(sessions);
    drop(shadows);
    drop(b1);
    quiet(
        &resident,
        bytes_with_packets - count * (12 + 3 * resident.state_bytes),
    )?;
    identical_packet_bindings(&resident, &source).await?;
    // Explicit fixture uploads begin only after the source-model zero-upload gate.
    selector_fixtures(&resident, count, eos).await?;
    let persistent = resident.service.resident_bytes()?;
    if persistent < original_bytes {
        return Err(Error::State("greedy lost original resident owner".into()));
    }
    let key = resident.service.physical_key();
    drop(resident);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State("greedy final physical envelope leaked".into()));
    }
    println!(
        "P12_FUSED_GREEDY_GPU_PASS B={count} source_steps=6 host_seed_steps=2 device_feedback_steps=4 \
         CPU_B1_native_full248320_logits=bit_exact selected_ID_max_bits=exact all42domains=exact \
         compact_readback_bytes={total_readback} bytes_per_cohort={} physical_dispatches={total_dispatches} \
         native_projection_dispatches={total_projections} native_sampler_dispatches={total_sampler} \
         swapped_duplicate_foreign_stale_restore_packets=refused same_epoch_old_root=refused \
         host_metadata=ignored_by_device_feedback \
         abandoned_reordered_stale_proposals=unpublished genuine_EOS_input=source_exact \
         private_packet_state_leases=reclaimed final_reserved_bytes=0 HTTP_fused_serving=not_qualified",
        4 + 12 * count,
    );
    Ok(())
}

async fn identical_packet_bindings(
    resident: &Arc<ResidentModel>,
    source: &Arc<LoadedQwenModel>,
) -> Result<()> {
    let mut sessions = Vec::with_capacity(2);
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(source))?;
    let expected = greedy(&cpu.prefill_transaction(&[11, 353, 17])?)?;
    for _ in 0..2 {
        let mut session = resident.session().await?;
        session.prefill_without_logits(&[11, 353], false).await?;
        sessions.push(session);
    }
    let candidate = host(resident, &sessions, &[17, 17]).await?;
    private_receipt(&candidate, 2, 2)?;
    let result = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
    let mut tokens = packets(result, &sessions, &[expected, expected])?;
    let cpu_state = cpu.snapshot().domain_values();
    for session in &sessions {
        compare_domains(&cpu_state, &session.snapshot().read_domains().await?)?;
    }
    let before = roots(&sessions);
    let state = domains(&sessions).await?;
    let bytes = resident.service.resident_bytes()?;
    tokens.swap(0, 1);
    if feedback(resident, &sessions, &tokens).await.is_ok() {
        return Err(Error::State(
            "identical-ID foreign producer packet accepted".into(),
        ));
    }
    unchanged(&sessions, &before)?;
    same_domains(&sessions, &state).await?;
    quiet(resident, bytes)?;
    drop(tokens);
    drop(sessions);
    quiet(resident, bytes - 2 * (resident.state_bytes + 12))?;
    Ok(())
}

/// Synthetic selector fixtures; expected invalid records never become tokens/roots.
async fn selector_fixtures(resident: &Arc<ResidentModel>, count: usize, eos: u32) -> Result<()> {
    use cubecl::prelude::{BufferArg, CubeCount, CubeDim};
    let service = &resident.service;
    let _admission = resident.allocation.lock().await;
    let baseline = service.resident_bytes()?;
    let sizes = crate::resident_batch_greedy::workspace_bytes(VOCABULARY, count)?;
    for rotation in 0..7 {
        let mut matrix = Vec::with_capacity(VOCABULARY * count);
        let mut expected = Vec::with_capacity(count);
        for row in 0..count {
            let case = (row + rotation) % 5;
            let mut values = vec![-1.0f32; VOCABULARY];
            match case {
                0 => {
                    values[17] = 4.0;
                    values[256] = 4.0;
                    values[VOCABULARY - 1] = 4.0;
                }
                1 => {
                    values.fill(-0.0);
                    values[17] = 0.0;
                    values[256] = 0.0;
                }
                2 => {
                    values.fill(f32::from_bits(0x8000_0001));
                    values[511] = f32::from_bits(1);
                }
                3 => {
                    let mut bits = 0x91_51_35_79u32 ^ row as u32;
                    for value in &mut values {
                        bits ^= bits << 13;
                        bits ^= bits >> 17;
                        bits ^= bits << 5;
                        let finite = if bits & 0x7f80_0000 == 0x7f80_0000 {
                            bits ^ 0x0080_0000
                        } else {
                            bits
                        };
                        *value = f32::from_bits(finite);
                    }
                    values[eos as usize] = -f32::MAX;
                }
                _ => values[eos as usize] = f32::MAX,
            }
            expected.push(greedy(&values)?);
            matrix.extend(
                values
                    .iter()
                    .flat_map(|value| value.to_bits().to_ne_bytes()),
            );
        }
        let expected_flags = match rotation {
            5 => {
                // Exactly one bad source row must poison the whole selection.
                let offset = (count * VOCABULARY - 1) * 4;
                matrix[offset..offset + 4].copy_from_slice(&f32::NAN.to_bits().to_ne_bytes());
                crate::program_kernels::NONFINITE_VALUE
            }
            6 => crate::program_kernels::INVALID_REDUCTION,
            _ => 0,
        };
        let required = matrix.len() + sizes.iter().sum::<usize>() + 12 * count + 4;
        if required > service.remaining_capacity()? {
            return Err(Error::Limit("native sampler fixture admission".into()));
        }
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: required as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut initialization = service.begin_initialization(0)?;
        let input = initialization.upload_bytes(&matrix, BufferRole::ImmutableWeights)?;
        let workspace = sizes
            .into_iter()
            .map(|size| initialization.allocate_empty(size, BufferRole::Scratch))
            .collect::<Result<Vec<_>>>()?;
        let private = (0..count)
            .map(|_| initialization.allocate_empty(12, BufferRole::Output))
            .collect::<Result<Vec<_>>>()?;
        let status = if rotation == 6 {
            initialization.upload_bytes(&expected_flags.to_ne_bytes(), BufferRole::Scratch)?
        } else {
            initialization.allocate_empty(4, BufferRole::Scratch)?
        };
        drop(initialization.complete_sync().await?);
        let handles: [Handle; 3] = std::array::from_fn(|i| workspace[i].handle());
        let outputs = private
            .iter()
            .map(ResidentLease::handle)
            .collect::<Vec<_>>();
        let mut aliased = outputs.clone();
        aliased[1] = aliased[0].clone();
        if crate::resident_batch_greedy::launch(
            service.client(),
            &input.handle(),
            VOCABULARY,
            count,
            &handles,
            &aliased,
            &status.handle(),
        )
        .is_ok()
        {
            return Err(Error::State(
                "native sampler overlapping packet allocations accepted".into(),
            ));
        }
        let mut accesses = vec![
            (input.clone(), DeviceAccess::Read),
            (status.clone(), DeviceAccess::Write),
        ];
        accesses.extend(
            workspace
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        accesses.extend(
            private
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        let flight = service.submit(0, &accesses)?;
        // SAFETY: one admitted/pinned status word, before every sampler writer.
        if rotation != 6 {
            // Status-poison fixture deliberately preserves the original flag.
            unsafe {
                crate::program_kernels::clear_step_status::launch_unchecked(
                    service.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(status.handle(), 1),
                );
            }
        }
        let launched = crate::resident_batch_greedy::launch(
            service.client(),
            &input.handle(),
            VOCABULARY,
            count,
            &handles,
            &outputs,
            &status.handle(),
        )?;
        if launched.dispatches != SAMPLER_DISPATCHES + count
            || launched.native_dispatches != SAMPLER_DISPATCHES
        {
            return Err(Error::State(
                "native sampler fixture physical dispatch count".into(),
            ));
        }
        let mut reads = vec![status.readback(0, 4)?];
        reads.extend(
            private
                .iter()
                .map(|lease| lease.readback(0, 12))
                .collect::<Result<Vec<_>>>()?,
        );
        let (ready, readback) = flight
            .complete_with_readback(&private, &reads, |read| {
                if read[0].len() != 4
                    || u32::from_ne_bytes(read[0][..4].try_into().unwrap()) != expected_flags
                {
                    return Err(Error::State("native sampler fixture step status".into()));
                }
                for (packet, &(id, bits)) in read[1..].iter().zip(&expected) {
                    if packet.len() != 12 {
                        return Err(Error::State("native sampler fixture packet size".into()));
                    }
                    let actual_id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                    let actual_bits = u32::from_ne_bytes(packet[4..8].try_into().unwrap());
                    let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                    if expected_flags == 0 && (actual_id != id || actual_bits != bits || flags != 0)
                    {
                        return Err(Error::State(
                            "native sampler finite tie/EOS/bit-order differs".into(),
                        ));
                    }
                    if expected_flags != 0 && (actual_id != u32::MAX || flags != expected_flags) {
                        return Err(Error::State(
                            "native sampler accepted poisoned row/status".into(),
                        ));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        if readback.iter().map(|bytes| bytes.len()).sum::<usize>() != 4 + 12 * count {
            return Err(Error::State(
                "native sampler fixture compact transfer".into(),
            ));
        }
        drop(ready);
        drop(handles);
        drop(outputs);
        drop(aliased);
        drop(accesses);
        drop(reads);
        drop(input);
        drop(status);
        drop(private);
        drop(workspace);
        quiet(resident, baseline)?;
    }
    println!(
        "P12_NATIVE_GREEDY_SYNTHETIC_FIXTURES_PASS B={count} full_vocab={VOCABULARY} \
        per_row_cases=5 tile_boundary_ties=first_ID signed_zero_subnormal_dense_bits=exact \
        genuine_tokenizer_EOS_winner=row_private packet_alias=refused \
        one_row_nonfinite=all_rows_invalid preexisting_status=preserved_all_rows_invalid \
        native_sampler_dispatches_per_case=4 private_packet_transport_dispatches_per_case={count} \
        published_roots=0 trained_model_EOS_stop=not_claimed"
    );
    Ok(())
}

/// Separate-process unknown completion proof includes original input/output packets.
#[cfg(feature = "hardware-abort-probe")]
pub(crate) async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    use std::sync::atomic::Ordering;

    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("greedy abort source path".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let resident = load_qualified(
        device.clone(),
        Arc::clone(&source),
        512,
        provider_cap,
        device_cap,
    )
    .await?;
    let client = device.client();
    let mut sessions = Vec::with_capacity(2);
    let mut selected = Vec::with_capacity(2);
    let mut expected = Vec::with_capacity(2);
    for (history, seed) in [([11, 353], 55), ([17, 42], 78)] {
        let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let mut session = resident.session().await?;
        exact(
            &cpu.prefill_transaction(&history)?,
            &session.forward(&history, false).await?.logits,
            "greedy abort accepted CPU/B1 prefix",
        )?;
        selected.push(greedy(&cpu.forward(seed)?)?);
        expected.push(cpu.snapshot().domain_values());
        sessions.push(session);
    }
    let seed = host(&resident, &sessions, &[55, 78]).await?;
    private_receipt(&seed, 2, 2)?;
    let seed = seed.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
    let incoming = packets(seed, &sessions, &selected)?;
    same_domains(&sessions, &expected).await?;
    let before = roots(&sessions);
    let saved = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let stale_ready = feedback(&resident, &sessions, &incoming).await?;
    private_receipt(&stale_ready, 2, 3)?;
    unchanged(&sessions, &before)?;
    let quarantined_before = crate::physical_quarantine::quarantined_handles_for(&client)?;
    resident.launch_seen.store(0, Ordering::Release);
    resident.abort_after_launch.store(1, Ordering::Release);
    let mut pending = Box::pin(feedback(&resident, &sessions, &incoming));
    loop {
        if let Some(completed) = futures_lite::future::poll_once(pending.as_mut()).await {
            return Err(Error::State(format!(
                "greedy feedback abort missed launch hook: success={}",
                completed.is_ok()
            )));
        }
        if resident.launch_seen.load(Ordering::Acquire) != 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    drop(pending);
    let submitted = resident.launch_seen.load(Ordering::Acquire);
    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
    if submitted == 0
        || quarantined.saturating_sub(quarantined_before) < submitted
        || resident.service.ensure_usable().is_ok()
        || stale_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
    {
        return Err(Error::State(
            "greedy abort original packet/state retention".into(),
        ));
    }
    unchanged(&sessions, &before)?;
    if feedback(&resident, &sessions, &incoming).await.is_ok()
        || host(&resident, &sessions, &[11, 353]).await.is_ok()
        || ResidentService::new(
            device,
            source.source_identity(),
            ResourceOwner::new("other-model", "after-greedy-feedback-abort")?,
            1024 * 1024,
            device_cap,
        )
        .is_ok()
    {
        return Err(Error::State(
            "greedy feedback fenced card admitted work/model".into(),
        ));
    }
    unchanged(&sessions, &before)?;
    // Diagnostic physical reads do not retire unknown events or permit reuse.
    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("greedy abort diagnostic barrier: {error}")))?;
    for (snapshot, expected) in saved.iter().zip(&expected) {
        abort_prior_bits(&client, snapshot, expected).await?;
    }
    let raw = client
        .read_async(incoming.iter().map(|token| token.lease.handle()).collect())
        .await
        .map_err(|error| Error::State(format!("greedy abort original packet read: {error}")))?;
    if raw.len() != 2 {
        return Err(Error::State("greedy abort packet read coverage".into()));
    }
    for (bytes, &(id, bits)) in raw.iter().zip(&selected) {
        if bytes.len() != 12
            || u32::from_ne_bytes(bytes[0..4].try_into().unwrap()) != id
            || u32::from_ne_bytes(bytes[4..8].try_into().unwrap()) != bits
            || bytes[8..12].iter().any(|byte| *byte != 0)
        {
            return Err(Error::State(
                "greedy aborted feedback mutated original packet".into(),
            ));
        }
    }
    let key = resident.service.physical_key();
    let weak = Arc::downgrade(resident.service());
    let cap = resident.service.capacity() as u64;
    drop(incoming);
    drop(saved);
    drop(sessions);
    drop(resident);
    let retained = weak
        .upgrade()
        .ok_or_else(|| Error::State("greedy abort lost charged owner".into()))?;
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != cap
        || !budget.fenced
        || retained.physical_remaining()? != device_cap - cap
        || retained.ensure_usable().is_ok()
        || crate::physical_quarantine::quarantined_handles_for(&client)? < quarantined
    {
        return Err(Error::State(
            "greedy abort physical envelope/quarantine released".into(),
        ));
    }
    println!(
        "P12_FUSED_GREEDY_FEEDBACK_ABORT_PASS B=2 prior_frontier=3 \
        actual_model_and_native_sampler_launch=true completion=unknown \
        submitted_original_handles={submitted} quarantined_original_handles={quarantined} \
        incoming_packet_bits=exact all42_prior_domains_per_row=bit_exact \
        canonical_future_KV_zero=exact roots_unchanged=true old_ready=refused \
        further_work_other_model=refused retained_envelope_bytes={cap} trusted_recovery=false"
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
async fn abort_prior_bits(
    client: &cubecl::client::Client,
    snapshot: &ResidentSnapshot,
    expected: &Domains,
) -> Result<()> {
    if snapshot.domains.len() != 42 || expected.len() != 42 {
        return Err(Error::State(
            "greedy abort full prior-state coverage".into(),
        ));
    }
    let mut expected_by_slot = std::collections::BTreeMap::new();
    for (slot, values) in expected {
        if expected_by_slot.insert(*slot, values.as_slice()).is_some() {
            return Err(Error::State(
                "greedy abort duplicate expected prior-state slot".into(),
            ));
        }
    }
    // CPU domains follow layer order; physical domains follow StateSlot order.
    // Remove each unique original slot so neither ordering can hide omissions.
    for domain in &snapshot.domains {
        let expected = expected_by_slot.remove(&domain.slot).ok_or_else(|| {
            Error::State("greedy abort missing/duplicate physical prior-state slot".into())
        })?;
        let raw = client
            .read_async(domain.leases.iter().map(ResidentLease::handle).collect())
            .await
            .map_err(|error| Error::State(format!("greedy abort prior physical read: {error}")))?;
        if raw.len() != domain.leases.len()
            || raw
                .iter()
                .zip(&domain.leases)
                .any(|(bytes, lease)| bytes.len() != lease.bytes())
        {
            return Err(Error::State(
                "greedy abort prior physical byte coverage".into(),
            ));
        }
        let mut actual = Vec::new();
        for (lease, bytes) in domain.leases.iter().zip(raw) {
            let values = words(&bytes)?;
            if matches!(&domain.buffers, StateBuffers::Kv { .. }) {
                let active =
                    lease.bytes() / 4 / snapshot.capacity * snapshot.binding.frontier() as usize;
                if values[active..].iter().any(|value| value.to_bits() != 0) {
                    return Err(Error::State(
                        "greedy abort changed immutable prior KV tail".into(),
                    ));
                }
                actual.extend_from_slice(&values[..active]);
            } else {
                actual.extend(values);
            }
        }
        exact(expected, &actual, "greedy aborted feedback immutable prior")?;
    }
    if !expected_by_slot.is_empty() {
        return Err(Error::State(
            "greedy abort unmatched prior-state slots".into(),
        ));
    }
    Ok(())
}
