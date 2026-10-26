//! Native greedy/feedback proof over original source math and independent roots.
use super::*;

pub const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];
pub const VOCABULARY: usize = 248_320;
pub const SAMPLER_DISPATCHES: usize = 4;
pub type Domains = Vec<(StateSlot, Vec<f32>)>;

pub fn greedy(values: &[f32]) -> Result<(u32, u32)> {
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

pub fn roots(sessions: &[ResidentSession]) -> Vec<RootBinding> {
    sessions.iter().map(ResidentSession::binding).collect()
}

pub fn unchanged(sessions: &[ResidentSession], expected: &[RootBinding]) -> Result<()> {
    if roots(sessions) != expected {
        return Err(Error::State(
            "private/refused greedy cohort changed roots".into(),
        ));
    }
    Ok(())
}

pub async fn domains(sessions: &[ResidentSession]) -> Result<Vec<Domains>> {
    let mut values = Vec::with_capacity(sessions.len());
    for session in sessions {
        values.push(session.snapshot().read_domains().await?);
    }
    Ok(values)
}

pub async fn same_domains(sessions: &[ResidentSession], expected: &[Domains]) -> Result<()> {
    if sessions.len() != expected.len() {
        return Err(Error::State("greedy state row coverage".into()));
    }
    for (session, expected) in sessions.iter().zip(expected) {
        compare_domains(expected, &session.snapshot().read_domains().await?)?;
    }
    Ok(())
}

pub fn quiet(resident: &ResidentModel, bytes: usize) -> Result<()> {
    if resident.service.pending_events()? != 0 || resident.service.resident_bytes()? != bytes {
        return Err(Error::State(
            "greedy candidate/packet/event lease not reclaimed".into(),
        ));
    }
    Ok(())
}

pub async fn host(
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

pub async fn feedback(
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

pub fn private_receipt(
    candidate: &PreparedResidentCohort,
    count: usize,
    frontier: u64,
) -> Result<()> {
    if candidate.len() != count
        || candidate.prior_frontier() != Some(frontier)
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

pub fn packets(
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

pub fn publication(before: &[RootBinding], sessions: &[ResidentSession]) -> Result<()> {
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

pub async fn refusals(
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

/// Synthetic selector fixtures; expected invalid records never become tokens/roots.
pub async fn selector_fixtures(
    resident: &Arc<ResidentModel>,
    count: usize,
    eos: u32,
) -> Result<()> {
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

#[cfg(feature = "hardware-abort-probe")]
pub async fn abort_prior_bits(
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
