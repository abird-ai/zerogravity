//! Batch submission uses the original ResidentModel and all-row publisher.
use super::*;

pub(super) fn finish_prompt_cohort(
    members: Vec<GpuActive>,
    active: &mut VecDeque<GpuActive>,
    stats: &WaveStats,
    native_rows: usize,
) {
    regroup(members, active, stats, Phase::Prefill, native_rows);
}

pub(super) fn advance_decode_cohort(
    members: Vec<GpuActive>,
    active: &mut VecDeque<GpuActive>,
    stats: &WaveStats,
    native_rows: usize,
) {
    regroup(members, active, stats, Phase::Decode, native_rows);
}

fn regroup(
    members: Vec<GpuActive>,
    active: &mut VecDeque<GpuActive>,
    stats: &WaveStats,
    phase: Phase,
    native_rows: usize,
) {
    // EOS, limits, cancellation and output credit can remove any wave row.
    // Repartition eligible survivors, never pad a native batch with a stop.
    let eligible = members
        .into_iter()
        .filter_map(|row| eligible_member(row, active, stats))
        .collect::<Vec<_>>();
    if native_rows == 0 {
        if phase == Phase::Prefill && eligible.len() > 1 {
            execute(eligible, active, stats, phase, 0);
        } else {
            for row in eligible {
                advance_member(row, active, stats);
            }
        }
        return;
    }
    let bands = match gpu_batch::native_bands(eligible.len(), native_rows) {
        Ok(bands) => bands,
        Err(error) => {
            for row in eligible {
                stats.fail(&row, error.clone());
            }
            return;
        }
    };
    let mut remaining = eligible.into_iter();
    for width in bands {
        if width == 1 {
            advance_member(remaining.next().expect("planned native row"), active, stats);
        } else {
            let members = remaining.by_ref().take(width).collect();
            execute(members, active, stats, phase, width);
        }
    }
}

fn execute(
    mut members: Vec<GpuActive>,
    active: &mut VecDeque<GpuActive>,
    stats: &WaveStats,
    phase: Phase,
    native_rows: usize,
) {
    if members
        .iter()
        .any(|row| row.cancelled.load(Ordering::Acquire))
    {
        regroup(members, active, stats, phase, native_rows);
        return;
    }
    let Some(first) = members.first() else { return };
    let model = Arc::clone(&first.resident);
    let roots = members
        .iter()
        .map(|row| row.session.binding())
        .collect::<Vec<_>>();
    let valid = ResidentBatchRows::new(&roots, first.hidden_width).and_then(|batch| {
        if members.iter().any(|row| {
            !Arc::ptr_eq(&row.resident, &model)
                || row.hidden_width != batch.row_elements()
                || (phase == Phase::Prefill
                    && (row.cursor + 1 != row.prompt.len() || row.device_token.is_some()))
                || (phase == Phase::Decode
                    && (row.cursor != row.prompt.len()
                        || !row.packet_presented
                        || row.device_token.is_none()
                        || row.blocked_since.is_some()))
        }) || (native_rows != 0
            && (!matches!(native_rows, 2 | 4 | 8) || members.len() != native_rows))
        {
            return Err(zerogravity::Error::State(
                "incompatible original GPU cohort".into(),
            ));
        }
        batch.require_current(&roots)
    });
    if let Err(error) = valid {
        let error = ServiceError::inference(error);
        for row in members {
            stats.fail(&row, error.clone());
        }
        return;
    }
    stats
        .attempted
        .fetch_add(members.len() as u64, Ordering::Relaxed);
    let prepared = if phase == Phase::Decode {
        let inputs = members
            .iter()
            .map(|row| {
                (
                    &row.session,
                    row.device_token.as_ref().expect("validated packet"),
                )
            })
            .collect::<Vec<_>>();
        block_on(model.prepare_device_tokens_fused_cohort_admitted(&inputs))
    } else {
        let inputs = members
            .iter()
            .map(|row| (&row.session, row.prompt[row.cursor]))
            .collect::<Vec<_>>();
        if native_rows != 0 {
            block_on(model.prepare_greedy_fused_cohort_admitted(&inputs))
        } else {
            block_on(model.prepare_greedy_cohort(&inputs)).map(Some)
        }
    };
    let cohort = match prepared {
        Ok(Some(value)) => value,
        Ok(None) => {
            stats
                .native_admission_refusals
                .fetch_add(1, Ordering::Relaxed);
            // The admitted API guarantees no launch or root/packet mutation.
            // Each capacity retry strictly reduces the attempted native band.
            regroup(members, active, stats, phase, native_rows / 2);
            return;
        }
        Err(error) => {
            let error = ServiceError::inference(error);
            for row in members {
                stats.fail(&row, error.clone());
            }
            return;
        }
    };
    let native = cohort.fused_dispatches() != 0 && cohort.native_greedy_dispatches() != 0;
    record_completion(stats, &cohort, native);
    if native_rows != 0 && !native {
        drop(cohort);
        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
        for row in members {
            stats.fail(
                &row,
                ServiceError::new(
                    500,
                    "missing_native_receipt",
                    "GPU cohort lacks native model/sampler receipts",
                ),
            );
        }
        return;
    }
    #[cfg(test)]
    {
        let hook = AFTER_COHORT_READY.with(|slot| slot.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }
    if members
        .iter()
        .any(|row| row.cancelled.load(Ordering::Acquire))
    {
        // Known completion: discard private results, preserve original roots
        // and packet-presented marks, then regroup only eligible survivors.
        drop(cohort);
        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
        regroup(members, active, stats, phase, native_rows);
        return;
    }
    let mut originals = members
        .iter_mut()
        .map(|row| &mut row.session)
        .collect::<Vec<_>>();
    let published = cohort.commit(&mut originals);
    drop(originals);
    match published {
        Ok(receipts) => {
            let count = members.len() as u64;
            stats.cohort_publications.fetch_add(1, Ordering::Relaxed);
            stats.cohort_rows.fetch_add(count, Ordering::Relaxed);
            stats.committed_roots.fetch_add(count, Ordering::Relaxed);
            for (mut row, read) in members.into_iter().zip(receipts) {
                if phase == Phase::Prefill {
                    row.cursor += 1;
                }
                match row.install(read) {
                    Ok(()) => active.push_back(row),
                    Err(error) => stats.fail(&row, error),
                }
            }
        }
        Err(error) => {
            stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
            let error = ServiceError::inference(error);
            for row in members {
                stats.fail(&row, error.clone());
            }
        }
    }
}

fn record_completion(stats: &WaveStats, cohort: &PreparedResidentCohort, native: bool) {
    let count = cohort.len() as u64;
    stats.completed.fetch_add(count, Ordering::Relaxed);
    stats
        .gpu_forwards
        .fetch_add(if native { 1 } else { count }, Ordering::Relaxed);
    stats
        .kernel_dispatches
        .fetch_add(cohort.dispatches() as u64, Ordering::Relaxed);
    stats.model_tokens.fetch_add(count, Ordering::Relaxed);
    if native {
        stats.fused_batches.fetch_add(1, Ordering::Relaxed);
        stats.fused_rows.fetch_add(count, Ordering::Relaxed);
        stats
            .native_projections
            .fetch_add(cohort.fused_dispatches() as u64, Ordering::Relaxed);
        stats
            .native_sampler
            .fetch_add(cohort.native_greedy_dispatches() as u64, Ordering::Relaxed);
        if let Some(index) = match cohort.len() {
            2 => Some(0),
            4 => Some(1),
            8 => Some(2),
            _ => None,
        } {
            stats.native_bands[index].fetch_add(1, Ordering::Relaxed);
        }
    }
}
