//! Equal47 prefill and genuine N1/B4 cached rows on the original fair worker.
use super::*;

fn compatible(first: &GpuActive, row: &GpuActive) -> bool {
    let (
        RequestEngine::Qwen35Cuda {
            model: a,
            session: sa,
            ..
        },
        RequestEngine::Qwen35Cuda {
            model: b,
            session: sb,
            ..
        },
    ) = (&first.engine, &row.engine)
    else {
        return false;
    };
    if !Arc::ptr_eq(a, b)
        || first.native_rows != 4
        || row.native_rows != 4
        || row.prompt.len() != 47
    {
        return false;
    }
    match (first.cursor == 0, row.cursor == 0) {
        (true, true) => a.resident().initial_cohort_width() == 4 && first.prompt == row.prompt,
        (false, false) => sa.cached_cohort_eligible() && sb.cached_cohort_eligible(),
        _ => false,
    }
}
pub(in crate::serve::gpu_engine) fn advance_cohort(
    members: Vec<GpuActive>,
    active: &mut VecDeque<GpuActive>,
    stats: &WaveStats,
) {
    let mut groups: Vec<Vec<GpuActive>> = Vec::new();
    let eligible = members
        .into_iter()
        .filter_map(|row| eligible_member(row, active, stats))
        .collect::<Vec<_>>();
    for row in eligible {
        let eligible = match &row.engine {
            RequestEngine::Qwen35Cuda { model, session, .. } => {
                row.prompt.len() == 47
                    && ((row.cursor == 0 && model.resident().initial_cohort_width() == 4)
                        || (row.cursor != 0 && session.cached_cohort_eligible()))
            }
            _ => false,
        };
        if !eligible {
            advance_member(row, active, stats);
            continue;
        }
        if let Some(group) = groups.iter_mut().find(|group| compatible(&group[0], &row)) {
            group.push(row);
        } else {
            groups.push(vec![row]);
        }
    }
    for group in groups {
        // Width8 remains unavailable until its separate full original capture,
        // lowering and real-root numerical qualification. No inherited B2.
        let bands = match gpu_batch::bf16_bands(group.len(), 4) {
            Ok(bands) => bands,
            Err(error) => {
                for row in group {
                    stats.fail(&row, error.clone());
                }
                continue;
            }
        };
        let mut remaining = group.into_iter();
        for width in bands {
            if width == 1 {
                advance_member(remaining.next().expect("source BF16 band"), active, stats);
            } else {
                execute(remaining.by_ref().take(width).collect(), active, stats);
            }
        }
    }
}
fn execute(mut members: Vec<GpuActive>, active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
    if members.len() != 4 || members.iter().any(|row| !compatible(&members[0], row)) {
        for row in members {
            stats.fail(
                &row,
                ServiceError::new(
                    500,
                    "bf16_cohort_membership",
                    "source B4 eligibility changed",
                ),
            );
        }
        return;
    }
    if members
        .iter()
        .any(|row| row.cancelled.load(Ordering::Acquire))
    {
        // Roots were untouched; survivors re-enter the next original fair round.
        for row in members {
            if let Some(row) = eligible_member(row, active, stats) {
                active.push_back(row);
            }
        }
        return;
    }
    let prefill = members[0].cursor == 0;
    let RequestEngine::Qwen35Cuda { model, .. } = &members[0].engine else {
        unreachable!("validated source cohort")
    };
    let model = model.clone();
    let cancellation = members
        .iter()
        .map(|row| row.cancelled.clone())
        .collect::<Vec<_>>();
    let cancelled = cancellation.iter().map(Arc::as_ref).collect::<Vec<_>>();
    stats.attempted.fetch_add(4, Ordering::Relaxed);
    let prepared = {
        let sessions = members
            .iter()
            .map(|row| match &row.engine {
                RequestEngine::Qwen35Cuda { session, .. } => session,
                _ => unreachable!("validated source cohort"),
            })
            .collect::<Vec<_>>();
        if prefill {
            block_on(model.prepare_fixed_prompt_cohort_status(
                &sessions,
                &members[0].prompt,
                &cancelled,
                None,
            ))
        } else {
            block_on(model.prepare_feedback_cohort_status(&sessions, &cancelled, None))
        }
    };
    let pending = match prepared {
        Ok(zerogravity::gpu::cuda::native::TextCohortPreparation::Ready(pending)) => pending,
        Ok(zerogravity::gpu::cuda::native::TextCohortPreparation::CancelledBeforeLaunch) => {
            for row in members {
                if let Some(row) = eligible_member(row, active, stats) {
                    active.push_back(row);
                }
            }
            return;
        }
        Ok(zerogravity::gpu::cuda::native::TextCohortPreparation::Capacity) => {
            // Only the admitted API's proved prelaunch capacity refusal permits
            // width reduction. Source/numerical/stale/unknown errors are fatal.
            stats
                .native_admission_refusals
                .fetch_add(1, Ordering::Relaxed);
            for row in members {
                advance_member(row, active, stats);
            }
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
    stats.completed.fetch_add(4, Ordering::Relaxed);
    stats.gpu_forwards.fetch_add(1, Ordering::Relaxed);
    stats
        .kernel_dispatches
        .fetch_add(pending.dispatches() as u64, Ordering::Relaxed);
    stats.native_projections.fetch_add(
        pending.native_projection_launches() as u64,
        Ordering::Relaxed,
    );
    stats
        .native_sampler
        .fetch_add(pending.sampler_dispatches() as u64, Ordering::Relaxed);
    stats
        .model_tokens
        .fetch_add(if prefill { 188 } else { 4 }, Ordering::Relaxed);
    stats.fused_batches.fetch_add(1, Ordering::Relaxed);
    stats.fused_greedy_batches.fetch_add(1, Ordering::Relaxed);
    stats.native_bands[1].fetch_add(1, Ordering::Relaxed);
    stats.fused_rows.fetch_add(4, Ordering::Relaxed);
    let published = {
        let mut sessions = members
            .iter_mut()
            .map(|row| match &mut row.engine {
                RequestEngine::Qwen35Cuda { session, .. } => session.engine_mut(),
                _ => unreachable!("validated source cohort"),
            })
            .collect::<Vec<_>>();
        pending.commit(&mut sessions, &cancelled)
    };
    let committed_at = Instant::now();
    match published {
        Ok(Some(receipt)) => {
            let sequence = stats
                .cohort_publications
                .fetch_add(1, Ordering::Relaxed)
                .checked_add(1);
            let observed = sequence.map(|publication_sequence| NativeCohortObservation {
                publication_sequence,
                width: 4,
            });
            stats.cohort_rows.fetch_add(4, Ordering::Relaxed);
            stats.committed_roots.fetch_add(4, Ordering::Relaxed);
            for (mut row, result) in members.into_iter().zip(receipt.rows) {
                let RequestEngine::Qwen35Cuda {
                    session, selected, ..
                } = &mut row.engine
                else {
                    unreachable!("validated source cohort")
                };
                if result.root != session.binding()
                    || result.consumed_tokens != if prefill { 47 } else { 1 }
                {
                    stats.fail(
                        &row,
                        ServiceError::new(
                            500,
                            "bf16_cohort_receipt",
                            "actual committed source row differs",
                        ),
                    );
                    continue;
                }
                *selected = Some(result.selected_id);
                if prefill {
                    row.cursor = row.prompt.len();
                }
                row.token_gpu_observation = Some(GpuTokenObservation {
                    producer_frontier: result.root.frontier(),
                    producer_root: Some(GpuRootObservation::committed(result.root)),
                    native_cohort: observed,
                });
                row.token_commit_observed_at = Some(committed_at);
                row.packet_presented = false;
                row.blocked_since = None;
                active.push_back(row);
            }
        }
        Ok(None) => {
            stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
            // Known-completed discard cannot jump ahead of other ready waves.
            for row in members {
                if let Some(row) = eligible_member(row, active, stats) {
                    active.push_back(row);
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
