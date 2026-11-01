//! Scheduler-owned membership, band reduction and retry order. Physical Ready
//! and all-row publication remain opaque executor operations.
use super::*;

pub(super) fn advance<A: Executor>(
    rows: Vec<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: Cohorts,
    ceiling: usize,
) {
    match policy {
        Cohorts::None => singles(rows, active, stats),
        Cohorts::PromptTails => {
            let batch = rows.len() > 1
                && rows.iter().all(|r| {
                    !r.cancelled.load(Ordering::Acquire)
                        && r.cursor < r.prompt.len()
                        && r.cursor + 1 == r.prompt.len()
                        && !A::has_feedback(&r.state)
                });
            if batch {
                regroup(rows, active, stats, policy, Phase::Prefill, 0);
            } else {
                singles(rows, active, stats);
            }
        }
        Cohorts::Native {
            split_prefix: true, ..
        } => {
            let prefill = rows.len() > 1
                && rows.iter().all(|r| {
                    !r.cancelled.load(Ordering::Acquire)
                        && r.cursor < r.prompt.len()
                        && !A::has_feedback(&r.state)
                });
            let decode = rows.len() > 1 && rows.iter().all(|r| r.cursor == r.prompt.len());
            if prefill || decode {
                regroup(
                    rows,
                    active,
                    stats,
                    policy,
                    if prefill {
                        Phase::Prefill
                    } else {
                        Phase::Decode
                    },
                    ceiling,
                );
            } else {
                singles(rows, active, stats);
            }
        }
        Cohorts::Native {
            split_prefix: false,
            ..
        } => {
            let mut groups: Vec<Vec<Active<A>>> = Vec::new();
            for row in rows
                .into_iter()
                .filter_map(|r| eligible(r, active, stats))
                .collect::<Vec<_>>()
            {
                if !A::cohort_eligible(&row) {
                    advance_single(row, active, stats);
                    continue;
                }
                if let Some(group) = groups.iter_mut().find(|g| A::compatible(&g[0], &row)) {
                    group.push(row);
                } else {
                    groups.push(vec![row]);
                }
            }
            for group in groups {
                let phase = if group[0].cursor < group[0].prompt.len() {
                    Phase::Prefill
                } else {
                    Phase::Decode
                };
                bands(group, active, stats, policy, phase, ceiling, false);
            }
        }
    }
}
fn singles<A: Executor>(rows: Vec<Active<A>>, active: &mut VecDeque<Active<A>>, stats: &WorkStats) {
    for row in rows {
        advance_single(row, active, stats);
    }
}
fn regroup<A: Executor>(
    rows: Vec<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: Cohorts,
    phase: Phase,
    ceiling: usize,
) {
    let eligible = rows
        .into_iter()
        .filter_map(|r| eligible(r, active, stats))
        .collect::<Vec<_>>();
    if matches!(policy, Cohorts::PromptTails) {
        if phase == Phase::Prefill && eligible.len() > 1 {
            execute(
                eligible,
                active,
                stats,
                policy,
                CohortWork {
                    phase,
                    state_only: false,
                    native_rows: 0,
                },
            );
        } else {
            singles(eligible, active, stats);
        }
    } else if phase == Phase::Prefill
        && matches!(
            policy,
            Cohorts::Native {
                split_prefix: true,
                ..
            }
        )
    {
        let (prefix, tail) = eligible
            .into_iter()
            .partition(|r: &Active<A>| r.prompt.len().saturating_sub(r.cursor) > 1);
        bands(prefix, active, stats, policy, phase, ceiling, true);
        bands(tail, active, stats, policy, phase, ceiling, false);
    } else {
        bands(eligible, active, stats, policy, phase, ceiling, false);
    }
}
fn bands<A: Executor>(
    rows: Vec<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: Cohorts,
    phase: Phase,
    ceiling: usize,
    state_only: bool,
) {
    let Cohorts::Native { widths, .. } = policy else {
        unreachable!("native band policy")
    };
    let widths = widths
        .iter()
        .copied()
        .filter(|&n| n <= ceiling)
        .collect::<Vec<_>>();
    let planned = match waves::qualified_bands(rows.len(), &widths) {
        Ok(widths) => widths,
        Err(error) => {
            for row in rows {
                fail(&row, stats, error.clone());
            }
            return;
        }
    };
    let mut remaining = rows.into_iter();
    for width in planned {
        if width == 1 {
            advance_single(remaining.next().expect("planned singleton"), active, stats);
        } else {
            execute(
                remaining.by_ref().take(width).collect(),
                active,
                stats,
                policy,
                CohortWork {
                    phase,
                    state_only,
                    native_rows: width,
                },
            );
        }
    }
}
fn next_round<A: Executor>(
    rows: Vec<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
) {
    for row in rows {
        if let Some(row) = eligible(row, active, stats) {
            active.push_back(row);
        }
    }
}
fn retry_cancel<A: Executor>(
    rows: Vec<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: Cohorts,
    work: CohortWork,
) {
    if matches!(
        policy,
        Cohorts::Native {
            retry: RetryOrder::NextRound,
            ..
        }
    ) {
        next_round(rows, active, stats);
    } else if rows.iter().any(|row| row.cancelled.load(Ordering::Acquire)) {
        regroup(rows, active, stats, policy, work.phase, work.native_rows);
    } else {
        // Immediate retry must remove a member. Capacity uses a smaller ceiling
        // separately; a malformed executor cannot grow an unbounded retry stack.
        for row in rows {
            fail(
                &row,
                stats,
                ServiceError::new(
                    500,
                    "cohort_retry",
                    "discarded cohort has no cancelled member to remove",
                ),
            );
        }
    }
}
fn execute<A: Executor>(
    mut rows: Vec<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: Cohorts,
    work: CohortWork,
) {
    if rows.is_empty() {
        return;
    }
    if rows.iter().any(|r| r.cancelled.load(Ordering::Acquire)) {
        retry_cancel(rows, active, stats, policy, work);
        return;
    }
    if let Err(error) = A::validate_cohort(&rows, work) {
        for row in rows {
            fail(&row, stats, error.clone());
        }
        return;
    }
    stats
        .attempted
        .fetch_add(rows.len() as u64, Ordering::Relaxed);
    let completed = match A::prepare(&mut rows, work) {
        Ok(Preparation::Ready(ready)) => ready,
        Ok(Preparation::Capacity) => {
            stats
                .native_admission_refusals
                .fetch_add(1, Ordering::Relaxed);
            if matches!(
                policy,
                Cohorts::Native {
                    retry: RetryOrder::NextRound,
                    ..
                }
            ) {
                singles(rows, active, stats);
            } else if work.native_rows > 1 {
                regroup(
                    rows,
                    active,
                    stats,
                    policy,
                    work.phase,
                    work.native_rows / 2,
                );
            } else {
                for row in rows {
                    fail(
                        &row,
                        stats,
                        ServiceError::new(
                            503,
                            "cohort_capacity",
                            "composite cohort has no smaller admitted band",
                        ),
                    );
                }
            }
            return;
        }
        Ok(Preparation::CancelledBeforeLaunch) => {
            retry_cancel(rows, active, stats, policy, work);
            return;
        }
        Err(error) => {
            for row in rows {
                fail(&row, stats, error.clone());
            }
            return;
        }
    };
    stats
        .completed
        .fetch_add(completed.completion.quanta, Ordering::Relaxed);
    stats.record(completed.completion);
    let native = completed.native;
    if work.native_rows != 0 && !native {
        drop(completed);
        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
        for row in rows {
            fail(
                &row,
                stats,
                ServiceError::new(
                    500,
                    "missing_native_receipt",
                    "cohort lacks actual native model/readout receipts",
                ),
            );
        }
        return;
    }
    // Composite/native GGUF Ready is known-complete before this cancellation
    // boundary. Other executors retain their original cancellation-aware commit.
    if !matches!(
        policy,
        Cohorts::Native {
            retry: RetryOrder::NextRound,
            ..
        }
    ) && rows.iter().any(|r| r.cancelled.load(Ordering::Acquire))
    {
        drop(completed);
        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
        retry_cancel(rows, active, stats, policy, work);
        return;
    }
    match A::commit(completed.ready, &mut rows, work) {
        Ok(Publication::Committed(receipts)) => {
            let count = rows.len() as u64;
            let sequence = stats
                .cohort_publications
                .fetch_add(1, Ordering::Relaxed)
                .checked_add(1);
            let observed = if native {
                sequence.map(|publication_sequence| NativeCohortObservation {
                    publication_sequence,
                    width: rows.len(),
                })
            } else {
                None
            };
            stats.cohort_rows.fetch_add(count, Ordering::Relaxed);
            stats.committed_roots.fetch_add(count, Ordering::Relaxed);
            if receipts.len() != rows.len() {
                for row in rows {
                    fail(
                        &row,
                        stats,
                        ServiceError::new(
                            500,
                            "cohort_receipt_count",
                            "published cohort receipt count differs from original rows",
                        ),
                    );
                }
                return;
            }
            for (mut row, receipt) in rows.into_iter().zip(receipts) {
                match receipt.and_then(|published| row.install(published, observed)) {
                    Ok(()) => active.push_back(row),
                    Err(error) => fail(&row, stats, error),
                }
            }
        }
        Ok(Publication::Discarded) => {
            stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
            retry_cancel(rows, active, stats, policy, work);
        }
        Err(error) => {
            stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
            for row in rows {
                fail(&row, stats, error.clone());
            }
        }
    }
}
