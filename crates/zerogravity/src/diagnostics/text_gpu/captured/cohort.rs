//! Actual captured 4-row cohort eligibility and original prepare/commit only.
use super::*;

pub(super) fn eligible(row: &Row) -> bool {
    match &row.state {
        State::Qwen { model, session, .. } => {
            row.prompt.len() == 47
                && ((row.cursor == 0 && model.resident().initial_cohort_width() == 4)
                    || (row.cursor != 0 && session.cached_cohort_eligible()))
        }
        _ => false,
    }
}
pub(super) fn compatible(first: &Row, row: &Row) -> bool {
    let (
        State::Qwen {
            model: a,
            session: sa,
            ..
        },
        State::Qwen {
            model: b,
            session: sb,
            ..
        },
    ) = (&first.state, &row.state)
    else {
        return false;
    };
    if !Arc::ptr_eq(a, b)
        || first.model.cohort_width != 4
        || row.model.cohort_width != 4
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
pub(super) fn prepare(rows: &mut [Row]) -> Result<Preparation<TextPendingCohort>, ServiceError> {
    let prefill = rows[0].cursor == 0;
    let State::Qwen { model, .. } = &rows[0].state else {
        unreachable!("validated source cohort")
    };
    let model = model.clone();
    let cancellation = rows.iter().map(|r| r.cancelled.clone()).collect::<Vec<_>>();
    let cancelled = cancellation.iter().map(Arc::as_ref).collect::<Vec<_>>();
    let sessions = rows
        .iter()
        .map(|r| match &r.state {
            State::Qwen { session, .. } => session,
            _ => unreachable!("validated source cohort"),
        })
        .collect::<Vec<_>>();
    let prepared = if prefill {
        block_on(model.prepare_fixed_prompt_cohort_status(
            &sessions,
            &rows[0].prompt,
            &cancelled,
            None,
        ))
    } else {
        block_on(model.prepare_feedback_cohort_status(&sessions, &cancelled, None))
    }
    .map_err(ServiceError::inference)?;
    use crate::gpu::cuda::native::TextCohortPreparation;
    let ready = match prepared {
        TextCohortPreparation::Ready(ready) => ready,
        TextCohortPreparation::Capacity => return Ok(Preparation::Capacity),
        TextCohortPreparation::CancelledBeforeLaunch => {
            return Ok(Preparation::CancelledBeforeLaunch);
        }
    };
    let completion = Completion {
        quanta: 4,
        forwards: 1,
        dispatches: ready.dispatches() as u64,
        model_tokens: if prefill { 188 } else { 4 },
        committed_roots: 0,
        fused_projections: ready.native_projection_launches() as u64,
        fused_sampler: ready.sampler_dispatches() as u64,
        cohort_width: 4,
        state_only: false,
    };
    Ok(Preparation::Ready(Completed {
        ready,
        completion,
        fused: true,
    }))
}
pub(super) fn commit(
    ready: TextPendingCohort,
    rows: &mut [Row],
) -> Result<Publication, ServiceError> {
    let prefill = rows[0].cursor == 0;
    let cancellation = rows.iter().map(|r| r.cancelled.clone()).collect::<Vec<_>>();
    let cancelled = cancellation.iter().map(Arc::as_ref).collect::<Vec<_>>();
    let mut sessions = rows
        .iter_mut()
        .map(|r| match &mut r.state {
            State::Qwen { session, .. } => session.engine_mut(),
            _ => unreachable!("validated source cohort"),
        })
        .collect::<Vec<_>>();
    let published = ready.commit(&mut sessions, &cancelled);
    let committed = Instant::now();
    drop(sessions);
    let Some(receipt) = published.map_err(ServiceError::inference)? else {
        return Ok(Publication::Discarded);
    };
    if receipt.rows.len() != rows.len() {
        return Err(ServiceError::new(
            500,
            "cohort_receipt_count",
            "actual committed source row count differs",
        ));
    }
    let reports = rows
        .iter_mut()
        .zip(receipt.rows)
        .map(|(row, result)| {
            let State::Qwen {
                session, selected, ..
            } = &mut row.state
            else {
                unreachable!("validated source cohort")
            };
            if result.root != session.binding()
                || result.consumed_tokens != if prefill { 47 } else { 1 }
            {
                return Err(ServiceError::new(
                    500,
                    "bf16_cohort_receipt",
                    "actual committed source row differs",
                ));
            }
            *selected = Some(result.selected_id);
            Ok(Published {
                consumed: result.consumed_tokens,
                readout: true,
                observed_at: committed,
                observation: Some(GpuTokenObservation {
                    producer_frontier: result.root.frontier(),
                    producer_root: Some(GpuRootObservation::committed(result.root)),
                    fused_cohort: None,
                }),
            })
        })
        .collect();
    Ok(Publication::Committed(reports))
}
