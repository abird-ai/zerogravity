//! Original resident prepare/Ready/all-row publisher. No grouping or fallback.
use super::*;
use zerogravity_engine::router::waves::Phase;

pub(super) fn validate_wave(rows: &[&Row]) -> Result<(), ServiceError> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    let roots = rows
        .iter()
        .map(|r| r.state.session.binding())
        .collect::<Vec<_>>();
    let batch = ResidentBatchRows::new(&roots, first.model.hidden_width)
        .map_err(ServiceError::inference)?;
    if rows
        .iter()
        .any(|r| r.model.hidden_width != batch.row_elements())
    {
        return Err(ServiceError::inference(crate::Error::State(
            "GPU model wave hidden widths differ".into(),
        )));
    }
    batch
        .require_current(&roots)
        .map_err(ServiceError::inference)
}
pub(super) fn validate(rows: &[Row], work: CohortWork) -> Result<(), ServiceError> {
    let Some(first) = rows.first() else {
        return Err(ServiceError::new(
            500,
            "empty_cohort",
            "original cohort has no rows",
        ));
    };
    let model = &first.model.resident;
    let roots = rows
        .iter()
        .map(|r| r.state.session.binding())
        .collect::<Vec<_>>();
    ResidentBatchRows::new(&roots, first.model.hidden_width)
        .and_then(|batch| {
            if rows.iter().any(|r| {
                !Arc::ptr_eq(&r.model.resident, model)
                    || r.model.hidden_width != batch.row_elements()
                    || (work.phase == Phase::Prefill
                        && (r.cursor >= r.prompt.len()
                            || r.prefill_has_successor() != work.state_only
                            || r.prefill_count() != first.prefill_count()
                            || r.state.device_token.is_some()))
                    || (work.phase == Phase::Decode
                        && (r.cursor != r.prompt.len()
                            || !r.packet_presented
                            || r.state.device_token.is_none()
                            || r.blocked_since.is_some()))
            }) || (work.state_only && (work.phase != Phase::Prefill || work.cohort_width == 0))
                || (work.cohort_width != 0
                    && (!matches!(work.cohort_width, 2 | 4 | 8) || rows.len() != work.cohort_width))
            {
                return Err(crate::Error::State(
                    "incompatible original GPU cohort".into(),
                ));
            }
            batch.require_current(&roots)
        })
        .map_err(ServiceError::inference)
}
pub(super) fn prepare(
    rows: &mut [Row],
    work: CohortWork,
) -> Result<Preparation<PreparedResidentCohort>, ServiceError> {
    let model = rows[0].model.resident.clone();
    let prepared = if work.phase == Phase::Decode {
        let inputs = rows
            .iter()
            .map(|r| {
                (
                    &r.state.session,
                    r.state.device_token.as_ref().expect("validated packet"),
                )
            })
            .collect::<Vec<_>>();
        block_on(model.prepare_device_tokens_fused_cohort_admitted(&inputs))
    } else {
        let tokens = rows[0].prefill_count();
        if tokens > 1 {
            let inputs = rows
                .iter()
                .map(|r| (&r.state.session, &r.prompt[r.cursor..r.cursor + tokens]))
                .collect::<Vec<_>>();
            if work.state_only {
                block_on(model.prepare_state_temporal_fused_cohort_admitted(&inputs))
            } else if work.cohort_width != 0 {
                block_on(model.prepare_greedy_temporal_fused_cohort_admitted(&inputs))
            } else {
                Err(crate::Error::State(
                    "non-native prompt tail cannot request temporal cohort".into(),
                ))
                .map(Some)
            }
        } else {
            let inputs = rows
                .iter()
                .map(|r| (&r.state.session, r.prompt[r.cursor]))
                .collect::<Vec<_>>();
            if work.state_only {
                block_on(model.prepare_state_fused_cohort_admitted(&inputs))
            } else if work.cohort_width != 0 {
                block_on(model.prepare_greedy_fused_cohort_admitted(&inputs))
            } else {
                block_on(model.prepare_greedy_cohort(&inputs)).map(Some)
            }
        }
    }
    .map_err(ServiceError::inference)?;
    let Some(ready) = prepared else {
        return Ok(Preparation::Capacity);
    };
    let fused = ready.fused_dispatches() != 0
        && if work.state_only {
            ready.native_greedy_dispatches() == 0
        } else {
            ready.native_greedy_dispatches() != 0
        };
    let tokens = if work.phase == Phase::Prefill {
        rows[0].prefill_count()
    } else {
        1
    };
    let completion = Completion {
        quanta: ready.len() as u64,
        forwards: if fused {
            tokens as u64
        } else {
            ready.len() as u64
        },
        dispatches: ready.dispatches() as u64,
        model_tokens: (ready.len() * tokens) as u64,
        committed_roots: 0,
        fused_projections: if fused {
            ready.fused_dispatches() as u64
        } else {
            0
        },
        fused_sampler: if fused {
            ready.native_greedy_dispatches() as u64
        } else {
            0
        },
        cohort_width: if fused { ready.len() } else { 0 },
        state_only: work.state_only,
    };
    #[cfg(test)]
    if work.cohort_width == 0 || fused {
        let hook = AFTER_COHORT_READY.with(|slot| slot.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }
    Ok(Preparation::Ready(Completed {
        ready,
        completion,
        fused,
    }))
}
pub(super) fn commit(
    ready: PreparedResidentCohort,
    rows: &mut [Row],
    work: CohortWork,
) -> Result<Publication, ServiceError> {
    let mut originals = rows
        .iter_mut()
        .map(|r| &mut r.state.session)
        .collect::<Vec<_>>();
    let published = ready.commit(&mut originals);
    let committed = Instant::now();
    drop(originals);
    let receipts = published.map_err(ServiceError::inference)?;
    if receipts.len() != rows.len() {
        return Err(ServiceError::new(
            500,
            "cohort_receipt_count",
            "original publisher returned a different row count",
        ));
    }
    let consumed = if work.phase == Phase::Prefill {
        rows[0].prefill_count()
    } else {
        1
    };
    let reports = rows
        .iter_mut()
        .zip(receipts)
        .map(|(row, read)| {
            if work.state_only {
                if read.state_only_steps != consumed
                    || read.logit_steps != 0
                    || !read.logits.is_empty()
                    || read.device_token.is_some()
                {
                    Err(ServiceError::new(
                        500,
                        "invalid_prefix_readout",
                        "state-only GPU completion produced a generated readout",
                    ))
                } else {
                    Ok(Published {
                        consumed,
                        readout: false,
                        observed_at: committed,
                        observation: None,
                    })
                }
            } else if work.phase == Phase::Prefill
                && (read.state_only_steps + read.logit_steps != consumed || read.logit_steps != 1)
            {
                Err(ServiceError::new(
                    500,
                    "invalid_temporal_readout",
                    "terminal temporal GPU completion has wrong step coverage",
                ))
            } else {
                install(row, read, consumed, committed)
            }
        })
        .collect();
    Ok(Publication::Committed(reports))
}
