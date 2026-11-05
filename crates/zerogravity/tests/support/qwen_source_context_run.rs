//! Full original trajectories; archived boundaries and digest-only logits stay distinct.
use super::*;
use reference::{ObservationMap, Observer};
use zerogravity::{
    accelerator::program::SourceCallMode,
    gpu::{ResidentOutput, SourceCallModel, SourceCallPreparation, SourceCallPublication},
    qwen35::plan::OrdinaryQwenPlan,
};

pub async fn trajectories(
    model: &Arc<SourceCallModel>,
    plan: &OrdinaryQwenPlan,
    expected: &Expected,
    case: usize,
    capacity: usize,
) -> Result<()> {
    let map = ObservationMap::new(plan)?;
    let mut sessions = [model.session().await?, model.session().await?];
    need(
        sessions[0].binding().session() != sessions[1].binding().session(),
        "independent original roots",
    )?;
    let count = expected
        .calls()
        .iter()
        .filter(|c| c.case == case && c.repeat == 0)
        .count();
    let mut feedback = [None, None];
    let (mut calls, mut hashes, mut words, mut tails, mut checkpoints) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for index in 0..count {
        for repeat in 0..2 {
            let call = expected
                .calls()
                .iter()
                .find(|c| c.key() == (repeat, case, index))
                .ok_or_else(|| zerogravity::Error::State("original context call absent".into()))?;
            let peer = sessions[1 - repeat].binding();
            let session = &mut sessions[repeat];
            let before = session.binding();
            need(
                before.frontier() == call.position as u64,
                "actual populated prior frontier",
            )?;
            let (mode, ids) = if call.phase == "decode" {
                (
                    SourceCallMode::Decode,
                    vec![feedback[repeat].ok_or_else(|| {
                        zerogravity::Error::State("actual context feedback absent".into())
                    })?],
                )
            } else {
                (SourceCallMode::Prefill, call.input_ids.clone())
            };
            need(
                ids == call.input_ids,
                "actual feedback, not reference-fed decode",
            )?;
            let invocation = plan.bind_invocation(mode, call.position, ids.into(), true)?;
            // All earlier full forwards run, but only original captured boundaries are observed.
            let mut observer = (repeat == 0 && call.checkpoint)
                .then(|| Observer::new(expected, &map, call))
                .transpose()?;
            let pending = if let Some(observer) = &mut observer {
                session
                    .prepare_observed(
                        invocation.bound_invocation(),
                        mode,
                        ResidentOutput::Logits,
                        None,
                        observer,
                    )
                    .await?
            } else {
                session
                    .prepare(
                        invocation.bound_invocation(),
                        mode,
                        ResidentOutput::Logits,
                        None,
                    )
                    .await?
            };
            let SourceCallPreparation::Ready(pending) = pending else {
                return Err(zerogravity::Error::State(
                    "populated uncancelled call refused".into(),
                ));
            };
            need(session.binding() == before, "private Ready did not publish")?;
            if let Some(observer) = &observer {
                observer.finish()?;
            }
            let read = pending.readback();
            hashes += expected.compare_logit_digest(call, &read.logits)?;
            let (selected, maximum) = zerogravity::numerics::greedy_finite(&read.logits)?;
            need(
                (selected, maximum.to_bits()) == (call.selected, call.max_bits),
                "actual full-vocabulary context selection",
            )?;
            let SourceCallPublication::Committed(output) = session.commit(pending)? else {
                return Err(zerogravity::Error::State(
                    "populated uncancelled call discarded".into(),
                ));
            };
            need(
                session.binding().frontier() == (call.position + call.rows) as u64
                    && zerogravity::numerics::greedy_finite(&output.logits)?.0 == selected,
                "same original result and advancing root",
            )?;
            need(
                sessions[1 - repeat].binding() == peer,
                "peer context root unchanged",
            )?;
            feedback[repeat] = Some(selected);
            calls += 1;
            if call.checkpoint {
                let (float_words, tail_words, observed) = if let Some(observer) = observer {
                    (observer.words, observer.kv_tail_words, true)
                } else {
                    let logits = expected.compare_logits(call, &output.logits)?;
                    let snapshot = sessions[repeat].snapshot()?;
                    let domains = snapshot.read_domains().await?;
                    let (state, tail) = expected.compare_domains(call, &domains, capacity)?;
                    (logits + state, tail, false)
                };
                checkpoints += 1;
                words += float_words;
                tails += tail_words;
                println!(
                    "SOURCE_CONTEXT_CHECKPOINT context={capacity} repeat={repeat} call={index} base={} rows={} selected={selected} observed={observed} float_words={float_words} kv_tail_words={tail_words} mismatches=0",
                    call.position, call.rows
                );
            }
            drop(output);
        }
    }
    if calls != count * 2 || checkpoints != 6 || hashes != calls * 248320 {
        return Err(zerogravity::Error::State(format!(
            "complete original trajectories and distinct checkpoint/digest coverage: calls={calls} expected_calls={} checkpoints={checkpoints} expected_checkpoints=6 digest_logit_words={hashes} expected_digest_logit_words={}",
            count * 2,
            calls * 248320
        )));
    }
    need(
        sessions
            .iter()
            .all(|s| s.binding().frontier() == capacity as u64),
        "actual fully populated roots",
    )?;
    expected.unchanged()?;
    println!(
        "SOURCE_CONTEXT_NUMERICS context={capacity} calls={calls} checkpoints={checkpoints} checkpoint_float_words={words} digest_logit_words={hashes} kv_tail_words={tails} mismatches=0 intermediate_state_comparisons=false sdk=false performance=false"
    );
    Ok(())
}
