//! Deadline diagnosis over original state-only prefill; host intervals include readback.
use super::{Expected, Result, reference, source::need};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};
#[path = "qwen_source_context_timing/kernels.rs"]
mod kernels;
use zerogravity::{
    accelerator::program::{InvocationPhase, PlanOperation, SourceOperatorKind},
    gpu::{
        ResidentOutput, ResidentTextModel, ResidentTextPreparation, ResidentTextPublication,
        TextFrame, TextFrameKind, TextObserver,
    },
    qwen35::plan::QwenForwardPlan,
};

struct Timing<'a> {
    labels: &'a BTreeMap<u32, String>,
    original: Option<reference::Observer<'a>>,
    last: Instant,
    intervals: BTreeMap<String, (usize, u128)>,
    kernels: Option<kernels::Trace<'a>>,
}
impl TextObserver for Timing<'_> {
    fn wants(&self, kind: TextFrameKind) -> bool {
        matches!(kind, TextFrameKind::Value(_))
            || self.original.as_ref().is_some_and(|o| o.wants(kind))
    }
    fn completed(&mut self, frame: TextFrame, actual: &[u8]) -> Result<()> {
        let elapsed = self.last.elapsed().as_nanos();
        let label = match frame.kind {
            TextFrameKind::Value(id) => self
                .labels
                .get(&id.0)
                .ok_or_else(|| zerogravity::Error::State("timing original producer absent".into()))?
                .as_str(),
            TextFrameKind::State { .. } => "state_readback",
        };
        let slot = self.intervals.entry(label.into()).or_default();
        slot.0 += 1;
        slot.1 += elapsed;
        if let Some(original) = &mut self.original
            && original.wants(frame.kind)
        {
            original.completed(frame, actual)?;
        }
        if let Some(kernels) = &mut self.kernels
            && frame.offset_words + actual.len() / 4 == frame.total_words
        {
            kernels.completed(frame.kind)?;
        }
        // Exclude the preceding CPU reference comparison, not GPU/readback overhead.
        self.last = Instant::now();
        Ok(())
    }
}

pub async fn run(
    model: &Arc<ResidentTextModel>,
    plan: &QwenForwardPlan,
    expected: &Expected,
    case: usize,
    capacity: usize,
    measure_kernels: bool,
) -> Result<()> {
    let map = reference::ObservationMap::new(plan)?;
    let mut labels = BTreeMap::new();
    for node in plan.plan().nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(zerogravity::Error::State(
                "timing original template absent".into(),
            ));
        };
        let kind = format!("{:?}", descriptor.kind());
        let label = kind.split(['{', '(']).next().unwrap().trim();
        for id in &node.outputs {
            need(
                labels.insert(id.0, label.to_owned()).is_none(),
                "unique original timing producer",
            )?;
        }
    }
    let mut starts = BTreeSet::new();
    let mut ends = BTreeSet::new();
    if measure_kernels {
        let mut previous = None;
        for node in plan.plan().nodes() {
            let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
                unreachable!("validated templates")
            };
            if matches!(descriptor.kind(), SourceOperatorKind::Attention { .. }) {
                need(node.outputs.len() == 1, "one original attention output")?;
                let previous = previous.ok_or_else(|| {
                    zerogravity::Error::State("attention timing predecessor absent".into())
                })?;
                starts.insert(previous);
                ends.insert(node.outputs[0].0);
            }
            // Include outputless KV append before attention, named separately in timings.
            if let Some(last) = node.outputs.last() {
                previous = Some(last.0);
            }
        }
        need(
            !starts.is_empty(),
            "original attention timing windows present",
        )?;
    }
    let mut session = model.session().await?;
    let mut feedback = None;
    let (mut calls, mut checkpoints, mut words, mut tails, mut digests) = (0, 0, 0, 0, 0);
    let started = Instant::now();
    for call in expected
        .calls()
        .iter()
        .filter(|c| c.case == case && c.repeat == 0)
    {
        let readout = call.checkpoint;
        let mode = if call.phase == "decode" {
            InvocationPhase::Decode
        } else {
            InvocationPhase::Prefill
        };
        let ids =
            if mode == InvocationPhase::Decode {
                vec![feedback.ok_or_else(|| {
                    zerogravity::Error::State("timing actual feedback absent".into())
                })?]
            } else {
                call.input_ids.clone()
            };
        need(ids == call.input_ids, "timing actual input IDs")?;
        let before = session.binding();
        need(
            before.frontier() == call.position as u64,
            "timing original frontier",
        )?;
        let invocation = plan.bind_invocation(mode, call.position, ids.into(), readout)?;
        let sample = call.position == 0 || call.position == capacity / 2 || readout;
        let original = readout
            .then(|| reference::Observer::new(expected, &map, call))
            .transpose()?;
        let mut timing = sample.then(|| Timing {
            labels: &labels,
            original,
            last: Instant::now(),
            intervals: BTreeMap::new(),
            kernels: (measure_kernels && call.position == capacity / 2)
                .then(|| kernels::Trace::new(&starts, &ends)),
        });
        let step_started = Instant::now();
        let output = if readout {
            ResidentOutput::Logits
        } else {
            ResidentOutput::StateOnly
        };
        let ready = if let Some(timing) = &mut timing {
            session
                .prepare_observed(invocation.bound_invocation(), mode, output, None, timing)
                .await?
        } else {
            session
                .prepare(invocation.bound_invocation(), mode, output, None)
                .await?
        };
        let ResidentTextPreparation::Ready(ready) = ready else {
            return Err(zerogravity::Error::State(
                "uncancelled timing call refused".into(),
            ));
        };
        need(
            session.binding() == before,
            "timing private Ready did not publish",
        )?;
        let dispatches = ready.readback().dispatches;
        let selected = if readout {
            let observer = timing.as_ref().and_then(|t| t.original.as_ref()).unwrap();
            observer.finish()?;
            words += observer.words;
            tails += observer.kv_tail_words;
            checkpoints += 1;
            digests += expected.compare_logit_digest(call, &ready.readback().logits)?;
            let (selected, maximum) =
                zerogravity::numerics::greedy_finite(&ready.readback().logits)?;
            need(
                (selected, maximum.to_bits()) == (call.selected, call.max_bits),
                "timing actual full-vocabulary selection",
            )?;
            Some(selected)
        } else {
            need(
                ready.readback().logits.is_empty(),
                "timing state-only call has no readout",
            )?;
            None
        };
        let ResidentTextPublication::Committed(result) = session.commit(ready)? else {
            return Err(zerogravity::Error::State(
                "uncancelled timing call discarded".into(),
            ));
        };
        need(
            session.binding().frontier() == (call.position + call.rows) as u64,
            "timing original publication",
        )?;
        feedback = selected;
        calls += 1;
        if sample || call.position.is_multiple_of(1024) {
            println!(
                "SOURCE_CONTEXT_PROGRESS context={capacity} call={} base={} rows={} frontier={} readout={readout} dispatches={dispatches} step_ms={} elapsed_ms={} observed={sample} sdk=false performance=false",
                call.call,
                call.position,
                call.rows,
                session.binding().frontier(),
                step_started.elapsed().as_millis(),
                started.elapsed().as_millis()
            );
        }
        if let Some(timing) = timing {
            if let Some(kernels) = &timing.kernels {
                kernels.report(capacity, call.position, call.rows)?;
            }
            for (kind, (frames, ns)) in timing.intervals {
                println!(
                    "SOURCE_CONTEXT_OPERATOR_INTERVAL context={capacity} base={} rows={} kind={kind} frames={frames} host_ns={ns} includes_readback=true device_time=false performance=false",
                    call.position, call.rows
                );
            }
        }
        drop(result);
    }
    need(
        calls == (capacity - 2) / 64 + 3
            && checkpoints == 3
            && digests == 3 * 248320
            && session.binding().frontier() == capacity as u64,
        "timing complete ordinary trajectory and final checkpoint coverage",
    )?;
    expected.unchanged()?;
    println!(
        "SOURCE_CONTEXT_TIMING_DONE context={capacity} calls={calls} checkpoints={checkpoints} checkpoint_float_words={words} digest_logit_words={digests} kv_tail_words={tails} mismatches=0 elapsed_ms={} state_only_prefill=true sdk=false performance=false",
        started.elapsed().as_millis()
    );
    Ok(())
}
