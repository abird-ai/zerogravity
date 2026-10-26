//! Genuine N1 private roots; independent same-prior N1 reference is expected
//! only. No captured/reference tensor is imported into a production root.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use zerogravity::accelerator::program::{StateSlot, ValueId};

#[path = "continuous_diagnostic.rs"]
mod diagnostic;
#[path = "continuous_export.rs"]
mod export;

struct Expected {
    values: BTreeMap<u32, Vec<u8>>,
    sums: BTreeMap<usize, Vec<u8>>,
    states: BTreeMap<(StateSlot, bool), Vec<Vec<u8>>>,
    input: Vec<u32>,
    selected: (u32, u32),
}
fn sum_nodes(record: &zerogravity::gpu::SourceBindings) -> Result<Vec<usize>> {
    let mut nodes = (1..record.selection().boundaries().len())
        .map(|layer| compare::input_rms_node(record, layer))
        .collect::<Result<Vec<_>>>()?;
    let final_inputs = *record
        .selection()
        .boundaries()
        .last()
        .ok_or_else(|| fail("actual final boundary pair"))?;
    let final_node = compare::residual_rms_node(record, final_inputs)?;
    let normalized = record
        .selection()
        .normalized()
        .ok_or_else(|| fail("actual final normalization selector"))?;
    ensure(
        record.plan().nodes()[final_node].outputs.first() == Some(&normalized),
        "actual final normalization is the same boundary-pair RMS producer",
    )?;
    nodes.push(final_node);
    Ok(nodes)
}

impl Expected {
    fn capture(read: native::TextReadback<'_>, limit: usize) -> Result<Self> {
        let mut copied = 0usize;
        let mut copy = |raw: &[u8]| -> Result<Vec<u8>> {
            copied = copied
                .checked_add(raw.len())
                .ok_or_else(|| fail("expected payload overflow"))?;
            ensure(copied <= limit, "charged expected row bound BEFORE clone")?;
            Ok(raw.to_vec())
        };
        let record = read.record();
        ensure(
            record.call().rows().len() == 1,
            "independent N1 cached reference only",
        )?;
        let mut ids = record
            .selection()
            .boundaries()
            .iter()
            .flatten()
            .map(|v| v.0)
            .collect::<BTreeSet<_>>();
        ids.extend([
            325,
            record
                .selection()
                .normalized()
                .ok_or_else(|| fail("actual final normalization selector"))?
                .0,
            record.selection().output().0,
        ]);
        // Existing retained layer0 producers; expected bytes never enter execution.
        ids.extend(0..10);
        ensure(
            ids.len() <= 61,
            "bounded original boundary/final/fullV population",
        )?;
        let values = ids
            .into_iter()
            .map(|id| {
                let bytes = read.value(ValueId(id))?;
                ensure(
                    id >= 10 || bytes.len() <= 65536,
                    "bounded early producer extent",
                )?;
                Ok((id, copy(bytes)?))
            })
            .collect::<Result<_>>()?;
        let nodes = sum_nodes(record)?;
        ensure(nodes.len() == 24, "bounded full dual-sum population")?;
        let sums = nodes
            .into_iter()
            .map(|node| Ok((node, copy(read.rms_sum(node)?)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure(sums.len() == 24, "all24 actual independent N1 dual sums")?;
        let mut states = BTreeMap::new();
        ensure(
            record.plan().proposed_domains().len() == 42,
            "bounded original42 domains",
        )?;
        for version in record.plan().proposed_domains() {
            for candidate in [false, true] {
                let (a, b) = read.state(version.slot, candidate)?;
                let mut planes = vec![copy(a)?];
                if let Some(b) = b {
                    planes.push(copy(b)?);
                }
                ensure(
                    states.insert((version.slot, candidate), planes).is_none(),
                    "unique original state domain",
                )?;
            }
        }
        ensure(states.len() == 84, "all42 consumed and candidate domains")?;
        ensure(
            record.call().rows().ids().len() == 1,
            "one original cached teacher ID",
        )?;
        Ok(Self {
            values,
            sums,
            states,
            input: record.call().rows().ids().to_vec(),
            selected: read.greedy(),
        })
    }
    fn bytes(&self) -> usize {
        self.values.values().map(Vec::len).sum::<usize>()
            + self.sums.values().map(Vec::len).sum::<usize>()
            + self.states.values().flatten().map(Vec::len).sum::<usize>()
    }
    fn diagnose(&self, read: &native::TextCohortReadback<'_>, row: usize) -> Result<()> {
        let record = &read.records[row];
        ensure(
            record.call().rows().ids() == self.input,
            "same diagnostic producer token",
        )?;
        println!(
            "QWEN35_BATCHING_SAME_PRIOR_ROW row={row} span={:?} profile={:?} source={:?}",
            record.call().rows(),
            record.plan().numerical_profile(),
            record.call().source()
        );
        // Report every early producer and prior before the original exact failure.
        for id in 0..10 {
            let producer = record
                .plan()
                .nodes()
                .iter()
                .enumerate()
                .find(|(_, node)| node.outputs.contains(&ValueId(id)))
                .ok_or_else(|| fail("early producer original operation"))?;
            let expected = self
                .values
                .get(&id)
                .ok_or_else(|| fail("captured early producer"))?;
            let actual = read.value(row, ValueId(id))?;
            println!(
                "QWEN35_BATCHING_SAME_PRIOR_VALUE row={row} producer={id} node={} name={:?} operation={:?} inputs={:?} difference={:?}",
                producer.0,
                producer.1.name,
                producer.1.operation,
                producer.1.inputs,
                diagnostic::difference(actual, expected)
            );
        }
        let mut prior_planes = 0usize;
        let mut prior_mismatches = 0usize;
        for ((slot, candidate), expected) in &self.states {
            if *candidate
                && !matches!(
                    slot,
                    StateSlot::Conv(zerogravity::accelerator::program::ConvDomain(0))
                        | StateSlot::Recurrent(zerogravity::accelerator::program::RecurrentDomain(
                            0
                        ))
                )
            {
                continue;
            }
            let actual = compare::state(read, row, *slot, *candidate)?;
            ensure(
                actual.1.is_some() == (expected.len() == 2),
                "diagnostic paired state population",
            )?;
            for (plane, (actual, expected)) in [Some(actual.0), actual.1]
                .into_iter()
                .flatten()
                .zip(expected)
                .enumerate()
            {
                let difference = diagnostic::difference(actual, expected);
                if !candidate {
                    prior_planes += 1;
                    prior_mismatches += usize::from(!difference.exact());
                }
                if !candidate && !difference.exact()
                    || matches!(
                        slot,
                        StateSlot::Conv(zerogravity::accelerator::program::ConvDomain(0))
                            | StateSlot::Recurrent(
                                zerogravity::accelerator::program::RecurrentDomain(0)
                            )
                    )
                {
                    println!(
                        "QWEN35_BATCHING_SAME_PRIOR_STATE row={row} slot={slot:?} candidate={candidate} plane={plane} difference={difference:?}"
                    );
                }
            }
        }
        println!(
            "QWEN35_BATCHING_SAME_PRIOR_PRIORS row={row} planes={prior_planes} mismatches={prior_mismatches}"
        );
        Ok(())
    }
    fn compare(&self, read: &native::TextCohortReadback<'_>, row: usize) -> Result<()> {
        ensure(
            read.records[row].call().rows().ids() == self.input,
            "same actual committed producer token",
        )?;
        for (&id, expected) in &self.values {
            if id < 9 {
                continue;
            } // Early producers are diagnostic only.
            oracle::exact(
                read.value(row, ValueId(id))?,
                expected,
                &format!("same-prior row{row} full producer{id}"),
            )?;
        }
        for (&node, expected) in &self.sums {
            oracle::exact(
                read.rms_sum(row, node)?,
                expected,
                &format!("same-prior row{row} dual sum{node}"),
            )?;
        }
        for ((slot, candidate), expected) in &self.states {
            let actual = compare::state(read, row, *slot, *candidate)?;
            oracle::exact(
                actual.0,
                &expected[0],
                &format!("same-prior row{row} domain{slot:?} candidate{candidate} first"),
            )?;
            ensure(
                actual.1.is_some() == (expected.len() == 2),
                "same typed paired state population",
            )?;
            if let Some(actual) = actual.1 {
                oracle::exact(actual, &expected[1], "same-prior full paired KV")?;
            }
        }
        ensure(
            read.selected[row] == self.selected,
            "same-prior natural ID/fullV numeric max/tie",
        )
    }
}
async fn step(
    model: &Arc<qwen35_cuda::Model>,
    sessions: &mut [qwen35_cuda::Session],
    host: &TransferAdmission,
) -> Result<()> {
    let loaded = model.resident();
    let flags = [const { AtomicBool::new(false) }; 4];
    let cancelled = flags.iter().collect::<Vec<_>>();
    let roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let resources = loaded.resources();
    ensure(
        resources.root_initialization_host_bytes.is_multiple_of(3),
        "source root initialization inventory",
    )?;
    let root_bytes = resources.root_initialization_host_bytes / 3;
    let row_payload = root_bytes
        .checked_mul(2)
        .and_then(|v| v.checked_add(51 * 2048 + 10 * 65536 + 248320 * 2 + 24 * 4096))
        .ok_or_else(|| fail("complete same-prior payload overflow"))?;
    let payload = row_payload
        .checked_mul(4)
        .ok_or_else(|| fail("four-row expected payload overflow"))?;
    ensure(
        payload
            .checked_add(16 * MIB)
            .is_some_and(|v| v <= 320 * MIB),
        "four full prior/candidate roots, all values/sums/fullV and bounded map/Vec/capture metadata",
    )?;
    // Cold initialization is already known-complete; its transient payloads and
    // backend staging cannot overlap this fixture-only expected arena. Source
    // steady owners include the SAME exclusive N1/B4 readback reservation.
    let steady = resources
        .host_existing_bytes
        .checked_add(resources.host_retained_additional_bytes)
        .ok_or_else(|| fail("source steady host overflow"))?;
    let host_limit = fixture_host::declared_limit(host)?;
    ensure(
        steady
            .checked_add(320 * MIB + 16 * MIB)
            .is_some_and(|v| v as u64 <= host_limit),
        "source steady + expected320MiB + original metadata16MiB fits declared HOST",
    )?;
    let before = host.arbiter.snapshot(&host.resource)?.reserved;
    let _expected = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: (320 * MIB) as u64,
        }],
    )?;
    let admitted = host.arbiter.snapshot(&host.resource)?.reserved;
    ensure(
        admitted == before + (320 * MIB) as u64 && admitted <= host_limit,
        "actual host ledger admitted expected arena BEFORE every capture clone",
    )?;
    println!(
        "QWEN35_BATCHING_EXPECTED_ADMISSION root_bytes={root_bytes} payload_bound={payload} arena={} source_steady={steady} actual_before={before} actual_admitted={admitted}",
        320 * MIB
    );
    let mut expected = Vec::with_capacity(4);
    for session in sessions.iter() {
        let mut captured = None;
        // Real original N1 execution against THIS SAME private root; discarded
        // proposal cannot alter the prior or manufacture cohort membership.
        let pending = session
            .prepare_next_with(None, |read| {
                captured = Some(Expected::capture(read, row_payload)?);
                Ok(())
            })
            .await?;
        drop(pending);
        expected.push(captured.ok_or_else(|| fail("actual completed N1 reference callback"))?);
        ensure(
            expected.iter().map(Expected::bytes).sum::<usize>() <= payload,
            "bounded charged reference bytes",
        )?;
    }
    ensure(
        sessions.iter().zip(&roots).all(|(s, r)| s.binding() == *r),
        "all N1 references leave actual roots unchanged",
    )?;
    let mut callbacks = 0;
    let mut validate = |read: native::TextCohortReadback<'_>| {
        callbacks += 1;
        for (row, expected) in expected.iter().enumerate() {
            expected.diagnose(&read, row)?;
        }
        for (row, expected) in expected.iter().enumerate() {
            expected.compare(&read, row)?;
        }
        Ok(())
    };
    let pending = model
        .prepare_feedback_cohort(
            &sessions.iter().collect::<Vec<_>>(),
            &cancelled,
            Some(&mut validate),
        )
        .await?
        .ok_or_else(|| fail("complete admitted continuous B4"))?;
    ensure(
        callbacks == 1,
        "one genuine same-prior all-row completion callback",
    )?;
    let receipt = pending
        .commit(
            &mut sessions
                .iter_mut()
                .map(|s| s.engine_mut())
                .collect::<Vec<_>>(),
            &cancelled,
        )?
        .ok_or_else(|| fail("same-prior all-row commit"))?;
    for (row, ((session, before), published)) in
        sessions.iter().zip(&roots).zip(&receipt.rows).enumerate()
    {
        ensure(
            session.binding() == published.root
                && published.root.session() == before.session()
                && published.root.frontier() == before.frontier() + 1
                && published.root.root() == before.root() + 1
                && published.selected_id == expected[row].selected.0
                && session.cohort_feedback_eligible(),
            "actual same-root mixed frontier publication/lineage",
        )?;
    }
    conservation(model, host)?;
    println!(
        "QWEN35_BATCHING_SAME_PRIOR_PASS frontiers={:?} request_batch_size=4 all24=exact dual_sums=24 all42_prior_candidate=exact fullV=exact natural=exact",
        roots.iter().map(|r| r.frontier()).collect::<Vec<_>>()
    );
    Ok(())
}
pub(super) async fn matrix(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    original: &oracle::Original,
) -> Result<()> {
    let ids = original.prompt()?;
    let mut sessions = Vec::new();
    for extra in [0, 1, 2, 3] {
        let mut session = model.session(None).await?;
        session.infer_prompt(&ids, None).await?;
        for _ in 0..extra {
            session.infer_next(None).await?;
        }
        ensure(
            !session.cohort_feedback_eligible() && session.cached_cohort_eligible(),
            "N1 current source root admits cache without forged B4 lineage",
        )?;
        sessions.push(session);
    }
    step(model, &mut sessions, host).await?;
    // Cancelled sibling is a typed prelaunch discard; no root changes. Retire
    // that actual request, admit a new independent N1-prefilled request, regroup.
    let roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let flags = [const { AtomicBool::new(false) }; 4];
    flags[1].store(true, Ordering::Release);
    ensure(
        matches!(
            model
                .prepare_feedback_cohort_status(
                    &sessions.iter().collect::<Vec<_>>(),
                    &flags.iter().collect::<Vec<_>>(),
                    None
                )
                .await?,
            native::TextCohortPreparation::CancelledBeforeLaunch
        ) && sessions.iter().zip(&roots).all(|(s, r)| s.binding() == *r),
        "typed checked prelaunch discard preserves ALL current roots",
    )?;
    drop(sessions.remove(1));
    let mut replacement = model.session(None).await?;
    replacement.infer_prompt(&ids, None).await?;
    sessions.push(replacement);
    step(model, &mut sessions, host).await?;
    // Stagger a genuine N1 successor between cohorts; it must remain eligible
    // with its own new feedback/root, never requiring a prior B4 stamp.
    sessions[0].infer_next(None).await?;
    ensure(
        !sessions[0].cohort_feedback_eligible() && sessions[0].cached_cohort_eligible(),
        "independent N1 successor rejoins actual cached cohort",
    )?;
    step(model, &mut sessions, host).await?;
    drop(sessions);
    conservation(model, host)
}

// Diagnostic proposals may publish their OWN computed state/feedback here only.
// Original matrix/step and every original N1-equivalence failure stay above.
async fn diagnostic_n1(
    session: &mut qwen35_cuda::Session,
    prompt: &[u32],
    history: &mut export::History,
    trace: &mut export::Trace,
    prefill: bool,
) -> Result<()> {
    let before = session.binding();
    let mut callbacks = 0;
    let mut capture = |read: native::TextReadback<'_>| {
        callbacks += 1;
        trace.capture(
            export::Completed::N1(&read),
            &[before],
            prompt,
            std::slice::from_ref(history),
        )
    };
    let receipt = if prefill {
        session
            .infer_prompt_with(prompt, None, &mut capture)
            .await?
    } else {
        session.infer_next_with(None, &mut capture).await?
    };
    ensure(
        callbacks == 1 && session.binding() == receipt.root,
        "diagnostic actual N1 completion/publication",
    )?;
    trace.committed(&[receipt.root], &[(receipt.selected_id, receipt.max_bits)])?;
    history.commit(receipt.root, receipt.selected_id, receipt.max_bits, 1)
}
async fn diagnostic_b4(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    sessions: &mut [qwen35_cuda::Session],
    prompt: &[u32],
    histories: &mut [export::History; 4],
    trace: &mut export::Trace,
) -> Result<()> {
    let before = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let flags = [const { AtomicBool::new(false) }; 4];
    let cancelled = flags.iter().collect::<Vec<_>>();
    let mut callbacks = 0;
    let mut capture = |read: native::TextCohortReadback<'_>| {
        callbacks += 1;
        trace.capture(export::Completed::B4(&read), &before, prompt, histories)
    };
    let pending = model
        .prepare_feedback_cohort(
            &sessions.iter().collect::<Vec<_>>(),
            &cancelled,
            Some(&mut capture),
        )
        .await?
        .ok_or_else(|| fail("diagnostic complete genuine B4"))?;
    ensure(
        callbacks == 1 && sessions.iter().zip(&before).all(|(s, r)| s.binding() == *r),
        "diagnostic callback before any row publication",
    )?;
    let receipt = pending
        .commit(
            &mut sessions
                .iter_mut()
                .map(|s| s.engine_mut())
                .collect::<Vec<_>>(),
            &cancelled,
        )?
        .ok_or_else(|| fail("diagnostic own computed all-row publication"))?;
    ensure(
        receipt.physical_rows == 4
            && receipt.rows.len() == 4
            && receipt.executed_nodes == 255
            && receipt.native_projection_launches == 115,
        "diagnostic original full B4 body",
    )?;
    let roots = receipt.rows.iter().map(|r| r.root).collect::<Vec<_>>();
    let selected = receipt
        .rows
        .iter()
        .map(|r| (r.selected_id, r.max_bits))
        .collect::<Vec<_>>();
    ensure(
        sessions.iter().zip(&roots).all(|(s, r)| s.binding() == *r),
        "diagnostic genuine row roots",
    )?;
    trace.committed(&roots, &selected)?;
    for (history, row) in histories.iter_mut().zip(&receipt.rows) {
        history.commit(row.root, row.selected_id, row.max_bits, 4)?;
    }
    conservation(model, host)
}
pub(super) async fn diagnostic_trajectory(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    original: &oracle::Original,
) -> Result<()> {
    diagnostic_trajectory_with_prompt(model, host, &original.prompt()?).await
}
pub(super) async fn diagnostic_trajectory_with_prompt(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    prompt: &[u32],
) -> Result<()> {
    ensure(
        prompt.len() == 47,
        "actual source fixed-prompt trajectory input",
    )?;
    let mut trace = export::Trace::configured(model, host)?;
    let mut histories = [export::History::empty(); 4];
    let mut sessions = Vec::with_capacity(4);
    for (row, history) in histories.iter_mut().enumerate() {
        let mut session = model.session(None).await?;
        diagnostic_n1(&mut session, prompt, history, &mut trace, true).await?;
        for _ in 0..row {
            diagnostic_n1(&mut session, prompt, history, &mut trace, false).await?;
        }
        sessions.push(session);
    }
    diagnostic_b4(
        model,
        host,
        &mut sessions,
        prompt,
        &mut histories,
        &mut trace,
    )
    .await?;
    let before = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let flags = [const { AtomicBool::new(false) }; 4];
    flags[1].store(true, Ordering::Release);
    ensure(
        matches!(
            model
                .prepare_feedback_cohort_status(
                    &sessions.iter().collect::<Vec<_>>(),
                    &flags.iter().collect::<Vec<_>>(),
                    None
                )
                .await?,
            native::TextCohortPreparation::CancelledBeforeLaunch
        ) && sessions.iter().zip(&before).all(|(s, r)| s.binding() == *r),
        "diagnostic typed cancellation preserves every root",
    )?;
    trace.event(serde_json::json!({"kind":"cancelled-before-launch","roots_before":export::Trace::roots(&before),
        "roots_after":export::Trace::roots(&before),"cancelled_request_index":1,"submitted":false}))?;
    trace.event(
        serde_json::json!({"kind":"retire","root":export::Trace::roots(&[sessions[1].binding()])}),
    )?;
    drop(sessions.remove(1));
    histories[1..].rotate_left(1);
    histories[3] = export::History::empty();
    let mut replacement = model.session(None).await?;
    diagnostic_n1(
        &mut replacement,
        prompt,
        &mut histories[3],
        &mut trace,
        true,
    )
    .await?;
    sessions.push(replacement);
    diagnostic_b4(
        model,
        host,
        &mut sessions,
        prompt,
        &mut histories,
        &mut trace,
    )
    .await?;
    let before = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let partial_flags = [const { AtomicBool::new(false) }; 3];
    ensure(
        matches!(
            model
                .prepare_feedback_cohort_status(
                    &sessions[..3].iter().collect::<Vec<_>>(),
                    &partial_flags.iter().collect::<Vec<_>>(),
                    None
                )
                .await,
            Err(Error::Malformed(_))
        ) && sessions.iter().zip(&before).all(|(s, r)| s.binding() == *r),
        "diagnostic partial width preflight has no effects",
    )?;
    trace.event(serde_json::json!({"kind":"partial-width-preflight-rejected","request_batch_size":3,
        "roots_before":export::Trace::roots(&before),"roots_after":export::Trace::roots(&before),"submitted":false,
        "worker_fallback_qualified":false}))?;
    diagnostic_n1(
        &mut sessions[0],
        prompt,
        &mut histories[0],
        &mut trace,
        false,
    )
    .await?;
    let before = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    sessions.swap(0, 2);
    histories.swap(0, 2);
    let after = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    trace.event(serde_json::json!({"kind":"permutation","order":[2,1,0,3],
        "roots_before":export::Trace::roots(&before),"roots_after":export::Trace::roots(&after)}))?;
    diagnostic_b4(
        model,
        host,
        &mut sessions,
        prompt,
        &mut histories,
        &mut trace,
    )
    .await?;
    let final_roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    trace.event(
        serde_json::json!({"kind":"final-roots","roots":export::Trace::roots(&final_roots)}),
    )?;
    drop(sessions);
    conservation(model, host)?;
    trace.finish()
}
