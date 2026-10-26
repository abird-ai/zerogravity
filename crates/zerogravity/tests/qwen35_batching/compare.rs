//! Every comparison follows the original known-completion/all-row boundary.
use super::*;
use oracle::{Original, exact};
use zerogravity::accelerator::program::{
    ConvDomain, KvDomain, RecurrentDomain, StateSlot, ValueId,
};
pub(super) fn state<'a>(
    read: &native::TextCohortReadback<'a>,
    row: usize,
    slot: StateSlot,
    candidate: bool,
) -> Result<(&'a [u8], Option<&'a [u8]>)> {
    let table = if candidate {
        read.candidate
    } else {
        read.prior
    };
    let found = table
        .get(row)
        .ok_or_else(|| fail("actual state row"))?
        .iter()
        .filter(|(s, _)| *s == slot)
        .collect::<Vec<_>>();
    ensure(found.len() == 1, "unique actual independent state domain")?;
    let range = found[0].1.clone();
    let raw = read
        .raw
        .get(range)
        .ok_or_else(|| fail("actual state readback range"))?;
    ensure(
        raw.len()
            == if matches!(slot, StateSlot::Kv(_)) {
                2
            } else {
                1
            },
        "canonical state physical planes",
    )?;
    Ok((&raw[0], raw.get(1).map(|v| &v[..])))
}
fn kv(actual: (&[u8], Option<&[u8]>), original: &[u8], active: usize) -> Result<()> {
    let (k, v) = actual;
    let v = v.ok_or_else(|| fail("actual paired K/V owner"))?;
    ensure(
        k.len() == 512 * 512 * 4 && v.len() == k.len() && original.len() == 2 * active * 512 * 2,
        "full canonical/native KV extents",
    )?;
    for position in 0..active {
        for (plane, offset) in [(k, 0usize), (v, 256)] {
            for head in 0..2 {
                let source = (head * active * 512 + position * 512 + offset) * 2;
                let target = (position * 512 + head * 256) * 4;
                exact(
                    &plane[target..target + 1024],
                    &oracle::widen(&original[source..source + 512])?,
                    "canonical exact active KV",
                )?;
            }
        }
    }
    ensure(
        k[active * 512 * 4..]
            .iter()
            .chain(&v[active * 512 * 4..])
            .all(|v| *v == 0),
        "canonical future KV is unchanged positive zero",
    )
}
pub(super) fn completed(
    read: native::TextCohortReadback<'_>,
    original: &Original,
    call: usize,
) -> Result<()> {
    ensure(
        read.records.len() == 4
            && read.prior.len() == 4
            && read.candidate.len() == 4
            && read.selected.len() == 4
            && read.sums.len() == 24
            && read.values.len() == 327,
        "complete actual B4 producers/independent domains/sums",
    )?;
    let rows = if call == 0 { 47 } else { 1 };
    let base = if call == 0 { 0 } else { 46 + call };
    let end = base + rows;
    let head = read
        .head_input
        .and_then(|i| read.raw.get(i))
        .ok_or_else(|| fail("actual full B4 head input"))?;
    ensure(
        head.len() == 4 * 1024 * 2,
        "actual terminal gathered source rows",
    )?;
    for row in 0..4 {
        let record = &read.records[row];
        ensure(
            record.plan().nodes().len() == 255
                && record.plan().values().len() == 327
                && record.selection().boundaries().len() == 24
                && record.call().rows().len() == rows
                && record.call().rows().ids() == original.input(row, call)?,
            "actual independent source-issued record/phase/natural feedback",
        )?;
        let mut previous: Option<(Vec<u8>, Vec<u8>)> = None;
        let mut sums = 0;
        for layer in 0..24 {
            let branch = original.get(
                row,
                call,
                Some(layer),
                "block.output.branch",
                "torch.bfloat16",
                &[rows, 1024],
            )?;
            let residual = original.get(
                row,
                call,
                Some(layer),
                "block.output.residual",
                "torch.bfloat16",
                &[rows, 1024],
            )?;
            let [a, b] = record.selection().boundaries()[layer];
            exact(
                read.value(row, a)?,
                &branch,
                &format!("B4 row{row} layer{layer} branch"),
            )?;
            exact(
                read.value(row, b)?,
                &residual,
                &format!("B4 row{row} layer{layer} residual"),
            )?;
            if let Some((a, b)) = &previous {
                let node = input_rms_node(record, layer)?;
                exact(
                    read.rms_sum(row, node)?,
                    &oracle::sum(a, b)?,
                    "independent actual unrounded dual F32 sum",
                )?;
                sums += 1;
            }
            let slots = if layer % 4 == 3 {
                vec![StateSlot::Kv(KvDomain(layer))]
            } else {
                vec![
                    StateSlot::Conv(ConvDomain(layer)),
                    StateSlot::Recurrent(RecurrentDomain(layer)),
                ]
            };
            for candidate in [false, true] {
                for slot in &slots {
                    let actual = state(&read, row, *slot, candidate)?;
                    if !candidate && call == 0 {
                        ensure(
                            actual
                                .0
                                .iter()
                                .chain(actual.1.unwrap_or(&[]))
                                .all(|v| *v == 0),
                            "genuine initial private row root is positive zero",
                        )?;
                        continue;
                    }
                    let suffix = if candidate { "after" } else { "before" };
                    match slot {
                        StateSlot::Kv(_) => {
                            let active = if candidate { end } else { base };
                            kv(
                                actual,
                                &original.get(
                                    row,
                                    call,
                                    Some(layer),
                                    &format!("state.kv.{suffix}"),
                                    "torch.bfloat16",
                                    &[2, active, 512],
                                )?,
                                active,
                            )?;
                        }
                        StateSlot::Conv(_) => exact(
                            actual.0,
                            &oracle::convolution(&original.get(
                                row,
                                call,
                                Some(layer),
                                &format!("state.conv.{suffix}"),
                                "torch.bfloat16",
                                &[3, 6144],
                            )?)?,
                            "exact actual consumed/candidate Conv3",
                        )?,
                        StateSlot::Recurrent(_) => exact(
                            actual.0,
                            &oracle::recurrent(&original.get(
                                row,
                                call,
                                Some(layer),
                                &format!("state.recurrent.{suffix}"),
                                "torch.float32",
                                &[16, 128, 128],
                            )?)?,
                            "exact actual consumed/candidate recurrent",
                        )?,
                    }
                }
            }
            previous = Some((branch, residual));
        }
        let (a, b) = previous.ok_or_else(|| fail("last complete original block"))?;
        exact(
            read.rms_sum(row, 253)?,
            &oracle::sum(&a, &b)?,
            "actual final unrounded F32 sum",
        )?;
        sums += 1;
        ensure(sums == 24, "all24 independent unrounded sums")?;
        let normalized = original.get(
            row,
            call,
            None,
            "final.normalized",
            "torch.bfloat16",
            &[rows, 1024],
        )?;
        exact(
            read.value(
                row,
                record
                    .selection()
                    .normalized()
                    .ok_or_else(|| fail("actual final normalization selector"))?,
            )?,
            &normalized,
            "full original final BF16 normalization",
        )?;
        let residual = oracle::sum(&a, &b)?
            .chunks_exact(4)
            .flat_map(|v| {
                let bits = u32::from_le_bytes(v.try_into().unwrap());
                let low = bits & 0xffff;
                let high = (bits >> 16) as u16;
                high.wrapping_add(u16::from(low > 0x8000 || low == 0x8000 && high & 1 != 0))
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        exact(
            read.value(row, ValueId(325))?,
            &residual,
            "full source final rounded residual",
        )?;
        exact(
            &head[row * 2048..(row + 1) * 2048],
            &original.get(row, call, None, "head.input", "torch.bfloat16", &[1, 1024])?,
            "actual same-flight gathered head producer",
        )?;
        let logits = original.get(
            row,
            call,
            None,
            "logits.pre_sampler",
            "torch.bfloat16",
            &[1, 248320],
        )?;
        exact(
            &logits,
            &original.get(
                row,
                call,
                None,
                "sampler.input",
                "torch.bfloat16",
                &[1, 248320],
            )?,
            "original complete head to sampler join",
        )?;
        exact(
            read.value(row, record.selection().output())?,
            &logits,
            "full[B,V] genuine original BF16 head",
        )?;
        let greedy = oracle::greedy(&logits)?;
        ensure(
            greedy.0 == original.selected(row, call)? && read.selected[row] == greedy,
            "natural original greedy ID/numeric tie/max bits",
        )?;
    }
    println!(
        "QWEN35_BATCHING_NUMERIC_PASS call={call} physical_rows={} actual_request_batch_size=4 all24=exact all42_priors=exact all42_candidates=exact dual_sums=24 fullV=exact all327_independent=false",
        4 * rows
    );
    Ok(())
}

pub(super) fn residual_rms_node(
    record: &zerogravity::gpu::SourceBindings,
    inputs: [ValueId; 2],
) -> Result<usize> {
    use zerogravity::accelerator::program::{PlanOperation, SourceOperatorKind};
    let mut found = record.plan().nodes().iter().enumerate().filter(|(_, node)| {
        node.inputs.as_slice() == inputs
            && matches!(&node.operation, PlanOperation::SourceOperator { descriptor, state: None }
                if matches!(descriptor.kind(), SourceOperatorKind::ResidualRms(_))
                    && descriptor.call() == record.call())
    });
    let (index, node) = found
        .next()
        .ok_or_else(|| fail("actual boundary-pair residual RMS producer"))?;
    ensure(
        found.next().is_none(),
        "unique actual boundary-pair residual RMS producer",
    )?;
    let output = *node
        .outputs
        .first()
        .ok_or_else(|| fail("actual residual RMS normalized output"))?;
    let producer = record
        .plan()
        .source_value_provenance(output)?
        .ok_or_else(|| fail("actual residual RMS SSA provenance"))?;
    ensure(
        producer.node_index() == index
            && producer.output_index() == 0
            && producer.call() == record.call()
            && producer.inputs() == inputs,
        "same actual residual RMS SSA inputs",
    )?;
    Ok(index)
}
pub(super) fn input_rms_node(
    record: &zerogravity::gpu::SourceBindings,
    layer: usize,
) -> Result<usize> {
    let previous = layer
        .checked_sub(1)
        .and_then(|i| record.selection().boundaries().get(i))
        .ok_or_else(|| fail("previous actual layer boundary pair"))?;
    residual_rms_node(record, *previous)
}
