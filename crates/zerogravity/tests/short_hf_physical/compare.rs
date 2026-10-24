//! Exact retained source values and canonical active state after completion.
use super::*;
use native::ShortHfReadback;
use oracle::{Archive, exact};
use zerogravity::accelerator::program::{
    ConvDomain, KvDomain, PlanOperation, RecurrentDomain, StateSlot, ValueId, Vllm08Kind,
    Vllm08RmsSite,
};

pub(super) fn completed(
    read: ShortHfReadback<'_>,
    reference: &Path,
    call: usize,
    prompt_rows: usize,
) -> Result<()> {
    // This function is invoked by the existing trusted-completion boundary.
    let fixed =
        oracle::FixedCell::open(reference, read.record().call().model_profile(), prompt_rows)?;
    let acceptance = fixed.json("trajectory-validation.json", 16 * MIB)?;
    ensure(
        acceptance["all24_all42_post_validation"] == "PASS"
            && acceptance["common_final_fullv_greedy_validation"] == "PASS"
            && acceptance["eight_calls_complete"] == true
            && acceptance["original_source_and_lifetime_validation"] == "PASS"
            && acceptance["original"]["source_receipt_sha256"] == *fixed.source_manifest()
            && acceptance["prompt_rows"].as_u64() == Some(prompt_rows as u64)
            && acceptance["outputs"] == 8,
        "fixed accepted whole original/source/lifetime reference",
    )?;
    let plain = reference.join("plain");
    let states = fixed.json("plain/all42/report.json", 8 * MIB)?;
    let worker = fixed.json("plain/worker-report.json", 16 * MIB)?;
    let result = fixed.json("plain/result.json", MIB)?;
    let expected_ids = result["cases"][0]["completion_ids"]
        .as_array()
        .ok_or_else(|| fail("original natural IDs"))?;
    ensure(
        expected_ids.len() == 8 && call < 8,
        "complete eight-call natural trajectory",
    )?;
    let record = read.record();
    ensure(
        record.plan().nodes().len() == 255 && record.boundaries().len() == 24,
        "same complete original Rust program",
    )?;
    let expected_input = if call == 0 {
        result["cases"][0]["prompt_ids"]
            .as_array()
            .ok_or_else(|| fail("original prompt IDs"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as u32)
                    .ok_or_else(|| fail("original prompt ID"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![
            expected_ids[call - 1]
                .as_u64()
                .ok_or_else(|| fail("original feedback ID"))? as u32,
        ]
    };
    ensure(
        record.call().request().teachers().ids() == expected_input,
        "actual computed feedback matches original; no caller-forced sampled ID",
    )?;
    let rows = if call == 0 { prompt_rows } else { 1 };
    let base = if call == 0 { 0 } else { prompt_rows + call - 1 };
    let end = base + rows;
    ensure(record.call().rows() == rows, "genuine unsplit phase rows")?;
    let mut previous: Option<(Vec<u8>, Vec<u8>)> = None;
    let mut independent_sums = 0;
    for layer in 0..24 {
        let key = layer.to_string();
        let archive = Archive::open(
            &plain.join(format!("all42/{layer}")),
            &states["archives"][&key],
        )?;
        let branch = archive.get(call, "block.output.branch")?;
        let residual = archive.get(call, "block.output.residual")?;
        let [a, b] = record.boundaries()[layer];
        exact(read.value(a)?, branch, &format!("layer{layer} branch"))?;
        exact(read.value(b)?, residual, &format!("layer{layer} residual"))?;
        if let Some((prior_branch, prior_residual)) = &previous {
            let node = record.plan().nodes().iter().position(|n| matches!(&n.operation,
                PlanOperation::TypedVllm08{descriptor,..} if descriptor.kind()==Vllm08Kind::ResidualRms(Vllm08RmsSite::Input)
                    && descriptor.layer().is_some_and(|l|l.index()==layer)))
                .ok_or_else(||fail("actual input residual RMS node"))?;
            exact(
                read.rms_sum(node)?,
                &oracle::sum(prior_branch, prior_residual)?,
                "unrounded input residual F32 sum",
            )?;
            independent_sums += 1;
        }
        if layer % 4 == 3 {
            oracle::kv(
                read.state(StateSlot::Kv(KvDomain(layer)), true)?,
                &archive,
                call,
                "kv.after",
                end,
            )?;
            exact(
                read.kv_transport_status(layer)?,
                &[0; 4],
                "actual KV transport status",
            )?;
        } else {
            let conv = read.state(StateSlot::Conv(ConvDomain(layer)), true)?;
            let recurrent = read.state(StateSlot::Recurrent(RecurrentDomain(layer)), true)?;
            ensure(
                conv.1.is_none() && recurrent.1.is_none(),
                "single canonical GDN state owner",
            )?;
            exact(
                conv.0,
                &oracle::convolution(archive.get(call, "state.conv.after")?)?,
                "canonical convolution",
            )?;
            exact(
                recurrent.0,
                &oracle::recurrent(archive.get(call, "state.recurrent.after")?)?,
                "canonical recurrent",
            )?;
        }
        if call > 0 {
            let prior = Archive::open(
                &plain.join(format!("all42-prior/{layer}")),
                &states["prior_archives"][&key],
            )?;
            if layer % 4 == 3 {
                oracle::kv(
                    read.state(StateSlot::Kv(KvDomain(layer)), false)?,
                    &prior,
                    call,
                    "kv.before",
                    base,
                )?;
            } else {
                exact(
                    read.state(StateSlot::Conv(ConvDomain(layer)), false)?.0,
                    &oracle::convolution(prior.get(call, "state.conv.before")?)?,
                    "consumed canonical convolution",
                )?;
                exact(
                    read.state(StateSlot::Recurrent(RecurrentDomain(layer)), false)?
                        .0,
                    &oracle::recurrent(prior.get(call, "state.recurrent.before")?)?,
                    "consumed canonical recurrent",
                )?;
            }
        } else {
            for slot in if layer % 4 == 3 {
                vec![StateSlot::Kv(KvDomain(layer))]
            } else {
                vec![
                    StateSlot::Conv(ConvDomain(layer)),
                    StateSlot::Recurrent(RecurrentDomain(layer)),
                ]
            } {
                let (a, b) = read.state(slot, false)?;
                ensure(
                    a.iter().chain(b.unwrap_or(&[])).all(|v| *v == 0),
                    "initial root is canonical positive zero",
                )?;
            }
        }
        previous = Some((branch.to_vec(), residual.to_vec()));
    }
    let final_archive = Archive::open(&plain.join("final"), &worker["final_head"]["raw"])?;
    exact(
        read.value(ValueId(324))?,
        final_archive.get(call, "final_norm.output.0")?,
        "final BF16 normalization",
    )?;
    exact(
        read.value(ValueId(325))?,
        final_archive.get(call, "final_norm.output.1")?,
        "final BF16 residual",
    )?;
    exact(
        read.rms_sum(253)?,
        &oracle::sum(
            final_archive.get(call, "final_norm.input.0")?,
            final_archive.get(call, "final_norm.input.1")?,
        )?,
        "final unrounded F32 sum",
    )?;
    independent_sums += 1;
    ensure(
        independent_sums == 24,
        "24 independently observed dual F32 sums",
    )?;
    let logits = final_archive.get(call, "sampler.input")?;
    exact(
        read.value(record.logits())?,
        logits,
        "full248320 BF16 logits",
    )?;
    let selected = expected_ids[call]
        .as_u64()
        .ok_or_else(|| fail("natural sampled ID"))? as u32;
    let max = greedy(logits)?;
    ensure(
        max.0 == selected && read.greedy() == max,
        "natural greedy ID/max/tie policy",
    )?;
    let positions = read.positions()?;
    ensure(
        positions.len() == 3 * 513 * 8,
        "actual three-axis pitch513 position owner",
    )?;
    for axis in 0..3 {
        for row in 0..513 {
            let at = (axis * 513 + row) * 8;
            let value = i64::from_le_bytes(positions[at..at + 8].try_into().unwrap());
            ensure(
                value == if row < rows { (base + row) as i64 } else { 0 },
                "actual positions and defined padding",
            )?;
        }
    }
    println!(
        "SHORT_HF_NUMERIC_CALL_PASS rows={prompt_rows} call={call} all24_boundaries=exact all42_post=exact consumed_priors=exact independent_dual_f32=24 fullV=exact natural_id={selected} all327_independent=false"
    );
    Ok(())
}

// Numeric ties include signed zero; the lower vocabulary ID wins.
fn greedy(logits: &[u8]) -> Result<(u32, u32)> {
    ensure(
        !logits.is_empty() && logits.len().is_multiple_of(2),
        "complete BF16 vocabulary words",
    )?;
    let mut best = None;
    for (id, word) in logits.chunks_exact(2).enumerate() {
        let bits = (u16::from_le_bytes([word[0], word[1]]) as u32) << 16;
        let value = f32::from_bits(bits);
        ensure(value.is_finite(), "original fullV finite logits")?;
        if best.is_none_or(|(_, old)| value > f32::from_bits(old)) {
            best = Some((id as u32, bits));
        }
    }
    best.ok_or_else(|| fail("complete vocabulary argmax"))
}

#[test]
fn finite_numeric_greedy_keeps_lowest_id_for_signed_zero_ties() {
    assert_eq!(greedy(&[0, 128, 0, 0]).unwrap(), (0, 0x8000_0000));
    assert_eq!(greedy(&[0, 0, 0, 128]).unwrap(), (0, 0));
    assert_eq!(greedy(&[128, 63, 128, 63]).unwrap(), (0, 0x3f80_0000));
    for invalid in [&[][..], &[0][..], &[128, 127][..], &[192, 127][..]] {
        assert!(greedy(invalid).is_err());
    }
}
