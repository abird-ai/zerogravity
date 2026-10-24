//! Ordinary optional-observer caller; original managed source and session publish.
use super::*;
use zerogravity::gemma4_cpu::Gemma4CpuInput;
use zerogravity::gpu::{Device, GemmaTextHostOwners, GemmaTextStorage};

pub(super) fn run(
    loaded: &Arc<Gemma4CpuLoaded>,
    input: Gemma4CpuInput,
    decoder: &Gemma4CpuInput,
    max_output: usize,
    flag: &AtomicBool,
) -> Result<()> {
    let (candidate, mut session) = futures_lite::future::block_on(loaded.load_hip_qualification(
        Device::Hip(Default::default()),
        input,
        GemmaTextHostOwners {
            additional_recorded_calls: 0,
            additional_frame_bytes: 0,
        },
        GemmaTextStorage::ResidentProjections,
        Some(flag),
    ))?;
    let mut ids = Vec::with_capacity(max_output);
    let first =
        futures_lite::future::block_on(
            session.infer_prompt(max_output, None, Some(flag), |_| Ok(())),
        )?;
    if !first.eos {
        ids.push(first.selected_id);
    }
    let mut calls = 1;
    while !session.stopped() {
        let call =
            futures_lite::future::block_on(session.infer_next(None, Some(flag), |_| Ok(())))?;
        if !call.eos {
            ids.push(call.selected_id);
        }
        calls += 1;
    }
    let text = decoder.decode_output(&ids, true, Some(flag))?;
    println!("{}", text.as_str());
    eprintln!(
        "Gemma HIP resident candidate: {} emitted, {calls} calls, frontier{}; observer=none private_feedback=true trained/HTTP/performance unqualified",
        ids.len(),
        session.binding().frontier()
    );
    drop((text, session, candidate));
    Ok(())
}
