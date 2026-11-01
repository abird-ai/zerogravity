//! Ordinary optional-observer caller; original managed source and session publish.
use super::*;
use crate::gemma4::execution::Input;
use crate::gpu::{Device, TextHostOwners, TextStorage};

pub(super) fn generate(
    loaded: &Arc<Model>,
    input: Input,
    decoder: &Input,
    max_output: usize,
    flag: &AtomicBool,
    report: &mut impl FnMut(Report<'_>) -> Result<()>,
) -> Result<()> {
    let (candidate, mut session) = futures_lite::future::block_on(loaded.load_hip_qualification(
        Device::Hip(Default::default()),
        input,
        TextHostOwners {
            additional_recorded_calls: 0,
            additional_frame_bytes: 0,
        },
        TextStorage::ResidentProjections,
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
    report(Report::Text(text.as_str()))?;
    report(Report::Finished {
        backend: Backend::HipResident,
        emitted: ids.len(),
        calls,
        frontier: session.binding().frontier(),
        first_root: None,
    })?;
    drop((text, session, candidate));
    Ok(())
}
