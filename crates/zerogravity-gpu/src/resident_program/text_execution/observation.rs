//! Bounded reads of actual producers under the original flight and source budget.
use super::{cold::Cold, *};
use crate::resident_leases::{ResidentLease, ResidentSubmission};
use crate::resident_program::{
    StateBinding,
    text_observation::{TextFrame, TextFrameKind, TextObserver},
};
use zerogravity_core::accelerator::program::TextInvocation;
impl Cold {
    pub(super) async fn observe_states(
        &self,
        flight: &mut ResidentSubmission<'_>,
        call: &TextInvocation,
        states: &[StateBinding],
        prior: bool,
        observer: &mut dyn TextObserver,
    ) -> Result<()> {
        for b in states {
            let buffers = if prior { &b.prior } else { &b.candidate };
            for (plane, buffer) in buffers.buffers().into_iter().enumerate() {
                self.observe(
                    flight,
                    call,
                    TextFrameKind::State {
                        slot: b.slot,
                        plane,
                        prior,
                    },
                    buffer.original_lease()?,
                    buffer.elements,
                    observer,
                )
                .await?;
            }
        }
        Ok(())
    }
    pub(super) async fn observe(
        &self,
        flight: &mut ResidentSubmission<'_>,
        call: &TextInvocation,
        kind: TextFrameKind,
        lease: &ResidentLease,
        words: usize,
        observer: &mut dyn TextObserver,
    ) -> Result<()> {
        if !observer.wants(kind) {
            return Ok(());
        }
        if words == 0 || words.checked_mul(4).is_none_or(|n| n > lease.bytes()) {
            return Err(Error::State(
                "text-execution observation exceeds original producer".into(),
            ));
        }
        let chunk = crate::resident_leases::source_page::PAGE_BYTES / 4;
        for offset in (0..words).step_by(chunk) {
            let n = (words - offset).min(chunk);
            let read = lease.readback(offset * 4, n * 4)?;
            let frame = TextFrame {
                base: call.tokens().base(),
                rows: call.tokens().len(),
                kind,
                offset_words: offset,
                total_words: words,
            };
            self.frame
                .compare(flight, &read, |actual| observer.completed(frame, actual))
                .await?;
        }
        Ok(())
    }
}
