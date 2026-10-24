//! Private computed feedback around the SAME ReferenceText/SessionAuthority.
use super::*;
use crate::gemma4::text::tokenizer::Gemma4DecodedText;
use crate::gemma4::text::{Gemma4ReferenceReadback, Gemma4ReferenceSession};
use crate::session_root::RootBinding;
use zerogravity_cpu::reference_text::ReferenceTextObserver;

#[derive(Clone, Copy, Debug)]
pub struct Gemma4CpuCall {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub eos: bool,
    pub emitted_tokens: usize,
}
struct Feedback {
    root: RootBinding,
    id: u32,
}
struct SessionTicket {
    loaded: Arc<Gemma4CpuLoaded>,
}
impl Drop for SessionTicket {
    fn drop(&mut self) {
        if let Ok(mut life) = self.loaded.life.lock() {
            life.sessions -= 1;
        }
    }
}
pub struct Gemma4CpuSession {
    inner: Gemma4ReferenceSession,
    input: Gemma4CpuInput,
    generated: Vec<u32>,
    feedback: Option<Feedback>,
    max_output: usize,
    calls: usize,
    stopped: bool,
    _ids: GlobalResourceLease,
    // Last: actual state/input/output owners retire BEFORE session-count release.
    ticket: SessionTicket,
}
impl Gemma4CpuSession {
    pub(super) fn new(loaded: Arc<Gemma4CpuLoaded>, input: Gemma4CpuInput) -> Result<Self> {
        let ticket = SessionTicket { loaded };
        let ids = ticket.loaded.context.host.arbiter.admit(
            &ticket.loaded.context.host.resource,
            ticket.loaded.context.host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: 65536,
            }],
        )?;
        let inner =
            Gemma4ReferenceSession::new_bf16_candidate(ticket.loaded.model.clone(), MAX_CONTEXT)?;
        Ok(Self {
            inner,
            input,
            generated: Vec::with_capacity(MAX_OUTPUT),
            feedback: None,
            max_output: 0,
            calls: 0,
            stopped: false,
            _ids: ids,
            ticket,
        })
    }
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    /// Borrow only this session's committed source-owned KV; no state input API.
    pub fn kv_row(
        &self,
        domain: crate::accelerator::program::KvDomain,
        token: usize,
    ) -> Result<(&[f32], &[f32])> {
        self.inner.kv_row(domain, token)
    }
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    pub fn infer_prompt(
        &mut self,
        max_output: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4CpuCall> {
        self.infer_prompt_observed(max_output, cancelled, None, |_| Ok(()))
    }
    pub fn infer_next(&mut self, cancelled: Option<&AtomicBool>) -> Result<Gemma4CpuCall> {
        self.infer_next_observed(cancelled, None, |_| Ok(()))
    }
    pub fn infer_prompt_observed(
        &mut self,
        max_output: usize,
        cancelled: Option<&AtomicBool>,
        observer: Option<&dyn ReferenceTextObserver>,
        validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
    ) -> Result<Gemma4CpuCall> {
        cancelled_check(cancelled)?;
        if self.binding().frontier() != 0
            || self.feedback.is_some()
            || self.calls != 0
            || !(1..=MAX_OUTPUT).contains(&max_output)
            || self.input.ids().len() + max_output - 1 > MAX_CONTEXT
        {
            return Err(Error::State(
                "Gemma genuine prompt/root/output/context bound".into(),
            ));
        }
        let ids = self.input.ids().to_vec();
        self.step(&ids, max_output, cancelled, observer, validate)
    }
    /// No supplied token ID, expected bytes or external feedback packet.
    pub fn infer_next_observed(
        &mut self,
        cancelled: Option<&AtomicBool>,
        observer: Option<&dyn ReferenceTextObserver>,
        validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
    ) -> Result<Gemma4CpuCall> {
        cancelled_check(cancelled)?;
        let feedback = self.feedback.as_ref().ok_or_else(|| {
            Error::State("Gemma next requires actual committed greedy feedback".into())
        })?;
        if self.stopped || self.calls >= self.max_output || feedback.root != self.binding() {
            return Err(Error::State(
                "Gemma feedback stopped/stale/consumed/output limit".into(),
            ));
        }
        let id = feedback.id;
        self.step(&[id], self.max_output, cancelled, observer, validate)
    }
    pub fn decode_generated(
        &self,
        skip_special: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4DecodedText> {
        let _operation = self.ticket.loaded.operation()?;
        self.input
            .prepared
            .decode_output(&self.generated, skip_special, cancelled)
    }
    fn step(
        &mut self,
        ids: &[u32],
        limit: usize,
        cancelled: Option<&AtomicBool>,
        observer: Option<&dyn ReferenceTextObserver>,
        mut validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
    ) -> Result<Gemma4CpuCall> {
        let loaded = self.ticket.loaded.clone();
        let _operation = loaded.operation()?;
        self.input.prepared.require_model(&loaded.model)?;
        let before = self.binding();
        let mut computed = None;
        let logits = self
            .inner
            .forward_reference_observed(ids, true, cancelled, observer, |completed| {
                let values = completed
                    .logits()?
                    .ok_or_else(|| Error::State("Gemma selected-last full vocabulary".into()))?;
                computed = Some(greedy(values)?);
                validate(completed)
            })?
            .ok_or_else(|| Error::State("Gemma actual full vocabulary completion".into()))?;
        let (id, bits) =
            computed.ok_or_else(|| Error::State("Gemma completed greedy missing".into()))?;
        let root = self.binding();
        if root.root() != before.root() + 1
            || root.frontier() != before.frontier() + ids.len() as u64
            || greedy(logits.as_slice()?)? != (id, bits)
        {
            return Err(Error::State(
                "Gemma original commit/output correspondence".into(),
            ));
        }
        let eos = self.input.prepared.eos_ids().contains(&id);
        self.calls += 1;
        self.max_output = limit;
        self.stopped = eos || self.calls == limit;
        if !eos {
            self.generated.push(id);
        }
        self.feedback = Some(Feedback { root, id });
        Ok(Gemma4CpuCall {
            root,
            selected_id: id,
            max_bits: bits,
            eos,
            emitted_tokens: self.generated.len(),
        })
    }
}
fn greedy(values: &[f32]) -> Result<(u32, u32)> {
    if values.len() != 262144 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed("Gemma finite complete fullV".into()));
    }
    let best = values
        .iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > values[best] { i } else { best });
    Ok((best as u32, values[best].to_bits()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_finite_greedy_uses_numeric_zero_ties_and_first_source_id() {
        let mut values = vec![-1.0; 262144];
        values[3] = -0.0;
        values[4] = 0.0;
        assert_eq!(greedy(&values).unwrap(), (3, (-0.0f32).to_bits()));
        values[12] = f32::NAN;
        assert!(greedy(&values).is_err());
        assert!(greedy(&[1.0]).is_err());
    }
}
