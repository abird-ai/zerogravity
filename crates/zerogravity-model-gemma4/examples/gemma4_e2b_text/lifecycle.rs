//! Selected learned lifecycle diagnostic; no publication/eligibility override.
use super::{io::*, producers::Producers, reference::Reference, source};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::{
    Error,
    accelerator::{SourceIdentity, program::KvDomain},
    session_root::RootBinding,
};
use zerogravity_cpu::reference_text::{
    ReferenceTextArithmetic, ReferenceTextObservation, ReferenceTextObserver,
    ReferenceTextOperation,
};
use zerogravity_model_gemma4::text::{Gemma4ReferenceSession, OriginalGemma4Text};

const COMPLETED_REFUSAL: &str = "deliberate learned completed proposal refusal";
const PRODUCER_REFUSAL: &str = "deliberate learned layer9 PV observer refusal";
const CANCELLED: &str = "reference text cancelled";
pub(super) const REFERENCE_SHA256: &str =
    "5c5917aa2485e06c0b5fa7a84ebdf2a3bff63b370edbd71fda3b397f8a1154af";

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    root: RootBinding,
    kv_sha256: Vec<[u8; 32]>,
}
impl Snapshot {
    fn of(session: &Gemma4ReferenceSession) -> Proof<Self> {
        let root = session.binding();
        need(root.frontier() <= 64, "selected diagnostic capacity")?;
        let mut kv_sha256 = Vec::with_capacity(15);
        for owner in 0..15 {
            let mut hash = Sha256::new();
            let width = if owner % 5 == 4 { 512 } else { 256 };
            for token in 0..root.frontier() as usize {
                let (key, value) = session.kv_row(KvDomain(owner), token)?;
                need(
                    key.len() == width && value.len() == width,
                    "actual complete KV owner geometry",
                )?;
                for x in key.iter().chain(value) {
                    need(x.is_finite(), "finite actual retained KV")?;
                    hash.update(x.to_bits().to_le_bytes());
                }
            }
            kv_sha256.push(hash.finalize().into());
        }
        Ok(Self { root, kv_sha256 })
    }
    fn report(&self) -> Value {
        let sha: Vec<_> = self
            .kv_sha256
            .iter()
            .map(|digest| {
                digest
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            })
            .collect();
        json!({"session":self.root.session().get(),"root":self.root.root(),
            "epoch":self.root.session_epoch(),"frontier":self.root.frontier(),"actual_kv_owner_sha256":sha})
    }
}
#[derive(Clone, Copy)]
enum Disposition {
    Commit,
    RejectCompleted,
    CancelCompleted,
}
struct Checks<'a> {
    reference: &'a Reference,
    directory: &'a Path,
    host: &'a TransferAdmission,
    source: SourceIdentity,
}
impl Checks<'_> {
    fn reserved(&self) -> Proof<u64> {
        Ok(self.host.arbiter.snapshot(&self.host.resource)?.reserved)
    }
    fn completed(
        &self,
        session: &mut Gemma4ReferenceSession,
        tokens: &[u32],
        call: usize,
        disposition: Disposition,
        events: &mut Vec<Value>,
        callbacks: &mut usize,
    ) -> Proof<Option<u32>> {
        // Fresh observer history for each actual attempt; immutable held bytes
        // and metadata are revalidated, never reset to hide a prior mismatch.
        let producers = Producers::open(
            self.directory,
            &self.reference.capture["producers"],
            &source::pins()?,
            self.source,
        )?;
        producers.require_lifecycle_scope()?;
        let before = Snapshot::of(session)?;
        let reserved = self.reserved()?;
        let cancelled = AtomicBool::new(false);
        let mut completed = 0;
        let result = session.forward_reference_observed(
            tokens,
            true,
            Some(&cancelled),
            Some(&producers),
            |actual| {
                completed += 1;
                *callbacks += 1;
                need(
                    actual.root_before() == before.root,
                    "actual callback from unchanged source root",
                )
                .map_err(|e| Error::State(e.to_string()))?;
                producers
                    .finish_call(call)
                    .map_err(|e| Error::State(e.to_string()))?;
                let outputs = self
                    .reference
                    .compare_completed_attempt(call, actual)
                    .map_err(|e| Error::State(e.to_string()))?;
                if !outputs || !producers.exact_call(call) {
                    return Err(Error::State(
                        "unqualified learned lifecycle arithmetic mismatch".into(),
                    ));
                }
                match disposition {
                    Disposition::Commit => Ok(()),
                    Disposition::RejectCompleted => Err(Error::State(COMPLETED_REFUSAL.into())),
                    Disposition::CancelCompleted => {
                        cancelled.store(true, Ordering::Release);
                        Ok(())
                    }
                }
            },
        );
        let after = Snapshot::of(session)?;
        let error = result.as_ref().err().map(ToString::to_string);
        let observed = producers.report_call(call);
        producers.after()?;
        let selected = match &result {
            Ok(Some(logits)) => {
                let values = logits.as_slice()?;
                need(
                    values.len() == 262144 && values.iter().all(|x| x.is_finite()),
                    "actual full finite vocabulary",
                )?;
                Some(
                    values
                        .iter()
                        .enumerate()
                        .fold(0, |best, (i, x)| if *x > values[best] { i } else { best })
                        as u32,
                )
            }
            _ => None,
        };
        events.push(json!({"kind":match disposition {Disposition::Commit=>"commit",Disposition::RejectCompleted=>"completed-refusal",Disposition::CancelCompleted=>"completed-cancel"},
            "call":call,"before":before.report(),"after":after.report(),"callbacks":completed,
            "selected_id":selected,"error":error,"producer_comparison":observed}));
        need(completed == 1, "one actual completed callback")?;
        match disposition {
            Disposition::Commit => {
                drop(result?);
                need(
                    after.root.session() == before.root.session()
                        && after.root.session_epoch() == before.root.session_epoch()
                        && after.root.root() == before.root.root() + 1
                        && after.root.frontier() == before.root.frontier() + tokens.len() as u64,
                    "one original complete publication",
                )?;
                need(selected.is_some(), "computed greedy feedback")?;
            }
            Disposition::RejectCompleted | Disposition::CancelCompleted => {
                let expected = if matches!(disposition, Disposition::RejectCompleted) {
                    COMPLETED_REFUSAL
                } else {
                    CANCELLED
                };
                need(
                    matches!(&result, Err(Error::State(why)) if why == expected),
                    "exact deliberate completed refusal; preserve unrelated failure",
                )?;
                need(
                    after == before && selected.is_none(),
                    "rejected proposal preserves actual root and all15 KV owners",
                )?;
                need(
                    self.reserved()? == reserved,
                    "rejected attempt has no retained output/proposal charge",
                )?;
            }
        }
        // Successful commits retain genuine new KV rows. The full session drop
        // below checks their retirement; failure requires exact prior charges.
        events.last_mut().ok_or("actual attempt event")?["reserved_before"] = json!(reserved);
        events.last_mut().ok_or("actual attempt event")?["reserved_after_output_drop"] =
            json!(self.reserved()?);
        Ok(selected)
    }
}
struct RefuseProducer {
    source: SourceIdentity,
    position: usize,
    seen: Cell<usize>,
}
impl ReferenceTextObserver for RefuseProducer {
    fn output(&self, event: ReferenceTextObservation<'_>) -> zerogravity_core::Result<()> {
        if event.source != self.source
            || event.arithmetic != ReferenceTextArithmetic::Bf16EagerCandidate
        {
            return Err(Error::State("learned observer source/arithmetic".into()));
        }
        if event.operation == ReferenceTextOperation::Boundary
            && event.layer == Some(9)
            && event.position == self.position
            && event.label == "attention.pv"
        {
            self.seen.set(self.seen.get() + 1);
            return Err(Error::State(PRODUCER_REFUSAL.into()));
        }
        Ok(())
    }
}
pub(super) fn run(
    model: &Arc<OriginalGemma4Text>,
    reference: &Reference,
    directory: &Path,
    host: &TransferAdmission,
    events: &mut Vec<Value>,
    callbacks: &mut usize,
) -> Proof<()> {
    need(
        reference.capture["calls"]
            .as_array()
            .is_some_and(|c| c.len() == 3)
            && reference.capture["prompt_ids"]
                .as_array()
                .is_some_and(|p| p.len() == 4),
        "selected healthy three-call expected-only trajectory",
    )?;
    let checks = Checks {
        reference,
        directory,
        host,
        source: model.source_identity(),
    };
    let prompt = reference.capture["prompt_ids"]
        .as_array()
        .ok_or("prompt")?
        .iter()
        .map(|id| u32::try_from(id.as_u64().ok_or("prompt token")?).map_err(Into::into))
        .collect::<Proof<Vec<_>>>()?;
    let baseline = checks.reserved()?;
    let mut a = Gemma4ReferenceSession::new_bf16_candidate(model.clone(), 64)?;
    let mut b = Gemma4ReferenceSession::new_bf16_candidate(model.clone(), 64)?;
    need(
        a.binding().session() != b.binding().session()
            && !a.trained_model_qualified()
            && !b.trained_model_qualified(),
        "independent actual source sessions without qualification widening",
    )?;
    let input_a = checks
        .completed(&mut a, &prompt, 0, Disposition::Commit, events, callbacks)?
        .ok_or("A greedy")?;
    let input_b = checks
        .completed(&mut b, &prompt, 0, Disposition::Commit, events, callbacks)?
        .ok_or("B greedy")?;
    let survivor = Snapshot::of(&b)?;
    let original = Snapshot::of(&a)?;
    let reserved = checks.reserved()?;
    let cancelled = AtomicBool::new(true);
    let mut pre_callbacks = 0;
    let result = a.forward_reference_with(&[input_a], true, Some(&cancelled), |_| {
        pre_callbacks += 1;
        Ok(())
    });
    need(
        matches!(result, Err(Error::State(why)) if why == CANCELLED)
            && pre_callbacks == 0
            && Snapshot::of(&a)? == original
            && checks.reserved()? == reserved,
        "pre-cancel has no proposal, callback, root or KV mutation",
    )?;
    events.push(json!({"kind":"pre-cancel","before":original.report(),"after":Snapshot::of(&a)?.report(),"callbacks":pre_callbacks}));
    let observer = RefuseProducer {
        source: model.source_identity(),
        position: original.root.frontier() as usize,
        seen: Cell::new(0),
    };
    let mut observed_callbacks = 0;
    let result = a.forward_reference_observed(&[input_a], true, None, Some(&observer), |_| {
        observed_callbacks += 1;
        Ok(())
    });
    need(
        matches!(result, Err(Error::State(why)) if why == PRODUCER_REFUSAL)
            && observer.seen.get() == 1
            && observed_callbacks == 0
            && Snapshot::of(&a)? == original
            && checks.reserved()? == reserved,
        "actual layer9 producer refusal rolls back the complete private proposal",
    )?;
    events.push(json!({"kind":"producer-refusal","layer":9,"producer":"attention.pv","observed":observer.seen.get(),
        "before":original.report(),"after":Snapshot::of(&a)?.report(),"callbacks":observed_callbacks}));
    checks.completed(
        &mut a,
        &[input_a],
        1,
        Disposition::RejectCompleted,
        events,
        callbacks,
    )?;
    checks.completed(
        &mut a,
        &[input_a],
        1,
        Disposition::CancelCompleted,
        events,
        callbacks,
    )?;
    need(
        Snapshot::of(&a)? == original && Snapshot::of(&b)? == survivor,
        "both original roots and all KV owners survive failed attempts",
    )?;
    let next_a = checks
        .completed(
            &mut a,
            &[input_a],
            1,
            Disposition::Commit,
            events,
            callbacks,
        )?
        .ok_or("A retry greedy")?;
    let next_b = checks
        .completed(
            &mut b,
            &[input_b],
            1,
            Disposition::Commit,
            events,
            callbacks,
        )?
        .ok_or("B survivor greedy")?;
    checks.completed(&mut a, &[next_a], 2, Disposition::Commit, events, callbacks)?;
    checks.completed(&mut b, &[next_b], 2, Disposition::Commit, events, callbacks)?;
    need(
        a.binding().root() == 3
            && b.binding().root() == 3
            && a.binding().frontier() == 6
            && b.binding().frontier() == 6
            && *callbacks == 8,
        "two independent natural three-call trajectories and two rejected completed attempts",
    )?;
    events.push(json!({"kind":"retire-sessions","a":Snapshot::of(&a)?.report(),"b":Snapshot::of(&b)?.report(),"completed_callbacks":*callbacks}));
    drop((a, b));
    need(
        checks.reserved()? == baseline,
        "all actual session/proposal/output owners retired",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn producer_fault_requires_actual_source_arithmetic_layer_position_and_boundary() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let observer = RefuseProducer {
            source,
            position: 4,
            seen: Cell::new(0),
        };
        let values = [0.0];
        let event = |layer, position, operation| ReferenceTextObservation {
            source,
            position,
            layer,
            arithmetic: ReferenceTextArithmetic::Bf16EagerCandidate,
            operation,
            label: "attention.pv",
            values: &values,
        };
        for (layer, position, operation) in [
            (Some(8), 4, ReferenceTextOperation::Boundary),
            (Some(9), 5, ReferenceTextOperation::Boundary),
            (Some(9), 4, ReferenceTextOperation::Pv),
        ] {
            observer.output(event(layer, position, operation)).unwrap();
        }
        assert_eq!(observer.seen.get(), 0);
        assert!(
            matches!(observer.output(event(Some(9), 4, ReferenceTextOperation::Boundary)), Err(Error::State(why)) if why == PRODUCER_REFUSAL)
        );
        assert_eq!(observer.seen.get(), 1);
        let mut wrong = event(Some(9), 4, ReferenceTextOperation::Boundary);
        wrong.arithmetic = ReferenceTextArithmetic::AnalyticF32;
        assert!(observer.output(wrong).is_err());
        let mut wrong = event(Some(9), 4, ReferenceTextOperation::Boundary);
        wrong.source = SourceIdentity::for_new_allocation().unwrap();
        assert!(observer.output(wrong).is_err());
        assert_eq!(observer.seen.get(), 1);
    }
}
