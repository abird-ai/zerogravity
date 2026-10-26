use super::{
    io::*,
    reference::{Expected, Matrix},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity::{
    Error,
    accelerator::program::KvDomain,
    cache::{CacheOptions, CacheStore},
    gemma4::execution::{Context, CpuSession, Model},
    session_root::RootBinding,
};
const MISMATCH: &str = "unqualified ordinary Gemma matrix arithmetic mismatch";
const REFUSAL: &str = "deliberate ordinary Gemma completed refusal";
#[derive(PartialEq, Eq)]
struct Snapshot {
    root: RootBinding,
    ids: Vec<u32>,
    kv: Vec<[u8; 32]>,
}
impl Snapshot {
    fn of(s: &CpuSession) -> Proof<Self> {
        let root = s.binding();
        let mut kv = Vec::with_capacity(15);
        for owner in 0..15 {
            let mut h = Sha256::new();
            for t in 0..root.frontier() as usize {
                let (k, v) = s.kv_row(KvDomain(owner), t)?;
                for x in k.iter().chain(v) {
                    h.update(x.to_bits().to_le_bytes());
                }
            }
            kv.push(h.finalize().into());
        }
        Ok(Self {
            root,
            ids: s.generated_ids().to_vec(),
            kv,
        })
    }
    fn report(&self) -> Value {
        json!({"root":self.root.root(),"frontier":self.root.frontier(),"session":self.root.session().get(),"ids":self.ids,"kv":self.kv.iter().map(|x|format!("{x:x?}")).collect::<Vec<_>>()})
    }
}
fn complete(s: &mut CpuSession, e: &Expected, call: usize, events: &mut Vec<Value>) -> Proof<bool> {
    e.begin(call);
    let before = Snapshot::of(s)?;
    let mut completed = 0;
    let mut comparison = None;
    let validate = |actual: zerogravity::gemma4::text::Gemma4ReferenceReadback<'_>| {
        completed += 1;
        let report = e
            .compare(call, actual)
            .map_err(|x| Error::State(x.to_string()))?;
        let exact = report["exact"] == true;
        comparison = Some(report);
        if exact {
            Ok(())
        } else {
            Err(Error::State(MISMATCH.into()))
        }
    };
    let result = if call == 0 {
        s.infer_prompt_observed(8, None, Some(e), validate)
    } else {
        s.infer_next_observed(None, Some(e), validate)
    };
    let after = Snapshot::of(s)?;
    events.push(json!({"kind":"actual-natural-call","before":before.report(),"after":after.report(),"callbacks":completed,"comparison":comparison,"error":result.as_ref().err().map(ToString::to_string)}));
    need(completed == 1, "one real completed proposal callback")?;
    match result {
        Ok(receipt) => {
            let c = &e.cell.calls[call];
            need(
                receipt.root == s.binding()
                    && receipt.root.root() == before.root.root() + 1
                    && receipt.root.frontier() == c.frontier as u64
                    && receipt.selected_id == c.selected_id
                    && receipt.eos == c.eos,
                "actual private greedy/root corresponds to independent source",
            )?;
            Ok(true)
        }
        Err(Error::State(why)) if why == MISMATCH => {
            need(
                before == after,
                "fatal numerical refusal leaves root, full committed KV and feedback unchanged",
            )?;
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}
fn lifecycle(
    s: &mut CpuSession,
    e: &Expected,
    context: &Arc<Context>,
    events: &mut Vec<Value>,
) -> Proof<()> {
    let before = Snapshot::of(s)?;
    let reserved = context.reserved_bytes()?;
    let flag = AtomicBool::new(true);
    let mut callbacks = 0;
    let result = s.infer_prompt_observed(8, Some(&flag), None, |_| {
        callbacks += 1;
        Ok(())
    });
    need(
        matches!(result,Err(Error::State(why))if why=="Gemma CPU request cancelled")
            && callbacks == 0
            && Snapshot::of(s)? == before
            && context.reserved_bytes()? == reserved,
        "pre-cancel leaves original facade root/state/feedback/owners unchanged",
    )?;
    events.push(json!({"kind":"pre-cancel","before":before.report(),"callbacks":callbacks}));
    for cancel in [false, true] {
        flag.store(false, Ordering::Release);
        e.begin(0);
        let mut callbacks = 0;
        let mut comparison = None;
        let result = s.infer_prompt_observed(8, Some(&flag), Some(e), |actual| {
            callbacks += 1;
            let report = e
                .compare(0, actual)
                .map_err(|x| Error::State(x.to_string()))?;
            let exact = report["exact"] == true;
            comparison = Some(report);
            if !exact {
                return Err(Error::State(MISMATCH.into()));
            }
            if cancel {
                flag.store(true, Ordering::Release);
                Ok(())
            } else {
                Err(Error::State(REFUSAL.into()))
            }
        });
        events.push(json!({"kind":if cancel{"completed-cancel"}else{"completed-refusal"},"comparison":comparison,"callbacks":callbacks,"error":result.as_ref().err().map(ToString::to_string)}));
        let expected = if cancel {
            "reference text cancelled"
        } else {
            REFUSAL
        };
        need(
            matches!(result,Err(Error::State(why))if why==expected)
                && callbacks == 1
                && Snapshot::of(s)? == before
                && context.reserved_bytes()? == reserved,
            "actual completed refusal/cancel preserves source root/KV/greedy plus all proposal owner charges",
        )?;
    }
    Ok(())
}
fn trajectory(s: &mut CpuSession, e: &Expected, events: &mut Vec<Value>) -> Proof<bool> {
    for call in 0..e.cell.calls.len() {
        if !complete(s, e, call, events)? {
            e.after()?;
            return Ok(false);
        }
    }
    need(s.stopped(), "actual EOS/limit, never forced extra tokens")?;
    let generated = e
        .cell
        .calls
        .iter()
        .filter(|c| !c.eos)
        .map(|c| c.selected_id)
        .collect::<Vec<_>>();
    need(
        s.generated_ids() == generated,
        "computed generated IDs, actual EOS excluded",
    )?;
    let decoded = s.decode_generated(true, None)?;
    events.push(json!({"kind":"source-ID-decoded-output","ids":s.generated_ids(),"text":decoded.as_str(),"sequence_decode_only":true}));
    drop(decoded);
    e.after()?;
    Ok(true)
}
pub fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        args.len() == 2,
        "usage: gemma4_cpu_matrix INVOCATION NEW_REPORT",
    )?;
    let invocation = PathBuf::from(&args[0]);
    let report = PathBuf::from(&args[1]);
    need(!report.exists(), "fresh actual result path")?;
    let matrix = Matrix::open(&invocation)?;
    let context = Context::new()?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let snapshot = Arc::new(
        store.open_snapshot("ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1")?,
    );
    // All genuine text inputs precede model loading; no reference IDs in operands.
    let mut inputs = Vec::with_capacity(8);
    let mut computed_inputs = Vec::with_capacity(7);
    for c in &matrix.request.cases {
        let input = context.prepare_text(&snapshot, &c.text, None)?;
        need(
            input.ids().len() == c.rows,
            "actual Rust tokenizer geometry",
        )?;
        computed_inputs.push(input.ids().to_vec());
        inputs.push(Some(input));
    }
    let extra = context.prepare_text(&snapshot, "One plus one is", None)?;
    need(
        matches!(
            context.prepare_text(&snapshot, "One", None),
            Err(Error::Limit(_))
        ),
        "ninth prepared source input refuses before tokenizer allocations",
    )?;
    let prepared_bytes = context.reserved_bytes()?;
    let loaded = Model::open(snapshot.clone(), context.clone(), None)?;
    need(
        !loaded.trained_model_qualified(),
        "candidate scope remains explicit",
    )?;
    let mut events = Vec::with_capacity(80);
    let mut exact = true;
    let result = (|| -> Proof<()> {
        let mut a = loaded.cpu_session(inputs[0].take().ok_or("actual input A")?)?;
        let mut b = loaded.cpu_session(inputs[1].take().ok_or("actual input B")?)?;
        need(
            a.binding().session() != b.binding().session(),
            "independent actual SessionAuthority issuers",
        )?;
        need(
            matches!(loaded.cpu_session(extra), Err(Error::Limit(_))),
            "third session pressure has no source/model duplication",
        )?;
        need(
            loaded.clone().close(None).is_err(),
            "live sessions forbid source close",
        )?;
        let survivor = Snapshot::of(&b)?;
        let e = matrix.cell(0, loaded.source_identity())?;
        need(
            computed_inputs[0] == e.cell.prompt_ids,
            "computed source input A compared only",
        )?;
        need(
            a.binding().frontier() == 0 && a.generated_ids().is_empty(),
            "fresh actual session",
        )?;
        need(
            a.generated_ids().is_empty() && inputs[0].is_none(),
            "opaque source preparation consumed only once",
        )?;
        exact &= trajectory(&mut a, &e, &mut events)?;
        drop(e);
        need(
            Snapshot::of(&b)? == survivor,
            "independent survivor unchanged during other actual trajectory",
        )?;
        let e = matrix.cell(1, loaded.source_identity())?;
        need(
            computed_inputs[1] == e.cell.prompt_ids,
            "computed source input B compared only",
        )?;
        exact &= trajectory(&mut b, &e, &mut events)?;
        drop(e);
        drop((a, b));
        for (i, slot) in inputs.iter_mut().enumerate().take(7).skip(2) {
            let input = slot.take().ok_or("actual next source text")?;
            let e = matrix.cell(i, loaded.source_identity())?;
            need(
                input.ids() == e.cell.prompt_ids,
                "computed input IDs compared only after preparation",
            )?;
            let mut s = loaded.cpu_session(input)?;
            if i == 3 {
                lifecycle(&mut s, &e, &context, &mut events)?;
            }
            exact &= trajectory(&mut s, &e, &mut events)?;
        }
        Ok(())
    })();
    drop(inputs);
    let closed = loaded.close(None);
    drop(snapshot);
    drop(store);
    let before_context_drop = context.reserved_bytes()?;
    drop(context);
    let final_bytes = zerogravity::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&zerogravity::resource_arbiter::PhysicalDeviceKey::new(
            "host",
            "system-ram",
        )?)?
        .reserved;
    write_new(
        &report,
        &json!({"schema":"zg.gemma4-e2b.ordinary-rust-matrix.v1","source_snapshot":"ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1","reference_invocation_sha256":sha(&read(&invocation,65536,None)?),"actual_text_prepared":true,"private_feedback_only":true,"cells_exact":exact&&result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"source_after":closed.is_ok(),"prepared_retained_bytes":prepared_bytes,"reserved_before_context_drop":before_context_drop,"final_reserved_bytes":final_bytes,"events":events,"trained_chat_native_http_or_performance_qualified":false}),
    )?;
    result?;
    closed?;
    need(
        before_context_drop == 3 * 1024 * 1024 && final_bytes == 0,
        "all actual model/source/input/state/decoder owners retired",
    )?;
    need(exact, "fatal independent full matrix numeric differences")?;
    println!("GEMMA4_CPU_ORDINARY_MATRIX_EXACT");
    Ok(())
}
