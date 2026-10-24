#[path = "comparison.rs"]
mod comparison;
#[path = "io.rs"]
mod io;
#[path = "lifecycle.rs"]
mod lifecycle;
#[cfg(feature = "text-tokenizer")]
#[path = "plain_text.rs"]
mod plain_text;
#[path = "producers.rs"]
mod producers;
#[path = "reference.rs"]
mod reference;
#[path = "source.rs"]
mod source;
use io::*;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::{
    Error,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_model_gemma4::text::Gemma4ReferenceSession;
struct Arguments {
    source: PathBuf,
    expected: PathBuf,
    sha: String,
    report: PathBuf,
    candidate: bool,
    lifecycle: bool,
    text: Option<String>,
}
fn args() -> Proof<Arguments> {
    let mut a = std::env::args().skip(1);
    let mut source = None;
    let mut expected = None;
    let mut sha = None;
    let mut report = None;
    let mut arithmetic = None;
    let mut lifecycle = false;
    let mut text = None;
    while let Some(k) = a.next() {
        if k == "--lifecycle" && !lifecycle {
            lifecycle = true;
            continue;
        }
        let v = a.next().ok_or("missing value")?;
        match k.as_str() {
            "--source" if source.is_none() => source = Some(PathBuf::from(v)),
            "--reference" if expected.is_none() => expected = Some(PathBuf::from(v)),
            "--reference-sha256" if sha.is_none() => sha = Some(v),
            "--report" if report.is_none() => report = Some(PathBuf::from(v)),
            "--arithmetic" if arithmetic.is_none() => arithmetic = Some(v),
            "--plain-text" if text.is_none() => text = Some(v),
            _ => {
                return Err(
                    "--source ABS --reference INVOCATION --reference-sha256 SHA --report NEW_ABS"
                        .into(),
                );
            }
        }
    }
    let source = source.ok_or("source")?;
    let expected = expected.ok_or("reference")?;
    let sha = sha.ok_or("expected SHA")?;
    let report = report.ok_or("report")?;
    need(
        source.is_absolute() && expected.is_absolute() && report.is_absolute() && !report.exists(),
        "absolute actual source/reference/fresh report",
    )?;
    let dest = report
        .parent()
        .ok_or("report parent")?
        .canonicalize()?
        .join(report.file_name().ok_or("report name")?);
    need(
        !dest.starts_with(source.canonicalize()?)
            && !dest.starts_with(
                expected
                    .parent()
                    .ok_or("reference parent")?
                    .canonicalize()?,
            ),
        "report outside immutable source/reference",
    )?;
    let candidate =
        match arithmetic.as_deref() {
            None | Some("analytic-f32") => false,
            Some("bf16-candidate") => true,
            _ => return Err(
                "arithmetic must be analytic-f32 or bf16-candidate; neither is trained qualified"
                    .into(),
            ),
        };
    need(
        !lifecycle || candidate,
        "learned lifecycle requires explicit bf16-candidate",
    )?;
    need(
        text.is_none() || (cfg!(feature = "text-tokenizer") && candidate && !lifecycle),
        "plain text requires text-tokenizer +explicit bf16-candidate; lifecycle stays separate",
    )?;
    need(
        text.as_ref()
            .is_none_or(|s| !s.is_empty() && s.len() <= 4096),
        "bounded genuine plain text, no invented BOS",
    )?;
    Ok(Arguments {
        source,
        expected,
        sha,
        report: dest,
        candidate,
        lifecycle,
        text,
    })
}
pub(super) fn run() -> Proof<()> {
    let Arguments {
        source: directory,
        expected,
        sha,
        report,
        candidate,
        lifecycle,
        text,
    } = args()?;
    need(
        (!lifecycle && text.is_none()) || sha == lifecycle::REFERENCE_SHA256,
        "selected text/lifecycle reuses exact unchanged healthy all-call v2",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "system-ram")?;
    arbiter.register(resource.clone(), 128 * 1024 * 1024)?;
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("gemma4-e2b", "learned-text-diagnostic")?,
    };
    // No learned/reference owners coexist with the tokenizer's construction phase.
    #[cfg(feature = "text-tokenizer")]
    let prepared = text
        .as_deref()
        .map(|t| plain_text::Input::prepare(&directory, t, &host))
        .transpose()?;
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 8 * 1024 * 1024,
        }],
    )?;
    let reference = reference::Reference::open(&expected, &sha)?;
    if let Some(text) = &text {
        need(
            reference.capture["request"]["text"] == text.as_str(),
            "plain user text matches comparison workload; reference IDs are not inputs",
        )?;
    }
    let (model, held) = source::load(&directory, &host)?;
    #[cfg(feature = "text-tokenizer")]
    if let Some(prepared) = &prepared {
        prepared.require_model(&model)?;
    }
    let model = Arc::new(model);
    let producers = if candidate && !lifecycle && !reference.capture["producers"].is_null() {
        Some(producers::Producers::open(
            expected.parent().ok_or("reference directory")?,
            &reference.capture["producers"],
            &source::pins()?,
            model.source_identity(),
        )?)
    } else {
        None
    };
    if let Some(producers) = &producers {
        producers.require_call_count(
            reference.capture["calls"]
                .as_array()
                .ok_or("natural calls")?
                .len(),
        )?;
    }
    let mut session = if lifecycle {
        None
    } else if candidate {
        Some(Gemma4ReferenceSession::new_bf16_candidate(
            model.clone(),
            64,
        )?)
    } else {
        Some(Gemma4ReferenceSession::new(model.clone(), 66)?)
    };
    let initial = session.as_ref().map(Gemma4ReferenceSession::binding);
    let mut last_before = initial;
    let mut rows = Vec::new();
    let mut expected_callbacks = 0;
    let mut lifecycle_events = Vec::with_capacity(11);
    // Legacy ID-input diagnostics remain explicit. The text route uses only
    // authentic tokenizer output; expected IDs never become its operands.
    let diagnostic_ids = || {
        reference.capture["prompt_ids"]
            .as_array()
            .ok_or("actual prompt")?
            .iter()
            .map(|x| u32::try_from(x.as_u64().ok_or("token")?).map_err(Into::into))
            .collect::<Proof<Vec<_>>>()
    };
    #[cfg(feature = "text-tokenizer")]
    let mut current = if let Some(prepared) = &prepared {
        prepared.ids().to_vec()
    } else {
        diagnostic_ids()?
    };
    #[cfg(not(feature = "text-tokenizer"))]
    let mut current = diagnostic_ids()?;
    #[cfg(feature = "text-tokenizer")]
    let text_preparation = prepared.as_ref().map(plain_text::Input::report);
    #[cfg(not(feature = "text-tokenizer"))]
    let text_preparation: Option<serde_json::Value> = None;
    let result = if lifecycle {
        lifecycle::run(
            &model,
            &reference,
            expected.parent().ok_or("reference directory")?,
            &host,
            &mut lifecycle_events,
            &mut expected_callbacks,
        )
    } else {
        (|| -> Proof<()> {
            let session = session.as_mut().ok_or("actual reference session")?;
            for (call, expected) in reference.capture["calls"]
                .as_array()
                .ok_or("natural calls")?
                .iter()
                .enumerate()
            {
                let before = session.binding();
                last_before = Some(before);
                let mut callbacks = 0;
                let logits = session
                .forward_reference_observed(
                    &current,
                    true,
                    None,
                    producers.as_ref().map(|p| {
                            p as &dyn zerogravity_cpu::reference_text::ReferenceTextObserver
                        }),
                    |completed| {
                        callbacks += 1;
                        expected_callbacks += 1;
                        if let Some(producers) = &producers {
                            producers
                                .finish_call(call)
                                .map_err(|e| Error::State(e.to_string()))?;
                        }
                        let outputs_exact = reference
                            .compare(call, completed)
                            .map_err(|e| Error::State(e.to_string()))?;
                        let producers_exact = producers.as_ref().is_none_or(|p| p.exact_call(call));
                        if !outputs_exact || !producers_exact {
                            return Err(Error::State(format!(
                                "unqualified text arithmetic mismatch call={call} fullV_KV_exact={outputs_exact} producer_exact={producers_exact}"
                            )));
                        }
                        Ok(())
                    },
                )?
                .ok_or("complete real full vocabulary")?;
                need(
                    callbacks == 1
                        && session.binding().root() == before.root() + 1
                        && session.binding().frontier() == before.frontier() + current.len() as u64,
                    "original completion callback then one real commit",
                )?;
                let values = logits.as_slice()?;
                need(
                    values.len() == 262144 && values.iter().all(|x| x.is_finite()),
                    "actual complete finite vocabulary",
                )?;
                let selected = values
                    .iter()
                    .enumerate()
                    .fold(0, |best, (i, x)| if *x > values[best] { i } else { best })
                    as u32;
                need(
                    expected["selected_id"] == selected,
                    "actual Rust greedy independent of expected operands",
                )?;
                rows.push(
                json!({"call":call,"frontier":session.binding().frontier(),"selected_id":selected}),
            );
                current = vec![selected];
            }
            Ok(())
        })()
    };
    let final_binding = session.as_ref().map(Gemma4ReferenceSession::binding);
    drop(session);
    drop(model);
    // Full source verification is needed once after this actual run, including failed comparisons.
    #[cfg(feature = "text-tokenizer")]
    let prepared_unchanged = prepared.as_ref().map_or(Ok(()), plain_text::Input::after);
    #[cfg(not(feature = "text-tokenizer"))]
    let prepared_unchanged: Proof<()> = Ok(());
    let unchanged = prepared_unchanged
        .and_then(|_| source::after(&held))
        .and_then(|_| reference.after())
        .and_then(|_| {
            if let Some(producers) = &producers {
                producers.after()
            } else {
                Ok(())
            }
        });
    let producer_report = producers.as_ref().map(producers::Producers::report);
    let completed_report = reference.comparison_report();
    drop(producers);
    drop(held);
    drop(reference);
    #[cfg(feature = "text-tokenizer")]
    drop(prepared);
    drop(control);
    let final_reserved = host.arbiter.snapshot(&host.resource)?.reserved;
    let failed = result
        .as_ref()
        .err()
        .map(ToString::to_string)
        .or_else(|| unchanged.as_ref().err().map(ToString::to_string));
    write_new(
        &report,
        &json!({"schema":"zg.gemma4-e2b.rust-text-diagnostic.v1","source":source::pins()?,
        "text_preparation":text_preparation,
        "profile":if candidate {"Bf16EagerCandidate_UNQUALIFIED"}else{"AnalyticF32TextV1"},
        "producer_comparison":producer_report,"completed_comparison":completed_report,
        "reference_profile":"HfTorchCpuBf16Avx2Eager5.17.0",
        "initial_frontier":initial.map(|b| b.frontier()),"final_frontier":final_binding.map(|b| b.frontier()),"completed_calls":rows,
        "initial_binding":initial.map(|b| json!({"session":b.session().get(),"root":b.root(),"epoch":b.session_epoch(),"frontier":b.frontier()})),
        "final_binding":final_binding.map(|b| json!({"session":b.session().get(),"root":b.root(),"epoch":b.session_epoch(),"frontier":b.frontier()})),
        "failed_call_root_unchanged":(!lifecycle && result.is_err()).then_some(final_binding==last_before),
        "lifecycle_mode":lifecycle,"lifecycle_events":lifecycle_events,
        "expected_callbacks":expected_callbacks,
        "failed":failed,"source_unchanged":unchanged.is_ok(),"final_reserved":final_reserved,
        "trained_model_qualified":false,"native_serving_or_performance_qualified":false}),
    )?;
    result?;
    unchanged?;
    need(
        final_reserved == 0,
        "all original source/state/reference owners retired",
    )
}
