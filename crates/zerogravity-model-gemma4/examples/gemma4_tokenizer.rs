//! Selected source tokenizer comparison + admitted preparation retirement.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::{
    Error,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_model_gemma4::text::tokenizer::{
    CONSTRUCTION_BYTES, CONTROL_BYTES, Gemma4TextTokenizer, Gemma4TokenizerFiles, RETAINED_BYTES,
};
type Proof<T> = Result<T, Box<dyn std::error::Error>>;
struct Tracked;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let old = LIVE.fetch_add(layout.size(), Ordering::SeqCst);
        if old.saturating_add(layout.size()) > LIMIT.load(Ordering::SeqCst) {
            LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        } else {
            PEAK.fetch_max(old + layout.size(), Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}
#[global_allocator]
static ALLOCATOR: Tracked = Tracked;
fn need(ok: bool, why: &str) -> Proof<()> {
    if ok { Ok(()) } else { Err(why.into()) }
}
fn open(path: &Path) -> Proof<File> {
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(0x20000 | 0x80000 | 0x800)
        .open(path)?)
}
fn files(dir: &Path) -> Proof<Gemma4TokenizerFiles> {
    Ok(Gemma4TokenizerFiles {
        tokenizer: open(&dir.join("tokenizer.json"))?,
        tokenizer_config: open(&dir.join("tokenizer_config.json"))?,
        generation_config: open(&dir.join("generation_config.json"))?,
        model_config: open(&dir.join("config.json"))?,
    })
}
fn json(path: &Path) -> Proof<(Value, String)> {
    let mut f = open(path)?;
    let before = f.metadata()?;
    need(
        before.is_file() && before.len() <= 65536,
        "bounded complete tokenizer metadata",
    )?;
    let mut raw = Vec::new();
    Read::by_ref(&mut f).take(65537).read_to_end(&mut raw)?;
    need(
        raw.len() as u64 == before.len(),
        "complete tokenizer metadata EOF",
    )?;
    Ok((
        serde_json::from_slice(&raw)?,
        format!("{:x}", Sha256::digest(raw)),
    ))
}
fn args() -> Proof<(PathBuf, PathBuf, String, PathBuf)> {
    let mut a = std::env::args().skip(1);
    let (mut source, mut original, mut sha, mut report) = (None, None, None, None);
    while let Some(k) = a.next() {
        let v = a.next().ok_or("argument value")?;
        match k.as_str() {
            "--source" if source.is_none() => source = Some(PathBuf::from(v)),
            "--reference" if original.is_none() => original = Some(PathBuf::from(v)),
            "--reference-sha256" if sha.is_none() => sha = Some(v),
            "--report" if report.is_none() => report = Some(PathBuf::from(v)),
            _ => return Err("source/reference/SHA/fresh report only".into()),
        }
    }
    let (s, o, h, r) = (
        source.ok_or("source")?,
        original.ok_or("original")?,
        sha.ok_or("expected-only pin")?,
        report.ok_or("report")?,
    );
    need(
        s.is_absolute() && o.is_absolute() && r.is_absolute() && !r.exists(),
        "absolute source/original/fresh report",
    )?;
    let dest = r
        .parent()
        .ok_or("report parent")?
        .canonicalize()?
        .join(r.file_name().ok_or("report name")?);
    need(
        !dest.starts_with(s.canonicalize()?)
            && !dest.starts_with(o.parent().ok_or("original parent")?.canonicalize()?),
        "report outside actual source/reference",
    )?;
    Ok((s, o, h, dest))
}
fn run() -> Proof<()> {
    let (source, original, pin, report) = args()?;
    let (receipt, receipt_sha) = json(&original)?;
    need(
        receipt_sha == pin
            && receipt["status"] == "EXECUTED_ORIGINAL_GEMMA4_TOKENIZER_EXPECTED_ONLY"
            && receipt["exit_code"] == 0
            && receipt["source_before"] == receipt["source_after"]
            && receipt["source_FDs_closed"] == true
            && receipt["cleanup_errors"] == json!([])
            && receipt["log_cleanup"]["pipe_eof"] == true
            && receipt["log_cleanup"]["all_closed"] == true
            && receipt["foreground_cleanup"]["leader_reaped"] == true
            && receipt["foreground_cleanup"]["group_absent"] == true
            && receipt["owned_container_absent_after_reap"] == true,
        "healthy original tokenizer/source/owner closure",
    )?;
    let (capture, capture_sha) = json(
        &original
            .parent()
            .ok_or("reference parent")?
            .join("capture.json"),
    )?;
    let expected: Value = serde_json::from_slice(include_bytes!(
        "../../../tools/gemma4-tokenizer-reference/cases.json"
    ))?;
    let source_pins: Value = serde_json::from_slice(include_bytes!(
        "../../../tools/gemma4-tokenizer-reference/source-pin.json"
    ))?;
    need(
        receipt["capture"]["sha256"] == capture_sha
            && capture["source"] == source_pins
            && capture["request"] == expected
            && capture["schema"] == "zg.gemma4-e2b.original-tokenizer.v2"
            && capture["repetitions"] == 2
            && capture["repetition_equal"] == true
            && capture["model_constructed"] == false
            && capture["weights_verified"] == false
            && capture["trained_rust_qualified"] == false,
        "entire actual original tokenizer scope; no weight/math claim",
    )?;
    let host = TransferAdmission {
        arbiter: ResourceArbiter::new(),
        resource: PhysicalDeviceKey::new("host", "Gemma tokenizer")?,
        owner: ResourceOwner::new("Gemma E2B", "text preparation")?,
    };
    host.arbiter
        .register(host.resource.clone(), 128 * 1024 * 1024)?;
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 4 * 1024 * 1024,
        }],
    )?;
    let reserved = || host.arbiter.snapshot(&host.resource).map(|x| x.reserved);
    let baseline = reserved()?;
    let flag = AtomicBool::new(true);
    need(
        matches!(
            Gemma4TextTokenizer::prepare(files(&source)?, &host, Some(&flag)),
            Err(Error::State(_))
        ) && reserved()? == baseline,
        "pre-cancel returns all actual source/charge owners",
    )?;
    let pressure = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 1024 * 1024,
        }],
    )?;
    need(
        matches!(
            Gemma4TextTokenizer::prepare(files(&source)?, &host, None),
            Err(Error::Limit(_))
        ),
        "real shared capacity refuses before parse; no cap change",
    )?;
    drop(pressure);
    need(
        reserved()? == baseline,
        "pressure/refused constructor retired",
    )?;
    let heap_before = LIVE.load(Ordering::SeqCst);
    PEAK.store(heap_before, Ordering::SeqCst);
    LIMIT.store(
        heap_before + (CONSTRUCTION_BYTES + CONTROL_BYTES) as usize,
        Ordering::SeqCst,
    );
    let constructed = Gemma4TextTokenizer::prepare(files(&source)?, &host, None);
    LIMIT.store(usize::MAX, Ordering::SeqCst);
    let tokenizer = constructed?;
    let heap_peak = PEAK.load(Ordering::SeqCst).saturating_sub(heap_before);
    let heap_retained = LIVE.load(Ordering::SeqCst).saturating_sub(heap_before);
    need(
        heap_peak as u64 <= CONSTRUCTION_BYTES + CONTROL_BYTES
            && heap_retained as u64 <= RETAINED_BYTES,
        "actual selected allocator construction/retained bounds",
    )?;
    let mut comparisons = Vec::new();
    let mut actual_sequences = Vec::new();
    let actual_rows = capture["rows"].as_array().ok_or("original rows")?;
    let requested = expected["cases"].as_array().ok_or("request cases")?;
    need(
        actual_rows.len() == requested.len() && requested.len() <= 64,
        "every declared original tokenizer case",
    )?;
    for (actual, case) in actual_rows.iter().zip(requested) {
        need(
            actual["name"] == case["name"]
                && actual["text"] == case["text"]
                && actual["ids"] == actual["without_special"],
            "same original text/identity single postprocessor",
        )?;
        let text = case["text"].as_str().ok_or("Unicode text")?;
        let encoded = tokenizer.encode_plain(text, None)?;
        let decoded = tokenizer.decode(encoded.ids(), false, None)?;
        let skipped = tokenizer.decode(encoded.ids(), true, None)?;
        need(
            json!(encoded.ids()) == actual["ids"]
                && Some(decoded.as_str()) == actual["decode"].as_str()
                && Some(skipped.as_str()) == actual["decode_skip"].as_str(),
            "strict original IDs and both complete decoder modes",
        )?;
        actual_sequences.push(encoded.ids().to_vec());
        comparisons.push(json!({"name":case["name"],"ids":encoded.ids(),"decode_exact":true,"decode_skip_exact":true}));
    }
    let mut decode_comparisons = Vec::new();
    let actual_decodes = capture["decodes"].as_array().ok_or("original decodes")?;
    let decode_requests = expected["decode_tokens"]
        .as_array()
        .ok_or("decode declarations")?;
    need(
        actual_decodes.len() == decode_requests.len() && decode_requests.len() <= 64,
        "entire independent byte-fallback cohort",
    )?;
    for (actual, case) in actual_decodes.iter().zip(decode_requests) {
        need(
            actual["name"] == case["name"] && actual["tokens"] == case["tokens"],
            "same declared source token names",
        )?;
        let pieces = case["tokens"].as_array().ok_or("source token names")?;
        need(
            !pieces.is_empty() && pieces.len() <= 64,
            "bounded source token sequence",
        )?;
        // Resolve from the Rust source vocabulary; expected IDs are compared
        // afterwards and never supply a decoder operand.
        let ids = pieces
            .iter()
            .map(|piece| {
                tokenizer
                    .token_id(piece.as_str().ok_or("source piece text")?)
                    .map_err(Into::into)
            })
            .collect::<Proof<Vec<_>>>()?;
        actual_sequences.push(ids.clone());
        let decoded = tokenizer.decode(&ids, false, None)?;
        let skipped = tokenizer.decode(&ids, true, None)?;
        need(
            json!(ids) == actual["ids"]
                && Some(decoded.as_str()) == actual["decode"].as_str()
                && Some(skipped.as_str()) == actual["decode_skip"].as_str(),
            "exact original invalid/incomplete UTF8 and special-token decoder behavior",
        )?;
        decode_comparisons.push(
            json!({"name":case["name"],"ids":ids,"decode":decoded.as_str(),
            "decode_skip":skipped.as_str(),"both_modes_exact":true}),
        );
    }
    need(
        tokenizer.encode_plain(&"x".repeat(4097), None).is_err()
            && tokenizer.decode(&[262144], false, None).is_err(),
        "source/input/output bounds refuse",
    )?;
    need(
        tokenizer.encode_plain("One", Some(&flag)).is_err(),
        "actual tokenizer pre-cancel",
    )?;
    tokenizer.verify_source(None)?;
    let prepared = tokenizer.into_plain_text("One plus one is", None)?;
    need(
        json!(prepared.ids()) == capture["rows"][0]["ids"] && reserved()? == baseline + 65536,
        "genuine text IDs survive; tokenizer and all construction charges retired",
    )?;
    let expected_sequences = capture["rows"]
        .as_array()
        .ok_or("source text decodes")?
        .iter()
        .chain(capture["decodes"].as_array().ok_or("source byte decodes")?);
    need(
        actual_sequences.len()
            == capture["rows"].as_array().ok_or("source text count")?.len()
                + capture["decodes"]
                    .as_array()
                    .ok_or("source byte count")?
                    .len(),
        "selected decoder must consume every text and byte-fallback source case",
    )?;
    for (ids, expected) in actual_sequences.iter().zip(expected_sequences) {
        // Only the earlier COMPUTED Rust IDs are decoder operands.
        let decoded = prepared.decode_output(ids, false, None)?;
        let skipped = prepared.decode_output(ids, true, None)?;
        need(
            Some(decoded.as_str()) == expected["decode"].as_str()
                && Some(skipped.as_str()) == expected["decode_skip"].as_str(),
            "bounded selected source decoder exact against independent original in both modes",
        )?;
    }
    prepared.verify_source(None)?;
    let ids = prepared.ids().to_vec();
    drop(prepared);
    need(
        reserved()? == baseline,
        "prepared IDs/source owners retired",
    )?;
    drop(control);
    need(
        reserved()? == 0,
        "final original resource reservations zero",
    )?;
    let data = json!({"schema":"zg.gemma4-e2b.rust-tokenizer-diagnostic.v1","reference_invocation_sha256":pin,
        "source":source_pins,"cases":comparisons,"byte_fallback_decodes":decode_comparisons,"actual_constructor_heap_peak_bytes":heap_peak,"actual_retained_heap_bytes":heap_retained,
        "heap_accounting":"Rust GlobalAlloc requested sizes, not process RSS; hard phase limit excludes preexisting caller control",
        "construction_admission_bytes":CONSTRUCTION_BYTES,"control_bytes":CONTROL_BYTES,"retained_bytes":RETAINED_BYTES,
        "actual_prepared_ids":ids,"tokenizer_retired_before_model":true,"selected_decoder_cases_exact":actual_sequences.len(),"model_loaded":false,"final_reserved":reserved()?,
        "trained_model_qualified":false,"native_serving_or_performance_qualified":false});
    let raw = serde_json::to_vec_pretty(&data)?;
    need(raw.len() <= 65536, "bounded complete tokenizer report")?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)?;
    out.write_all(&raw)?;
    out.write_all(b"\n")?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("Gemma tokenizer diagnostic refused: {error}");
        std::process::exit(1)
    }
}
