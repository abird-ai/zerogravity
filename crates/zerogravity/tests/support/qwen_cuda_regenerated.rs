//! Direct diagnostic against audited vLLM captures; no production capability grant.
#[path = "qwen_cuda_composition.rs"]
mod composition;
use super::{MIB, block_on, ensure, env, fail};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};
use zerogravity::{
    Result,
    accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, SourceCallMode, StateSlot},
    gpu::Device,
    qwen35_cuda::{self, forward_candidate::Model},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn bounded_file(path: &Path, bound: usize) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    ensure(
        metadata.is_file() && metadata.len() <= bound as u64,
        "bounded reference file extent",
    )?;
    let size = usize::try_from(metadata.len()).map_err(|_| fail("reference extent overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| fail("reference allocation refused"))?;
    bytes.resize(size, 0);
    file.read_exact(&mut bytes)?;
    ensure(file.read(&mut [0u8; 1])? == 0, "reference grew during read")?;
    Ok(bytes)
}

#[test]
fn reference_reader_bounds_files_before_allocation() -> Result<()> {
    let path = std::env::temp_dir().join(format!("zg-cuda-reference-bound-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let result = (|| {
        use std::io::Write;
        file.write_all(b"{}")?;
        ensure(bounded_file(&path, 2)? == b"{}", "exact bound")?;
        ensure(bounded_file(&path, 1).is_err(), "oversized input refused")?;
        ensure(
            json_file(&path, &digest(b"{}"), 2)? == serde_json::json!({}),
            "bounded JSON",
        )?;
        ensure(
            json_file(&path, &digest(b"[]"), 2).is_err(),
            "identity still required",
        )?;
        file.set_len(64 * MIB as u64)?;
        ensure(
            bounded_file(&path, MIB).is_err(),
            "sparse oversized input refused",
        )?;
        Ok(())
    })();
    drop(file);
    std::fs::remove_file(path)?;
    result
}

fn json_file(path: &Path, expected: &str, bound: usize) -> Result<Value> {
    let bytes = bounded_file(path, bound)?;
    ensure(
        bytes.len() <= bound && digest(&bytes) == expected,
        "audited reference identity",
    )?;
    serde_json::from_slice(&bytes).map_err(|e| fail(e.to_string()))
}
fn number(row: &Value, name: &str) -> Result<usize> {
    row[name]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| fail(format!("reference field {name}")))
}
fn raw(file: &mut File, row: &Value, expanded: bool) -> Result<Vec<u8>> {
    let bytes = number(row, if expanded { "data_bytes" } else { "bytes" })?;
    let offset = number(row, if expanded { "data_offset" } else { "offset" })?;
    ensure(
        bytes <= 2 * MIB
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end as u64 <= file.metadata().map_or(0, |m| m.len())),
        "reference raw extent",
    )?;
    file.seek(SeekFrom::Start(offset as u64))?;
    let mut data = vec![0; bytes];
    file.read_exact(&mut data)?;
    ensure(
        row["sha256"] == digest(&data),
        "reference selected raw hash",
    )?;
    Ok(data)
}
fn float(bytes: &[u8], index: usize, bf16: bool) -> f32 {
    if bf16 {
        f32::from_bits(
            u32::from(u16::from_le_bytes(
                bytes[index * 2..index * 2 + 2].try_into().unwrap(),
            )) << 16,
        )
    } else {
        f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
    }
}
fn state(slot: StateSlot) -> (usize, &'static str) {
    match slot {
        StateSlot::Conv(ConvDomain(layer)) => (layer, "conv"),
        StateSlot::Recurrent(RecurrentDomain(layer)) => (layer, "recurrent"),
        StateSlot::Kv(KvDomain(layer)) => (layer, "kv"),
    }
}
fn compare(expected: impl Iterator<Item = f32>, actual: &[f32]) -> Result<(usize, f32)> {
    let mut count = 0;
    let mut mismatch = 0;
    let mut max = 0.0f32;
    for (e, a) in expected.zip(actual) {
        ensure(
            e.is_finite() && a.is_finite(),
            "finite reference/native words",
        )?;
        count += 1;
        mismatch += usize::from(e.to_bits() != a.to_bits());
        max = max.max((e - a).abs());
    }
    ensure(count == actual.len(), "complete comparison extent")?;
    Ok((mismatch, max))
}

#[test]
#[ignore = "real-source diagnostic: approved owned CUDA process and audited independent vLLM capture"]
fn regenerated_vllm_numerics_and_state() -> Result<()> {
    compare_regenerated(false)
}

#[test]
#[ignore = "real-source diagnostic: bounded streaming versus the unchanged first prefill/two-decode reference"]
fn streamed_vllm_numerics_and_state() -> Result<()> {
    compare_regenerated(true)
}

fn compare_regenerated(streamed: bool) -> Result<()> {
    ensure(!cfg!(debug_assertions), "release diagnostic required")?;
    let source = env("ZG_QWEN35_MODEL_DIR")?;
    let reference = env("ZG_QWEN35_REGENERATED_REFERENCE")?;
    let report = env("ZG_QWEN35_REGENERATED_REVIEW")?;
    let hash =
        std::env::var("ZG_QWEN35_REGENERATED_REVIEW_SHA256").map_err(|e| fail(e.to_string()))?;
    ensure(
        matches!(
            hash.as_str(),
            "eefa1272e211f7f3ad949a24db136da844d90993c34fc009ca458a49d67016f8"
                | "0c492b7bd63efb129422fca66ab7c5aa2a2ba05a81191d204ecb031957c03443"
        ),
        "accepted consumer review",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let host = PhysicalDeviceKey::new("host", "zerogravity-qwen35-cuda-ordinary")?;
    arbiter.register_matching(host.clone(), (qwen35_cuda::HOST_MIB * MIB) as u64)?;
    let _support = arbiter.admit(
        &host,
        ResourceOwner::new("regenerated-cuda-diagnostic", "reference")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 64 * MIB as u64,
        }],
    )?;
    let review = json_file(&report, &hash, MIB)?;
    ensure(
        review["calls"] == 48 && review["foreground_group_container_eof_retired"] == true,
        "accepted consumer scope",
    )?;
    let vocab = json_file(
        &reference.join("full-vocabulary.json"),
        review["evidence_sha256"]["full-vocabulary.json"]
            .as_str()
            .ok_or_else(|| fail("manifest pin"))?,
        MIB,
    )?;
    let manifest = bounded_file(&reference.join("expanded.jsonl"), 32 * MIB)?;
    ensure(
        manifest.len() <= 32 * MIB
            && review["evidence_sha256"]["expanded.jsonl"] == digest(&manifest),
        "state manifest pin",
    )?;
    let mut states = BTreeMap::new();
    for line in BufReader::new(manifest.as_slice()).lines() {
        let row: Value = serde_json::from_str(&line?).map_err(|e| fail(e.to_string()))?;
        let m = &row["metadata"];
        if m["boundary"] != "after" || !m["label"].as_str().is_some_and(|s| s.starts_with("state."))
        {
            continue;
        }
        let key = (
            number(m, "step")?,
            number(m, "layer")?,
            m["domain"]
                .as_str()
                .ok_or_else(|| fail("state domain"))?
                .to_owned(),
        );
        ensure(
            states.insert(key, row).is_none(),
            "one captured logical page per short-call domain",
        )?;
    }
    drop(manifest);
    let records = vocab["records"]
        .as_array()
        .ok_or_else(|| fail("reference calls"))?;
    ensure(
        records.len() == 48 && states.len() == 48 * 42,
        "complete source call/state matrix",
    )?;
    let mut logits_file = File::open(reference.join("full-vocabulary.raw"))?;
    let mut state_file = File::open(reference.join("expanded.native.bin"))?;
    block_on(async {
        let started = Instant::now();
        let device = Device::cuda(0).map_err(|e| fail(e.to_string()))?;
        let model = Model::load(
            device,
            &source,
            1024,
            qwen35_cuda::HOST_MIB,
            if streamed {
                512 * MIB
            } else {
                qwen35_cuda::PROVIDER_MIB * MIB
            },
            (qwen35_cuda::DEVICE_MIB * MIB) as u64,
            None,
        )
        .await?;
        println!(
            "REGENERATED_CUDA_LOADED elapsed_ms={} family={} streamed={streamed}",
            started.elapsed().as_millis(),
            model.source_metadata().family
        );
        let result=async {
            ensure(model.source_metadata().repository==review["repo"] && model.source_metadata().revision==review["revision"],"native original source identity")?;
            for request in records.chunks_exact(3).take(if streamed { 1 } else { 16 }) {
                let mut session=model.session().await?;
                let mut feedback=None;
                for (call,r) in request.iter().enumerate() {
                    let step=number(r,"ordinal")?;
                    let ids=r["model_input_ids"].as_array().ok_or_else(||fail("original model IDs"))?.iter().map(|v|v.as_u64().and_then(|n|u32::try_from(n).ok()).ok_or_else(||fail("token ID"))).collect::<Result<Vec<_>>>()?;
                    if let Some(id)=feedback {ensure(ids==[id],"genuine native feedback agrees with reference history")?;}
                    let mode=if call==0 {SourceCallMode::Prefill}else{SourceCallMode::Decode};
                    let now=Instant::now();
                    let output=session.forward(mode,&ids,true).await?;
                    let frontier=session.binding().frontier() as usize;
                    ensure(output.logits.len()==248320 && number(r,"bytes")?==248320*2,"full native/reference vocabulary")?;
                    let bytes=raw(&mut logits_file,r,false)?;
                    let (lm,max)=compare((0..248320).map(|i|float(&bytes,i,true)),&output.logits)?;
                    let selected=output.logits.iter().enumerate().fold(0,|best,(i,v)|if *v>output.logits[best]{i}else{best}) as u32;
                    feedback=Some(selected);
                    let domains=session.snapshot().read_domains().await?;
                    ensure(domains.len()==42,"all native state domains")?;
                    let mut sm=0;
                    for (slot,actual) in domains {
                        let (layer,kind)=state(slot);
                        let row=states.get(&(step,layer,kind.to_owned())).ok_or_else(||fail("captured state missing"))?;
                        let bytes=raw(&mut state_file,row,true)?;
                        let (shape,count,bf16)=match kind {
                            "conv"=>(serde_json::json!([3,6144]),6144*3,true),
                            "recurrent"=>(serde_json::json!([16,128,128]),16*128*128,false),
                            _=>(serde_json::json!([2,544,512]),frontier*512*2,true),
                        };
                        ensure(row["source"]["shape"]==shape && actual.len()==count,"state geometry")?;
                        let values=(0..count).map(|i| {
                            let j=match kind {
                                "conv"=>(i%3)*6144+i/3,
                                "recurrent"=>(i/16384)*16384+(i%128)*128+(i/128)%128,
                                _=> {let p=i/(frontier*512);let t=(i%(frontier*512))/512;let h=(i%512)/256;h*544*512+t*512+p*256+i%256},
                            };
                            float(&bytes,j,bf16)
                        });
                        let (n,error)=compare(values,&actual)?; sm+=n;
                        if n!=0 {println!("REGENERATED_CUDA_STATE_DIFF step={step} layer={layer} domain={kind} mismatches={n} max_abs={error}");}
                    }
                    println!("REGENERATED_CUDA_CALL step={step} rows={} frontier={frontier} selected={selected} reference_selected={} logit_mismatches={lm} state_mismatches={sm} logit_max_abs={max} elapsed_ms={} full_reference_acceptance=false",ids.len(),r["selected"],now.elapsed().as_millis());
                    ensure(lm==0 && sm==0 && r["selected"]==selected,"current-vLLM numerical/state mismatch; diagnostic only")?;
                }
            }
            Ok(())
        }.await;
        if let Err(error) = &result {
            println!("REGENERATED_CUDA_FAILURE error={error:?}");
        }
        let closed = model.shutdown().await;
        if let Err(error) = &closed {
            println!("REGENERATED_CUDA_CLOSE_ERROR error={error:?}");
        }
        println!(
            "REGENERATED_CUDA_CLOSED closed={} numerical_match={} qualification=false",
            closed.is_ok(),
            result.is_ok()
        );
        result.and(closed)
    })
}

#[test]
#[ignore = "owned CUDA C1 benchmark; accepted inputs, no state/logit observer"]
fn regenerated_vllm_c1_benchmark() -> Result<()> {
    let source = env("ZG_QWEN35_MODEL_DIR")?;
    let reference = env("ZG_QWEN35_REGENERATED_REFERENCE")?;
    let report = env("ZG_QWEN35_REGENERATED_REVIEW")?;
    let hash =
        std::env::var("ZG_QWEN35_REGENERATED_REVIEW_SHA256").map_err(|e| fail(e.to_string()))?;
    ensure(!cfg!(debug_assertions), "release benchmark required")?;
    ensure(
        matches!(
            hash.as_str(),
            "eefa1272e211f7f3ad949a24db136da844d90993c34fc009ca458a49d67016f8"
                | "0c492b7bd63efb129422fca66ab7c5aa2a2ba05a81191d204ecb031957c03443"
        ),
        "accepted consumer review",
    )?;
    let review = json_file(&report, &hash, MIB)?;
    let path = reference.join("full-vocabulary.json");
    ensure(
        std::fs::metadata(&path)?.len() <= MIB as u64,
        "bounded benchmark manifest",
    )?;
    let manifest = json_file(
        &path,
        review["evidence_sha256"]["full-vocabulary.json"]
            .as_str()
            .ok_or_else(|| fail("manifest pin"))?,
        MIB,
    )?;
    let records = manifest["records"]
        .as_array()
        .ok_or_else(|| fail("reference calls"))?;
    ensure(records.len() == 48, "accepted request schedule")?;
    let cases = records
        .chunks_exact(3)
        .map(|r| {
            let ids = r[0]["model_input_ids"]
                .as_array()
                .ok_or_else(|| fail("prompt IDs"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or_else(|| fail("token ID"))
                })
                .collect::<Result<Vec<_>>>()?;
            let expected = r
                .iter()
                .map(|v| number(v, "selected").map(|n| n as u32))
                .collect::<Result<Vec<_>>>()?;
            Ok((ids, expected))
        })
        .collect::<Result<Vec<_>>>()?;
    block_on(async {
        let started = Instant::now();
        let device = Device::cuda(0).map_err(|e| fail(e.to_string()))?;
        let model = Model::load(
            device,
            &source,
            1024,
            qwen35_cuda::HOST_MIB,
            qwen35_cuda::PROVIDER_MIB * MIB,
            (qwen35_cuda::DEVICE_MIB * MIB) as u64,
            None,
        )
        .await?;
        println!(
            "C1_BENCH_LOAD {}",
            serde_json::json!({"engine":"native",
            "repo":model.source_metadata().repository,"load_ms":started.elapsed().as_secs_f64()*1000.0})
        );
        let result = async {
            ensure(
                model.source_metadata().repository == review["repo"]
                    && model.source_metadata().revision == review["revision"],
                "matched source",
            )?;
            let matrix = std::env::var_os("ZG_QWEN35_BENCH_MATRIX").is_some();
            let mut matched = true;
            for iteration in 0..if matrix { 8 } else { 24 } {
                let warmup = iteration < 8;
                let case = if warmup { iteration } else { iteration - 8 };
                let (prompt, expected) = &cases[case];
                let now = Instant::now();
                let mut session = model.session().await?;
                let mut ids = prompt.clone();
                let mut generated = Vec::with_capacity(3);
                let mut token_ms = Vec::with_capacity(3);
                for step in 0..3 {
                    let mode = if step == 0 {
                        SourceCallMode::Prefill
                    } else {
                        SourceCallMode::Decode
                    };
                    let output = session.forward(mode, &ids, true).await?;
                    ensure(output.logits.len() == 248320, "full output vocabulary")?;
                    let selected = output.logits.iter().enumerate().fold(0, |best, (i, v)| {
                        if *v > output.logits[best] { i } else { best }
                    }) as u32;
                    generated.push(selected);
                    token_ms.push(now.elapsed().as_secs_f64() * 1000.0);
                    ids = vec![selected];
                }
                drop(session);
                let elapsed_ms = now.elapsed().as_secs_f64() * 1000.0;
                let output_match = generated == *expected;
                matched &= output_match;
                println!(
                    "C1_BENCH_REQUEST {}",
                    serde_json::json!({"engine":"native",
                    "case":case,"warmup":warmup,"prompt_tokens":prompt.len(),
                    "token_ids":generated,"expected":expected,"output_match":output_match,
                    "elapsed_ms":elapsed_ms,"token_ms":token_ms})
                );
            }
            ensure(matched, "benchmark generated IDs match accepted reference")?;
            if matrix {
                benchmark_populations(&model, &cases)?;
            }
            Ok(())
        }
        .await;
        let closed = model.shutdown().await;
        println!(
            "C1_BENCH_CLOSED closed={} output_match={}",
            closed.is_ok(),
            result.is_ok()
        );
        closed?;
        result
    })
}

// Direct clients sharing the existing physical queue; not fused native batches.
fn benchmark_populations(
    model: &std::sync::Arc<Model>,
    cases: &[(Vec<u32>, Vec<u32>)],
) -> Result<()> {
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    let service = model.resident().qualification();
    let state_bytes = model.resident().state_bytes();
    for clients in [1usize, 4, 8, 16, 32, 64] {
        let required = 2 * clients * state_bytes;
        let available = service.remaining_capacity()?;
        if required > available {
            println!(
                "POP_BENCH_REFUSED {}",
                serde_json::json!({"engine":"native", "clients":clients,
                "reason":"independent prior/candidate roots exceed unchanged provider cap",
                "required_state_bytes":required,"available_bytes":available})
            );
            continue;
        }
        let measured_waves = 16usize.div_ceil(clients).max(2);
        let mut all_match = true;
        for wave in 0..=measured_waves {
            let warmup = wave == 0;
            let barrier = Barrier::new(clients + 1);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let (rows, elapsed_ms) = std::thread::scope(|scope| -> Result<_> {
                let mut handles = Vec::with_capacity(clients);
                for client in 0..clients {
                    let case = ((wave.saturating_sub(1)) * clients + client) % cases.len();
                    let (prompt, expected) = &cases[case];
                    let barrier = &barrier;
                    let active = &active;
                    let peak = &peak;
                    handles.push(scope.spawn(move || -> Result<serde_json::Value> {
                        barrier.wait();
                        block_on(async {
                            let start = Instant::now();
                            let mut session = model.session().await?;
                            let session_id = session.binding().session().get();
                            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(current, Ordering::SeqCst);
                            let mut ids = prompt.clone();
                            let mut generated = Vec::with_capacity(3);
                            let mut token_ms = Vec::with_capacity(3);
                            let result = async {
                                for step in 0..3 {
                                    let mode = if step == 0 { SourceCallMode::Prefill } else { SourceCallMode::Decode };
                                    let output = session.forward(mode, &ids, true).await?;
                                    ensure(output.logits.len() == 248320, "full vocabulary")?;
                                    let selected = output.logits.iter().enumerate().fold(0, |best, (i, v)|
                                        if *v > output.logits[best] { i } else { best }) as u32;
                                    generated.push(selected);
                                    token_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                                    ids = vec![selected];
                                }
                                ensure(session.binding().frontier() as usize == prompt.len() + 2, "native frontier")
                            }.await;
                            drop(session);
                            active.fetch_sub(1, Ordering::SeqCst);
                            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                            result?;
                            Ok(serde_json::json!({"case":case,"session_id":session_id,
                                "prompt_tokens":prompt.len(),"token_ids":generated,"expected":expected,
                                "output_match":generated==*expected,"elapsed_ms":elapsed_ms,"token_ms":token_ms}))
                        })
                    }));
                }
                let start = Instant::now();
                barrier.wait();
                // Join every owned client even if one fails; shutdown follows outside this scope.
                let results = handles
                    .into_iter()
                    .map(|h| h.join().map_err(|_| fail("benchmark client panicked"))?)
                    .collect::<Result<Vec<_>>>()?;
                Ok((results, start.elapsed().as_secs_f64() * 1000.0))
            })?;
            ensure(
                active.load(Ordering::SeqCst) == 0,
                "all native clients retired",
            )?;
            let unique = rows
                .iter()
                .map(|r| r["session_id"].as_u64().unwrap())
                .collect::<std::collections::BTreeSet<_>>();
            ensure(unique.len() == clients, "independent request sessions")?;
            all_match &= rows.iter().all(|r| r["output_match"] == true);
            println!(
                "POP_BENCH_WAVE {}",
                serde_json::json!({"engine":"native","clients":clients,
                "wave":wave,"warmup":warmup,"elapsed_ms":elapsed_ms,"completed":rows.len(),
                "output_tokens":3*rows.len(),"peak_active_sessions":peak.load(Ordering::SeqCst),
                "native_batch_width":1,"requests":rows})
            );
        }
        ensure(all_match, "concurrent benchmark output IDs")?;
    }
    Ok(())
}
