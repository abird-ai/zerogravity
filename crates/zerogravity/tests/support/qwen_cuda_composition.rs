//! Compact large-source composition check; CPU model agreement is not CUDA vendor parity.
use super::*;
use crate::text_invocations_reference::Expected;

fn state_digest(domains: &[(StateSlot, Vec<f32>)]) -> Result<String> {
    let mut hash = Sha256::new();
    for (slot, words) in domains {
        hash.update(format!("{slot:?}:{}", words.len()));
        for word in words {
            ensure(word.is_finite(), "finite composed state")?;
            hash.update(word.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[test]
#[ignore = "approved large source on owned CUDA; grouped heads, genuine prefill/decode, immutable prior"]
fn large_source_prefill_and_decode() -> Result<()> {
    ensure(!cfg!(debug_assertions), "release composition required")?;
    let source = env("ZG_QWEN35_MODEL_DIR")?;
    let reference = env("ZG_QWEN35_LARGE_REFERENCE")?;
    let arbiter = ResourceArbiter::process_shared();
    let host = PhysicalDeviceKey::new("host", "zerogravity-qwen35-cuda-ordinary")?;
    arbiter.register_matching(host.clone(), (qwen35_cuda::HOST_MIB * MIB) as u64)?;
    let _support = arbiter.admit(
        &host,
        ResourceOwner::new("cuda-composition", "reference-state")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 768 * MIB as u64,
        }],
    )?;
    let original = json_file(
        &reference.join("reference.json"),
        "f226ad211bc3ded7377046ef777186950ae3d2a0c3957449da7e5e5066f958cf",
        16 * MIB,
    )?;
    let expected = Expected::open_pinned(
        &reference,
        "Qwen/Qwen3.8-27B",
        "f226ad211bc3ded7377046ef777186950ae3d2a0c3957449da7e5e5066f958cf",
        None,
        false,
    )?;
    let calls = original["calls"]
        .as_array()
        .ok_or_else(|| fail("original calls"))?;
    block_on(async {
        let now = Instant::now();
        let model = Model::load(
            Device::cuda(0).map_err(|e| fail(e.to_string()))?,
            &source,
            1024,
            qwen35_cuda::HOST_MIB,
            qwen35_cuda::PROVIDER_MIB * MIB,
            (qwen35_cuda::DEVICE_MIB * MIB) as u64,
            None,
        )
        .await?;
        println!(
            "CUDA_COMPOSITION_LOADED elapsed_ms={} family={} groups={}/{}",
            now.elapsed().as_millis(),
            model.source_metadata().family,
            model.model_profile().key_heads(),
            model.model_profile().value_heads()
        );
        let result: Result<()> = async {
            ensure(original["repository"] == model.source_metadata().repository
                && original["revision"] == model.source_metadata().revision
                && model.model_profile().key_heads() == 16
                && model.model_profile().value_heads() == 48, "approved grouped source")?;
            let mut session = model.session().await?;
            let mut selected = None;
            for (step, call) in calls.iter().take(3).enumerate() {
                ensure(call["repeat"] == 0 && call["case"] == 0 && call["call"] == step,
                    "original first trajectory")?;
                let expected_call = expected
                    .calls()
                    .iter()
                    .find(|candidate| candidate.key() == (0, 0, step))
                    .ok_or_else(|| fail("original expected call"))?;
                let ids: Vec<u32> = call["input_ids"].as_array().ok_or_else(|| fail("original IDs"))?
                    .iter().map(|id| id.as_u64().and_then(|n| u32::try_from(n).ok())
                        .ok_or_else(|| fail("original token"))).collect::<Result<_>>()?;
                ensure(expected_call.input_ids == ids, "shared reference call identity")?;
                if let Some(previous) = selected { ensure(ids == [previous], "genuine greedy feedback")?; }
                let prior = session.snapshot();
                let before = state_digest(&prior.read_domains().await?)?;
                let now = Instant::now();
                let output = session.forward(if step == 0 { InvocationPhase::Prefill } else { InvocationPhase::Decode }, &ids, true).await?;
                ensure(output.logits.len() == model.model_profile().vocabulary_size()
                    && output.logits.iter().all(|x| x.is_finite()), "complete finite logits")?;
                let token = output.logits.iter().enumerate().fold((0, f32::NEG_INFINITY),
                    |best, (i, &x)| if x > best.1 { (i, x) } else { best }).0 as u32;
                selected = Some(token);
                ensure(before == state_digest(&prior.read_domains().await?)?, "prior state immutable")?;
                let state = session.snapshot().read_domains().await?;
                ensure(state.len() == model.model_profile().state_domains()
                    && state.iter().all(|(_, words)| words.iter().any(|&x| x != 0.0)), "all composed state domains populated")?;
                let state_hash = state_digest(&state)?;
                let metrics = expected.measure_cuda_cross_profile(
                    expected_call,
                    &output.logits,
                    &state,
                    model.context_capacity_tokens(),
                )?;
                ensure(
                    metrics.logits.actual_energy > 0.0
                        && metrics.state.actual_energy > 0.0
                        && session.binding().frontier() == (47 + step) as u64,
                    "frontier and cross-profile numerical energy",
                )?;
                println!("CUDA_COMPOSITION_CALL step={step} selected={token} expected={} domains={} prior_unchanged=true state_sha256={state_hash} cpu_logit_words={} cpu_logit_nmse={} cpu_logit_max_abs={} cpu_state_words={} cpu_state_nmse={} cpu_state_max_abs={} runtime_checked_kv_tail_words={} elapsed_ms={} cross_profile=true same_contract_reference=false vendor_bitwise_parity=false full_reference_acceptance=false",
                    call["selected"], state.len(), metrics.logits.words, metrics.logits.nmse(), metrics.logits.max_abs,
                    metrics.state.words, metrics.state.nmse(), metrics.state.max_abs, metrics.state_tail_words,
                    now.elapsed().as_millis());
                ensure(call["selected"] == token, "independent original greedy IDs")?;
            }
            Ok(())
        }.await;
        if let Err(error) = &result {
            println!("CUDA_COMPOSITION_FAILURE error={error:?}");
        }
        let closed = model.shutdown().await;
        if let Err(error) = &closed {
            println!("CUDA_COMPOSITION_CLOSE_ERROR error={error:?}");
        }
        println!(
            "CUDA_COMPOSITION_CLOSED closed={} model_ids_match={} vendor_bitwise_parity=false",
            closed.is_ok(),
            result.is_ok()
        );
        result.and(closed)
    })
}

#[cfg(feature = "serve")]
#[path = "qwen_cuda_serving.rs"]
mod serving;
