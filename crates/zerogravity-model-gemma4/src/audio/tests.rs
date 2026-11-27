use super::*;
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::Ordering;
use zerogravity_artifacts::cache::{CacheOptions, CacheStore};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

fn profile() -> (serde_json::Value, serde_json::Value) {
    (
        json!({"model_type":"gemma4","text_config":{"hidden_size":4},"audio_config":{
            "model_type":"gemma4_audio","hidden_size":4,"num_hidden_layers":1,
            "num_attention_heads":2,"output_proj_dims":4,"subsampling_conv_channels":[4,2],
            "attention_chunk_size":12,"attention_context_left":13,"attention_context_right":0,
            "attention_logit_cap":50.0,"attention_invalid_logits_value":-1e9,
            "conv_kernel_size":5,"hidden_act":"silu","residual_weight":0.5,
            "gradient_clipping":1e10,"rms_norm_eps":1e-6,"use_clipped_linears":true
        }}),
        json!({"audio_seq_length":4,"audio_ms_per_token":40,"feature_extractor":{
            "feature_extractor_type":"Gemma4AudioFeatureExtractor","feature_size":4,
            "sampling_rate":16000,"frame_length":320,"hop_length":160,"fft_length":512,
            "fft_overdrive":false,"dither":0.0,"preemphasis":0.0,"input_scale_factor":1.0,
            "min_frequency":0.0,"max_frequency":8000.0,"mel_floor":0.001,
            "padding_side":"right","padding_value":0.0,"per_bin_mean":null,"per_bin_stddev":null,
            "return_attention_mask":true
        }}),
    )
}
fn parsed(config: &serde_json::Value, processor: &serde_json::Value) -> Result<Config> {
    Config::parse(
        &serde_json::to_vec(config).unwrap(),
        &serde_json::to_vec(processor).unwrap(),
    )
}
#[test]
fn source_profile_and_unpadded_silence_frontend() {
    let (config, processor) = profile();
    let c = parsed(&config, &processor).unwrap();
    let f = MelFrontend::new(&c.frontend).unwrap();
    assert_eq!(c.max_samples, 2560);
    assert!(MelFrontend::frames(&c.frontend, 160).is_err());
    assert_eq!(MelFrontend::frames(&c.frontend, 161).unwrap(), 1);
    let frames = MelFrontend::frames(&c.frontend, 321).unwrap();
    assert_eq!(frames, 2);
    let features = f.extract(&c.frontend, &[0.; 321], frames, None).unwrap();
    assert_eq!(features.len(), 8);
    assert!(features.iter().all(|&v| (v - 0.001f32.ln()).abs() < 1e-6));
    let mut unsupported = config.clone();
    unsupported["audio_config"]["attention_context_right"] = json!(1);
    assert!(parsed(&unsupported, &processor).is_err());
    let mut unsupported = processor.clone();
    unsupported["feature_extractor"]["preemphasis"] = json!(0.97);
    assert!(parsed(&config, &unsupported).is_err());
}

#[derive(Deserialize)]
struct Case {
    samples: Vec<f32>,
    frames: usize,
    features: Vec<f32>,
    tokens: usize,
    width: usize,
    values: Vec<f32>,
}
fn reserved(a: &TransferAdmission) -> u64 {
    a.arbiter.snapshot(&a.resource).unwrap().reserved
}
fn compare(label: &str, actual: &[f32], reference: &[f32], absolute: f64, relative: f64) {
    assert_eq!(actual.len(), reference.len());
    let mut max = 0.0f64;
    let mut squared = 0.0f64;
    let mut norm = 0.0f64;
    let mut violations = 0;
    for (&a, &r) in actual.iter().zip(reference) {
        let (a, r) = (f64::from(a), f64::from(r));
        assert!(a.is_finite() && r.is_finite());
        let error = (a - r).abs();
        max = max.max(error);
        squared += error * error;
        norm += r * r;
        if error > absolute + relative * r.abs() {
            violations += 1;
        }
    }
    let rel_l2 = (squared / norm.max(1e-30)).sqrt();
    println!(
        "{label}: values={} max_abs={max:.9} relative_l2={rel_l2:.9} violations={violations}",
        actual.len()
    );
    assert_eq!(violations, 0, "{label} exceeds coordinate tolerance");
    assert!(rel_l2 <= relative, "{label} exceeds relative L2 tolerance");
}

fn fixture() -> (
    Vec<Case>,
    zerogravity_artifacts::cache::SnapshotLease,
    TransferAdmission,
) {
    let reference = std::env::var_os("ZG_GEMMA4_AUDIO_REFERENCE").expect("reference JSON path");
    let digest = std::env::var("ZG_GEMMA4_AUDIO_SNAPSHOT").expect("original cache snapshot digest");
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    assert_eq!(cases.len(), 2);
    let store = CacheStore::open_existing(CacheOptions::new(
        std::env::var_os("ZG_GEMMA4_AUDIO_CACHE").map(Into::into),
        std::env::var("ZG_GEMMA4_AUDIO_CACHE_BYTES")
            .expect("existing cache byte policy")
            .parse()
            .unwrap(),
    ))
    .unwrap();
    let lease = store.open_snapshot(&digest).unwrap();
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "gemma4-audio-reference").unwrap();
    arbiter
        .register_matching(resource.clone(), 16 << 30)
        .unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("gemma4", "audio-reference").unwrap(),
    };
    (cases, lease, admission)
}

/// Real original learned source, not a synthetic oracle. Reference script
/// runs the installed Transformers CPU model with BF16 coefficients widened
/// to F32, exact frontend and audio-to-language projection.
#[test]
#[ignore = "requires original cached Gemma4 audio checkpoint and independent offline reference"]
fn actual_original_audio_tower_matches_transformers_and_retires() {
    let (cases, lease, admission) = fixture();
    let baseline = reserved(&admission);
    let start = std::time::Instant::now();
    let model = Gemma4AudioEncoder::load(&lease, 11 << 30, &admission).unwrap();
    println!(
        "AUDIO_SOURCE_BOUND elapsed={:.3}s retained_bytes={}",
        start.elapsed().as_secs_f64(),
        reserved(&admission) - baseline
    );
    assert!(model.source().is_bound());
    assert!(
        reserved(&admission) - baseline < 128 * 1024 * 1024,
        "window source must not retain the whole 10GB checkpoint"
    );
    let retained = reserved(&admission);
    let mut last = None;
    for (i, case) in cases.iter().enumerate() {
        let start = std::time::Instant::now();
        let features = model.features(&case.samples, 16000, None).unwrap();
        assert_eq!(features.frames(), case.frames);
        assert_eq!(features.width(), 128);
        compare(
            "original log-mel",
            features.values(),
            &case.features,
            2e-5,
            2e-5,
        );
        let output = model.encode(&features, None).unwrap();
        assert_eq!(
            (output.rows(), output.width(), output.source()),
            (case.tokens, case.width, model.source())
        );
        println!(
            "AUDIO_FORWARD case={i} elapsed={:.3}s rows={}",
            start.elapsed().as_secs_f64(),
            output.rows()
        );
        compare(
            "learned language projection",
            output.values(),
            &case.values,
            2e-3,
            2e-4,
        );
        drop(features);
        drop(last.replace(output));
    }
    let output = last.unwrap();
    let with_result = reserved(&admission);
    assert_eq!(with_result, retained + (output.values().len() as u64) * 4);
    assert!(model.features(&[0.; 320], 48000, None).is_err());
    assert!(model.features(&[f32::NAN; 320], 16000, None).is_err());
    assert!(model.features(&[0.; 160], 16000, None).is_err());
    assert_eq!(reserved(&admission), with_result);
    let features = model.features(&cases[1].samples, 16000, None).unwrap();
    let before_cancel = reserved(&admission);
    let cancel = AtomicBool::new(false);
    // Cancel after actual scratch admission, not a fabricated completed
    // result. Observe the same physical ledger; no extra execution owner.
    std::thread::scope(|scope| {
        let signal = scope.spawn(|| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while reserved(&admission) == before_cancel {
                assert!(
                    std::time::Instant::now() < deadline,
                    "audio work did not admit"
                );
                std::thread::yield_now();
            }
            cancel.store(true, Ordering::Release);
        });
        assert!(model.encode(&features, Some(&cancel)).is_err());
        signal.join().unwrap();
    });
    assert_eq!(reserved(&admission), before_cancel);
    drop(features);
    assert_eq!(reserved(&admission), with_result);
    drop(model);
    assert_eq!(
        reserved(&admission),
        baseline + output.values().len() as u64 * 4
    );
    assert!(output.source().is_bound());
    drop(output);
    assert_eq!(reserved(&admission), baseline);
    println!("AUDIO_ORIGINAL_SOURCE_REFERENCE_PASS");
}

#[derive(Deserialize)]
struct LanguageCase {
    prompt: Vec<u32>,
    audio_case: usize,
    decode_token: u32,
    prefill: Vec<f32>,
    decode: Vec<f32>,
}
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

#[test]
#[ignore = "requires original cached Gemma4 source and independent audio-conditioned text reference"]
fn actual_audio_language_prefill_and_decode_match_transformers() {
    let (cases, lease, admission) = fixture();
    let reference: LanguageCase = serde_json::from_slice(
        &std::fs::read(
            std::env::var_os("ZG_GEMMA4_AUDIO_LANGUAGE_REFERENCE")
                .expect("original audio language reference"),
        )
        .unwrap(),
    )
    .unwrap();
    let case = &cases[reference.audio_case];
    let baseline = reserved(&admission);
    let start = std::time::Instant::now();
    let model = Gemma4AudioLanguage::load(&lease, 11 << 30, &admission, None).unwrap();
    let retained = reserved(&admission) - baseline;
    println!(
        "AUDIO_LANGUAGE_BOUND elapsed={:.3}s retained_bytes={retained}",
        start.elapsed().as_secs_f64()
    );
    assert_eq!(model.source(), model.encoder().source());
    assert!(
        retained < 128 * 1024 * 1024,
        "one small original source and prepared parameter set"
    );
    let witness = model.encoder().verified_source().weak_owner();
    let output = model
        .encoder()
        .encode_pcm(&case.samples, 16000, None)
        .unwrap();
    assert_eq!(output.source(), model.source());
    compare(
        "shared encoder actual rows",
        output.values(),
        &case.values,
        2e-3,
        2e-4,
    );
    let mut session = model.session(reference.prompt.len() + 2).unwrap();
    let initial = session.binding();
    // Bad media cardinality and pre-cancel never create committed KV.
    let mut wrong = reference.prompt.clone();
    let index = wrong.iter().position(|&id| id == 258881).unwrap();
    wrong.remove(index);
    assert!(session.prefill(&wrong, &output, None).is_err());
    let cancel = AtomicBool::new(true);
    assert!(
        session
            .prefill(&reference.prompt, &output, Some(&cancel))
            .is_err()
    );
    assert_eq!(session.binding(), initial);
    let start = std::time::Instant::now();
    let prefill = session.prefill(&reference.prompt, &output, None).unwrap();
    println!(
        "AUDIO_LANGUAGE_PREFILL tokens={} elapsed={:.3}s",
        reference.prompt.len(),
        start.elapsed().as_secs_f64()
    );
    assert_eq!(session.binding().frontier(), reference.prompt.len() as u64);
    assert_eq!(prefill.as_slice().unwrap().len(), 262144);
    // Whole vocabulary, independent original F32 PyTorch arithmetic.
    compare(
        "audio-conditioned prefill logits",
        prefill.as_slice().unwrap(),
        &reference.prefill,
        0.01,
        0.002,
    );
    assert_eq!(
        argmax(prefill.as_slice().unwrap()),
        reference.decode_token as usize
    );
    drop(prefill);
    let before = session.binding();
    let start = std::time::Instant::now();
    let decode = session.decode(reference.decode_token, None).unwrap();
    println!(
        "AUDIO_LANGUAGE_DECODE elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    assert_eq!(session.binding().frontier(), before.frontier() + 1);
    compare(
        "audio-conditioned cached decode logits",
        decode.as_slice().unwrap(),
        &reference.decode,
        0.01,
        0.002,
    );
    assert_eq!(
        argmax(decode.as_slice().unwrap()),
        argmax(&reference.decode)
    );
    drop(decode);
    let before_cancel = session.binding();
    let before_bytes = reserved(&admission);
    cancel.store(false, Ordering::Release);
    // Cancel after real provisional state allocation; the old audio/text
    // frontier and every KV owner remain authoritative.
    std::thread::scope(|scope| {
        let signal = scope.spawn(|| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while reserved(&admission) == before_bytes {
                assert!(
                    std::time::Instant::now() < deadline,
                    "decoder did not admit private state"
                );
                std::thread::yield_now();
            }
            cancel.store(true, Ordering::Release);
        });
        assert!(
            session
                .decode(reference.decode_token, Some(&cancel))
                .is_err()
        );
        signal.join().unwrap();
    });
    assert_eq!(session.binding(), before_cancel);
    assert_eq!(reserved(&admission), before_bytes);
    drop(session);
    drop(model);
    assert!(
        !witness.retained(),
        "encoder result must not retain checkpoint backing"
    );
    assert_eq!(
        reserved(&admission),
        baseline + output.values().len() as u64 * 4
    );
    drop(output);
    assert_eq!(reserved(&admission), baseline);
    println!("AUDIO_LANGUAGE_ORIGINAL_SOURCE_REFERENCE_PASS");
}
