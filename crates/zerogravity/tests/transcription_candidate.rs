#![cfg(all(feature = "model-gemma4-audio-candidate", target_os = "linux"))]
//! Original cached learned speech, independent Transformers reference and
//! the actual public native SDK. No network, fake transcript or HTTP math.
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};
use zerogravity::{
    Runtime,
    embedding::ServingHost,
    transcription::{
        Gemma4SpeechCandidate, SpeechRecognizer, TranscriptionConfig, TranscriptionFinish,
        TranscriptionRequest,
    },
};
use zerogravity_artifacts::cache::{CacheOptions, CacheStore};

#[derive(Deserialize)]
struct Reference {
    samples: Vec<f32>,
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    text: String,
    finish: String,
}
#[test]
#[ignore = "requires cached original Gemma4 E2B and independent spoken reference"]
fn actual_spoken_transcription_through_runtime() {
    let path =
        std::env::var_os("ZG_GEMMA4_SPEECH_REFERENCE").expect("independent speech reference");
    let reference: Reference = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(reference.text.trim().to_lowercase(), "hello world");
    assert_eq!(reference.finish, "stop");
    let store = CacheStore::open_existing(CacheOptions::new(
        std::env::var_os("ZG_GEMMA4_AUDIO_CACHE").map(Into::into),
        std::env::var("ZG_GEMMA4_AUDIO_CACHE_BYTES")
            .unwrap()
            .parse()
            .unwrap(),
    ))
    .unwrap();
    let snapshot = store
        .open_snapshot(&std::env::var("ZG_GEMMA4_AUDIO_SNAPSHOT").unwrap())
        .unwrap();
    let host = ServingHost::new(512).unwrap();
    let baseline = host.snapshot().unwrap().1;
    let runtime = Runtime::builder().build().unwrap();
    let start = Instant::now();
    runtime
        .models()
        .load_transcribers(
            TranscriptionConfig {
                host_mib: 512,
                max_context: 128,
                max_output_tokens: 8,
                ..Default::default()
            },
            |host| {
                let model = Gemma4SpeechCandidate::prepare(&snapshot, host, "gemma-speech", None)
                    .map_err(zerogravity::runtime::text::ServiceError::inference)?;
                let prompt = model
                    .prompt_tokens(reference.samples.len(), 120)
                    .map_err(zerogravity::runtime::text::ServiceError::inference)?;
                assert_eq!(prompt.ids(), reference.prompt);
                drop(prompt);
                let model: Arc<dyn SpeechRecognizer> = model;
                Ok(BTreeMap::from([
                    ("gemma-speech".into(), model.clone()),
                    ("speech-alias".into(), model),
                ]))
            },
        )
        .unwrap();
    let catalog = runtime.models().snapshot().unwrap();
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].aliases, ["gemma-speech", "speech-alias"]);
    assert!(catalog[0].capabilities.as_ref().unwrap().transcription);
    assert!(runtime.models().list().unwrap().is_empty());
    println!(
        "SPEECH_RUNTIME_BOUND {:.3}s reserved={}",
        start.elapsed().as_secs_f64(),
        host.snapshot().unwrap().1 - baseline
    );
    let request = |samples: Vec<f32>, rate| TranscriptionRequest {
        model: "speech-alias".into(),
        samples,
        sample_rate: rate,
        max_tokens: 8,
    };
    assert!(
        runtime
            .request_transcription(request(vec![0.; 320], 48000))
            .is_err()
    );
    assert!(
        runtime
            .request_transcription(request(vec![f32::NAN; 320], 16000))
            .is_err()
    );
    let start = Instant::now();
    let receipt = runtime
        .request_transcription(request(reference.samples.clone(), 16000))
        .unwrap();
    // Native-only speech registration must never be advertised as a public
    // text or embedding model.
    #[cfg(feature = "serve")]
    {
        use zerogravity_serving::TextService;
        assert!(runtime.model_ids().unwrap().is_empty());
    }
    let result = receipt
        .events
        .recv_timeout(Duration::from_secs(480))
        .unwrap()
        .unwrap();
    println!(
        "SPEECH_RUNTIME_COMPLETE {:.3}s text={:?} usage={:?}",
        start.elapsed().as_secs_f64(),
        result.text(),
        result.usage()
    );
    assert_eq!(result.text(), reference.text);
    assert_eq!(result.usage().input_tokens, reference.prompt.len());
    assert_eq!(result.usage().output_tokens, reference.tokens.len());
    assert_eq!(result.usage().finish, TranscriptionFinish::Stop);
    assert_eq!(result.source(), catalog[0].source);
    drop(receipt);
    let stats = runtime.scheduler_snapshot().unwrap().unwrap().counters;
    assert_eq!(stats.transcription_requests, 1);
    assert_eq!(
        stats.transcription_input_tokens,
        reference.prompt.len() as u64
    );
    assert_eq!(
        stats.transcription_output_tokens,
        reference.tokens.len() as u64
    );
    assert_eq!(stats.embedding_requests, 0);
    // Cancel an admitted actual speech request and join retirement. No
    // completed text may escape from its private state.
    let cancelled = runtime
        .request_transcription(request(reference.samples, 16000))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while runtime
        .scheduler_snapshot()
        .unwrap()
        .unwrap()
        .counters
        .active
        == 0
    {
        assert!(Instant::now() < deadline, "speech request not admitted");
        std::thread::yield_now();
    }
    cancelled.cancel();
    assert!(
        cancelled
            .events
            .recv_timeout(Duration::from_secs(30))
            .is_err()
    );
    drop(cancelled);
    runtime
        .models()
        .unload_registration("speech-alias", catalog[0].registration_id)
        .unwrap();
    assert!(runtime.models().snapshot().unwrap().is_empty());
    assert_eq!(result.text(), "Hello world");
    assert!(host.snapshot().unwrap().1 > baseline);
    drop(result);
    assert_eq!(host.snapshot().unwrap().1, baseline);
    runtime.shutdown().unwrap();
}
