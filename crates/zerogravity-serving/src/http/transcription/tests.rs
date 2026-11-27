//! Actual listener/registry/receipt tests. The control adapter checks
//! transport and lifecycle only, not learned speech accuracy.
use super::*;
use crate::{
    ServerBuilder,
    ingress::IngressLimits,
    test_support::{self, HostBytes},
};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_core::{
    identity::SourceIdentity,
    resource_arbiter::{AllocationClass, GlobalResourceLease, TransferAdmission},
};
use zerogravity_engine::{
    model_instance::execution::{WorkPolicy, WorkerConfig},
    router::Router,
    transcription::{SpeechRecognizer, SpeechSession, TranscriptionUsage},
};

fn wav(samples: &[i16]) -> Vec<u8> {
    let n = (samples.len() * 2) as u32;
    let mut out = b"RIFF".to_vec();
    out.extend((36 + n).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(16000u32.to_le_bytes());
    out.extend(32000u32.to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend(n.to_le_bytes());
    for sample in samples {
        out.extend(sample.to_le_bytes());
    }
    out
}
fn chunk(bytes: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    bytes.extend(kind);
    bytes.extend((data.len() as u32).to_le_bytes());
    bytes.extend(data);
    if !data.len().is_multiple_of(2) {
        bytes.push(0);
    }
    let size = (bytes.len() - 8) as u32;
    bytes[4..8].copy_from_slice(&size.to_le_bytes());
}
#[test]
fn wav_checks_actual_geometry_and_extent() {
    let audio = wav(&[i16::MIN, 0, i16::MAX]);
    assert_eq!(pcm16_wav(&audio, 3).unwrap(), &audio[44..]);
    assert_eq!(pcm16_wav(&audio, 2).unwrap_err().status, 413);
    for (offset, value) in [(20, 3), (22, 2), (24, 0), (28, 1), (32, 4), (34, 8)] {
        let mut wrong = audio.clone();
        wrong[offset] = value;
        assert!(pcm16_wav(&wrong, 3).is_err(), "{offset}");
    }
    for len in 0..audio.len() {
        assert!(pcm16_wav(&audio[..len], 3).is_err(), "{len}");
    }
    let mut ancillary = audio.clone();
    chunk(&mut ancillary, b"JUNK", b"x");
    assert_eq!(pcm16_wav(&ancillary, 3).unwrap(), &audio[44..]);
    ancillary.pop();
    let n = (ancillary.len() - 8) as u32;
    ancillary[4..8].copy_from_slice(&n.to_le_bytes());
    assert!(pcm16_wav(&ancillary, 3).is_err());
    let mut duplicate = audio.clone();
    chunk(&mut duplicate, b"data", &[0, 0]);
    assert!(pcm16_wav(&duplicate, 3).is_err());
    assert!(pcm16_wav(&wav(&[]), 3).is_err());
}

struct Probe {
    source: SourceIdentity,
    memory: GlobalResourceLease,
    public: bool,
    starts: Arc<AtomicUsize>,
    entered: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}
impl SpeechRecognizer for Probe {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn max_samples(&self) -> usize {
        4096
    }
    fn max_context(&self) -> usize {
        32
    }
    fn vocabulary_size(&self) -> u64 {
        4
    }
    fn max_output_tokens(&self) -> usize {
        2
    }
    fn max_output_bytes(&self) -> usize {
        2048
    }
    fn public_transcription(&self) -> bool {
        self.public
    }
    fn validate_admission(&self, a: &TransferAdmission) -> zerogravity_engine::Result<()> {
        self.memory
            .bytes_in_domain(&a.arbiter, &a.resource)
            .map(|_| ())
    }
    fn start(
        self: Arc<Self>,
        samples: &[f32],
        rate: u32,
        context: usize,
        max_tokens: usize,
        cancelled: &AtomicBool,
    ) -> zerogravity_engine::Result<Box<dyn SpeechSession>> {
        assert_eq!((rate, context, max_tokens), (16000, 32, 2));
        self.starts.fetch_add(1, Ordering::AcqRel);
        let mode = match samples {
            [a, b, c] if [*a, *b, *c] == [-1., 0., 32767. / 32768.] => 0,
            [0.25] => 1,
            [0.5] => 2,
            [0.75] => 3,
            _ => panic!("PCM conversion or request geometry changed"),
        };
        if mode == 2 {
            self.entered.store(true, Ordering::Release);
            let deadline = std::time::Instant::now() + Duration::from_secs(8);
            while !cancelled.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.cancelled
                .store(cancelled.load(Ordering::Acquire), Ordering::Release);
            return Err(zerogravity_engine::Error::State(
                "controlled cancellation".into(),
            ));
        }
        Ok(Box::new(Session { mode, steps: 0 }))
    }
}
struct Session {
    mode: u8,
    steps: usize,
}
impl SpeechSession for Session {
    fn step(
        &mut self,
        output: &mut String,
        _: &AtomicBool,
    ) -> zerogravity_engine::Result<Option<TranscriptionUsage>> {
        self.steps += 1;
        if self.steps == 1 {
            return Ok(None);
        }
        if self.mode == 3 {
            for _ in 0..1500 {
                output.push('x');
            }
        } else {
            output.push_str("wire probe");
        }
        Ok(Some(TranscriptionUsage {
            input_tokens: 5,
            output_tokens: 2,
            finish: if self.mode == 1 {
                TranscriptionFinish::Length
            } else {
                TranscriptionFinish::Stop
            },
        }))
    }
}
fn form(model: &str, audio: &[u8], extras: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    // File first, arbitrary field order, opaque client filename.
    out.extend(b"--zg-speech\r\nContent-Disposition: form-data; name=\"file\"; filename=\"../../clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n");
    out.extend(audio);
    for (key, value) in std::iter::once(("model", model)).chain(extras.iter().copied()) {
        write!(
            &mut out,
            "\r\n--zg-speech\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}"
        )
        .unwrap();
    }
    out.extend(b"\r\n--zg-speech--\r\n");
    out
}
fn open(address: SocketAddr, data: &[u8], auth: bool, chunked: bool) -> TcpStream {
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(socket,"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=zg-speech\r\nConnection: close\r\n").unwrap();
    if auth {
        write!(socket, "Authorization: Bearer secret\r\n").unwrap();
    }
    if chunked {
        write!(socket, "Transfer-Encoding: chunked\r\n\r\n").unwrap();
        for piece in data.chunks(37) {
            write!(socket, "{:x}\r\n", piece.len()).unwrap();
            socket.write_all(piece).unwrap();
            socket.write_all(b"\r\n").unwrap();
        }
        socket.write_all(b"0\r\n\r\n").unwrap();
    } else {
        write!(socket, "Content-Length: {}\r\n\r\n", data.len()).unwrap();
        socket.write_all(data).unwrap();
    }
    socket
}
fn call(address: SocketAddr, data: &[u8], auth: bool, chunked: bool) -> (u16, String, String) {
    let mut socket = open(address, data, auth, chunked);
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").expect("HTTP response");
    assert!(head.to_ascii_lowercase().contains("x-request-id:"));
    (
        head.split_whitespace().nth(1).unwrap().parse().unwrap(),
        head.to_owned(),
        body.to_owned(),
    )
}
fn wait(flag: &AtomicBool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !flag.load(Ordering::Acquire) {
        assert!(
            std::time::Instant::now() < deadline,
            "speech fixture timed out"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn multipart_wire_uses_original_speech_registration_and_receipt() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let starts = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let model = |public| -> Arc<dyn SpeechRecognizer> {
        Arc::new(Probe {
            source: SourceIdentity::for_new_allocation().unwrap(),
            memory: host
                .admit(
                    "speech-probe",
                    "source",
                    AllocationClass::ImmutableWeights,
                    64,
                )
                .unwrap(),
            public,
            starts: starts.clone(),
            entered: entered.clone(),
            cancelled: cancelled.clone(),
        })
    };
    let public = model(true);
    let router = Arc::new(
        Router::with_transcription(
            BTreeMap::from([
                ("fixture".into(), public.clone()),
                ("alias".into(), public),
                ("private".into(), model(false)),
            ]),
            host.clone(),
            WorkerConfig {
                workers: 1,
                active: 1,
                context: 32,
                output: 2,
                max_pending: Some(4),
                policy: WorkPolicy::Sequential,
            },
        )
        .unwrap(),
    );
    assert_eq!(
        router.serving_model_ids().collect::<Vec<_>>(),
        ["alias", "fixture"]
    );
    let limits = router.public_transcription_limits("alias").unwrap();
    let mut wrong = limits;
    wrong.registration_id += 1;
    let request = |model: &str| TranscriptionRequest {
        model: model.into(),
        samples: vec![-1., 0., 32767. / 32768.],
        sample_rate: 16000,
        max_tokens: 2,
    };
    assert_eq!(
        router
            .request_public_transcription(request("fixture"), wrong)
            .err()
            .unwrap()
            .code,
        "model_changed"
    );
    assert_eq!(
        router
            .request_public_transcription(request("private"), limits)
            .err()
            .unwrap()
            .code,
        "not_found"
    );
    let server = ServerBuilder::from_service(router.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    let audio = wav(&[i16::MIN, 0, i16::MAX]);
    let valid = form("alias", &audio, &[]);
    assert_eq!(call(address, &valid, false, false).0, 401);
    for (field, value) in [
        ("response_format", "verbose_json"),
        ("stream", "true"),
        ("temperature", "0.5"),
        ("language", "en"),
        ("prompt", "x"),
        ("include[]", "logprobs"),
        ("timestamp_granularities[]", "word"),
        ("chunking_strategy", "auto"),
        ("model", "fixture"),
    ] {
        let (status, _, body) = call(
            address,
            &form("fixture", &audio, &[(field, value)]),
            true,
            false,
        );
        assert_eq!(status, 400, "{field}: {body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], field.trim_end_matches("[]"));
    }
    assert_eq!(
        call(address, &form("private", &audio, &[]), true, false).0,
        404
    );
    assert_eq!(
        call(address, &form("missing", &audio, &[]), true, false).0,
        404
    );
    assert_eq!(
        call(address, &form("fixture", &audio[..43], &[]), true, false).0,
        400
    );
    assert_eq!(
        call(
            address,
            &form("fixture", &wav(&vec![0; 4097]), &[]),
            true,
            false
        )
        .0,
        413
    );
    assert_eq!(starts.load(Ordering::Acquire), 0);
    let (status, head, json) = call(address, &valid, true, false);
    assert_eq!(status, 200, "{json}");
    assert!(head.to_ascii_lowercase().contains("application/json"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap(),
        serde_json::json!({"text":"wire probe","usage":{"type":"tokens","input_tokens":5,"output_tokens":2,"total_tokens":7}})
    );
    let (status, head, body) = call(
        address,
        &form(
            "fixture",
            &audio,
            &[
                ("response_format", "text"),
                ("stream", "false"),
                ("temperature", "0.0"),
            ],
        ),
        true,
        true,
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        head.to_ascii_lowercase()
            .contains("text/plain; charset=utf-8")
    );
    assert_eq!(body, "wire probe");
    let (status, _, body) = call(address, &form("fixture", &wav(&[8192]), &[]), true, false);
    assert_eq!(status, 422, "{body}");
    assert!(!body.contains("wire probe"));
    assert!(body.contains("transcription_incomplete"));
    let socket = open(address, &form("fixture", &wav(&[16384]), &[]), true, false);
    wait(&entered);
    drop(socket);
    wait(&cancelled);
    server.shutdown().unwrap();
    let small = ServerBuilder::from_service(router.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .limits(IngressLimits {
            output_bytes: 1024,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let (status, _, body) = call(
        small.local_addr(),
        &form("fixture", &wav(&[24576]), &[]),
        true,
        false,
    );
    assert_eq!(status, 413, "{body}");
    assert!(body.contains("output_too_large"));
    small.shutdown().unwrap();
    assert_eq!(
        router.scheduler_snapshot().counters.transcription_requests,
        4
    );
    let mut router = Arc::try_unwrap(router).ok().unwrap();
    router.shutdown().unwrap();
    drop(router);
    assert_eq!(host.serving_bytes(), baseline);
}
