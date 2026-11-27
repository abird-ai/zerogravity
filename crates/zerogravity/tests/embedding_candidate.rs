#![cfg(all(feature = "model-bert-cpu-candidate", target_os = "linux"))]
//! Actual source-bound synthetic BERT forward through the public native SDK.
//! This establishes adapter/worker parity, not trained MiniLM accuracy.
#[path = "../../zerogravity-model-bert/tests/support/mod.rs"]
mod support;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use zerogravity::{
    Runtime,
    embedding::{
        BertEncoderCandidate, EmbeddingEncoder, EmbeddingInput, EmbeddingRequest, EncoderConfig,
        ServingHost,
    },
    runtime::text::{SemanticRequest, TextInput},
};

#[test]
fn original_encoder_source_flows_through_runtime_batch_and_retirement() {
    let host = ServingHost::new(256).unwrap();
    let baseline = host.snapshot().unwrap().1;
    let runtime = Runtime::builder().build().unwrap();
    let fixture = support::Fixture::new();
    runtime
        .models()
        .load_encoders(
            EncoderConfig {
                host_mib: 256,
                max_context: 4,
                ..Default::default()
            },
            |host| {
                let admission = host.transfer_admission("bert", "fixture")?;
                let (_store, snapshot) = fixture.snapshot(
                    support::config(),
                    support::weights(false, false),
                    &admission,
                );
                let model: Arc<dyn EmbeddingEncoder> =
                    BertEncoderCandidate::prepare(&snapshot, host, "bert")
                        .map_err(zerogravity::runtime::text::ServiceError::inference)?;
                Ok(BTreeMap::from([
                    ("bert".into(), model.clone()),
                    ("alias".into(), model),
                ]))
            },
        )
        .unwrap();
    let catalog = runtime.models().snapshot().unwrap();
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].aliases, ["alias", "bert"]);
    let source = catalog[0].source;
    let registration = catalog[0].registration_id;
    let capabilities = catalog[0].capabilities.as_ref().unwrap();
    assert_eq!(capabilities.embedding_dimensions, Some(4));
    assert!(
        capabilities.prefill.is_none()
            && capabilities.max_token_bytes.is_none()
            && !capabilities.chat_allowed
    );
    assert!(runtime.models().list().unwrap().is_empty());
    let reply = runtime
        .request_embeddings(EmbeddingRequest {
            model: "alias".into(),
            inputs: vec![
                EmbeddingInput::Text("HÉLLOS".into()),
                EmbeddingInput::Tokens(vec![2, 5, 6, 3]),
                EmbeddingInput::Tokens(vec![1, 2, 3]),
            ],
        })
        .unwrap();
    let batch = reply
        .events
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    drop(reply);
    assert_eq!(
        (
            batch.len(),
            batch.dimensions(),
            batch.input_tokens(),
            batch.source()
        ),
        (3, 4, 11, source)
    );
    assert_eq!(batch.get(0), batch.get(1));
    for (&v, expected) in
        batch
            .get(2)
            .unwrap()
            .iter()
            .zip([0.33089855f64, -0.59064984, 0.56863701, -0.46721610])
    {
        assert!((f64::from(v) - expected).abs() < 2e-5);
    }
    assert!(batch.get(3).is_none());
    let wrong = runtime.request_semantic(SemanticRequest {
        model: "bert".into(),
        input: TextInput::Raw("hello".into()),
        max_tokens: 1,
        stream: false,
        token_metrics: None,
    });
    assert_eq!(wrong.err().unwrap().code, "unsupported_task");
    assert_eq!(
        runtime
            .count_input_tokens("bert", &TextInput::Raw("hello".into()))
            .unwrap_err()
            .code,
        "unsupported_task"
    );
    // A second sequence whose rendered token length exceeds the runtime
    // context must discard the entire privately calculated batch.
    let bad = runtime
        .request_embeddings(EmbeddingRequest {
            model: "bert".into(),
            inputs: vec![
                EmbeddingInput::Tokens(vec![1, 2, 3]),
                EmbeddingInput::Text("hello hello hello".into()),
            ],
        })
        .unwrap();
    assert!(
        bad.events
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .is_err()
    );
    drop(bad);
    #[cfg(feature = "serve")]
    {
        use std::io::{Read, Write};
        let server = runtime
            .server()
            .bind("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap()
            .start()
            .unwrap();
        let call = |method: &str, path: &str, body: &str| {
            let mut socket = std::net::TcpStream::connect(server.local_addr()).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            write!(socket,"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            let mut response = String::new();
            socket.read_to_string(&mut response).unwrap();
            response
        };
        let models = call("GET", "/v1/models", "");
        assert!(models.starts_with("HTTP/1.1 200"), "{models}");
        assert!(
            !models.contains("\"bert\"") && !models.contains("\"alias\""),
            "{models}"
        );
        let refused = call(
            "POST",
            "/v1/embeddings",
            r#"{"model":"bert","input":"hello"}"#,
        );
        assert!(refused.starts_with("HTTP/1.1 404"), "{refused}");
        assert!(refused.contains("\"not_found\""), "{refused}");
        server.shutdown().unwrap();
    }
    let counters = runtime.scheduler_snapshot().unwrap().unwrap().counters;
    assert_eq!(counters.embedding_requests, 1);
    assert_eq!(counters.embedding_input_tokens, 11);
    assert_eq!(counters.output_tokens, 0);
    assert_eq!(counters.committed_roots, 0);
    runtime
        .models()
        .unload_registration("alias", registration)
        .unwrap();
    assert!(runtime.models().snapshot().unwrap().is_empty());
    // Receipt/result retention does not pin the model or worker.
    assert!(host.snapshot().unwrap().1 > baseline);
    assert_eq!(batch.source(), source);
    drop(batch);
    assert_eq!(host.snapshot().unwrap().1, baseline);
    #[cfg(feature = "serve")]
    wire_fixture(&runtime, &host, baseline);
    runtime.shutdown().unwrap();
}

// Explicit test adapter declaration only. Built-in candidate remains
// native-only even when the feature is selected.
#[cfg(feature = "serve")]
mod wire {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD};
    use serde_json::{Value, json};
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicBool, Ordering};
    use zerogravity_core::{identity::SourceIdentity, resource_arbiter::TransferAdmission};

    struct PublicFixture {
        model: Arc<BertEncoderCandidate>,
        entered: Arc<AtomicBool>,
        cancelled: Arc<AtomicBool>,
    }
    impl EmbeddingEncoder for PublicFixture {
        fn source(&self) -> SourceIdentity {
            self.model.source()
        }
        fn max_context(&self) -> usize {
            self.model.max_context()
        }
        fn vocabulary_size(&self) -> u64 {
            self.model.vocabulary_size()
        }
        fn dimensions(&self) -> usize {
            self.model.dimensions()
        }
        fn public_embeddings(&self) -> bool {
            true
        }
        fn validate_admission(&self, a: &TransferAdmission) -> zerogravity::Result<()> {
            self.model.validate_admission(a)
        }
        fn embed_into(
            &self,
            input: &EmbeddingInput,
            max_tokens: usize,
            out: &mut [f32],
            cancel: &AtomicBool,
        ) -> zerogravity::Result<usize> {
            if matches!(input,EmbeddingInput::Text(text) if text=="cancel fixture") {
                self.entered.store(true, Ordering::Release);
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while !cancel.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.cancelled
                    .store(cancel.load(Ordering::Acquire), Ordering::Release);
                return Err(zerogravity::Error::State("controlled cancellation".into()));
            }
            self.model.embed_into(input, max_tokens, out, cancel)
        }
    }
    fn request(
        address: std::net::SocketAddr,
        method: &str,
        path: &str,
        value: &Value,
        key: bool,
        chunked: bool,
    ) -> (u16, Value) {
        let body = if method == "POST" {
            value.to_string()
        } else {
            String::new()
        };
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(stream,"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\n").unwrap();
        if key {
            write!(stream, "Authorization: Bearer secret\r\n").unwrap();
        }
        if chunked {
            write!(
                stream,
                "Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                body.len(),
                body
            )
            .unwrap();
        } else {
            write!(stream, "Content-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        }
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
        assert!(headers.to_ascii_lowercase().contains("x-request-id:"));
        (status, serde_json::from_str(body).expect(&response))
    }

    fn wait(flag: &AtomicBool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !flag.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "encoder gate timed out"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    pub(super) fn exercise(runtime: &Runtime, host: &ServingHost, baseline: u64) {
        let fixture = support::Fixture::new();
        let entered = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));
        runtime
            .models()
            .load_encoders(
                EncoderConfig {
                    host_mib: 256,
                    max_context: 4,
                    ..Default::default()
                },
                |host| {
                    let admission = host.transfer_admission("wire-fixture", "fixture")?;
                    let (_store, snapshot) = fixture.snapshot(
                        support::config(),
                        support::weights(false, false),
                        &admission,
                    );
                    let model = BertEncoderCandidate::prepare(&snapshot, host, "wire-fixture")
                        .map_err(zerogravity::runtime::text::ServiceError::inference)?;
                    let public: Arc<dyn EmbeddingEncoder> = Arc::new(PublicFixture {
                        model: model.clone(),
                        entered: entered.clone(),
                        cancelled: cancelled.clone(),
                    });
                    // Native-only alias uses the original unqualified candidate;
                    // public declaration cannot leak through equal source identity.
                    let native: Arc<dyn EmbeddingEncoder> = model;
                    Ok(BTreeMap::from([
                        ("fixture".into(), public.clone()),
                        ("public-alias".into(), public),
                        ("private".into(), native),
                    ]))
                },
            )
            .unwrap();
        let server = runtime
            .server()
            .bind("127.0.0.1:0".parse().unwrap())
            .api_key(Some("secret".into()))
            .build()
            .unwrap()
            .start()
            .unwrap();
        let address = server.local_addr();
        let call = |value: &Value| request(address, "POST", "/v1/embeddings", value, true, false);
        let (_, catalog) = request(address, "GET", "/v1/models", &Value::Null, true, false);
        assert_eq!(
            catalog["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["fixture", "public-alias"]
        );
        assert_eq!(
            request(
                address,
                "GET",
                "/v1/models/private",
                &Value::Null,
                true,
                false
            )
            .0,
            404
        );
        assert_eq!(
            request(
                address,
                "GET",
                "/v1/models/public-alias",
                &Value::Null,
                true,
                false
            )
            .0,
            200
        );
        let value =
            json!({"model":"public-alias","input":["HÉLLOS","hello"],"encoding_format":"float"});
        assert_eq!(
            request(address, "POST", "/v1/embeddings", &value, false, false).0,
            401
        );
        let (status, output) = call(&value);
        assert_eq!(status, 200, "{output}");
        assert_eq!(output["object"], "list");
        assert_eq!(output["model"], "public-alias");
        assert_eq!(output["usage"], json!({"prompt_tokens":7,"total_tokens":7}));
        assert_eq!(output["data"][0]["index"], 0);
        assert_eq!(output["data"][1]["index"], 1);
        assert_eq!(output["data"][0]["object"], "embedding");
        let (status, ids) = request(
            address,
            "POST",
            "/v1/embeddings",
            &json!({"model":"fixture","input":[[2,5,6,3],[2,5,3]],"dimensions":4}),
            true,
            true,
        );
        assert_eq!(status, 200, "{ids}");
        assert_eq!(ids["data"], output["data"]);
        assert_eq!(ids["usage"], output["usage"]);
        let (status, single) = call(&json!({"model":"fixture","input":[2,5,6,3]}));
        assert_eq!(status, 200, "{single}");
        assert_eq!(single["data"][0], output["data"][0]);
        let (status, encoded) =
            call(&json!({"model":"fixture","input":"HÉLLOS","encoding_format":"base64"}));
        assert_eq!(status, 200, "{encoded}");
        let raw = STANDARD
            .decode(encoded["data"][0]["embedding"].as_str().unwrap())
            .unwrap();
        let expected = output["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|v| (v.as_f64().unwrap() as f32).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(raw, expected);
        use zerogravity_serving::TextService;
        let native_request = |alias: &str| EmbeddingRequest {
            model: alias.into(),
            inputs: vec![EmbeddingInput::Text("hello".into())],
        };
        assert_eq!(
            runtime
                .request_embeddings_admitted(native_request("private"), 4, vec![])
                .err()
                .unwrap()
                .code,
            "not_found"
        );
        assert_eq!(
            runtime
                .request_embeddings_admitted(native_request("fixture"), 3, vec![])
                .err()
                .unwrap()
                .code,
            "model_changed"
        );
        let before = runtime
            .scheduler_snapshot()
            .unwrap()
            .unwrap()
            .counters
            .embedding_requests;
        for (param, value) in [
            ("input", json!([])),
            ("input", json!([""])),
            ("input", json!([[]])),
            ("input", json!([1.5])),
            ("input", json!([-1])),
            ("input", json!([4294967296u64])),
            ("input", json!([7])),
            ("input", json!([1, 2, 3, 4, 5])),
            ("input", json!(["hello", [1, 2]])),
            ("input", json!(vec!["hello"; 17])),
            ("encoding_format", json!("hex")),
            ("dimensions", json!(2)),
            ("user", json!("ignored")),
        ] {
            let mut invalid = json!({"model":"fixture","input":"hello"});
            invalid[param] = value;
            let (status, error) = call(&invalid);
            assert_eq!(status, 400, "{invalid}: {error}");
            assert_eq!(error["error"]["param"], param);
        }
        assert_eq!(call(&json!({"model":"private","input":"hello"})).0, 404);
        assert_eq!(call(&json!({"model":"unknown","input":"hello"})).0, 404);
        assert_eq!(
            call(&json!({"model":"fixture","input":"hello","stream":true})).0,
            400
        );
        assert_eq!(
            runtime
                .scheduler_snapshot()
                .unwrap()
                .unwrap()
                .counters
                .embedding_requests,
            before
        );
        // The existing installed SDK is an optional real consumer, not a
        // runtime dependency or a required download.
        if let Some(python) = std::env::var_os("ZG_TEST_OPENAI_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/embedding_sdk.py");
            let status = std::process::Command::new(python)
                .arg("-B")
                .arg(script)
                .arg(address.to_string())
                .env(
                    "PYTHONPATH",
                    std::env::var_os("ZG_TEST_OPENAI_PYTHONPATH").unwrap_or_default(),
                )
                .status()
                .unwrap();
            assert!(
                status.success(),
                "installed OpenAI SDK embedding canary failed"
            );
        }
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        let body = r#"{"model":"fixture","input":["hello","cancel fixture"]}"#;
        write!(stream,"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();
        wait(&entered);
        drop(stream);
        server.shutdown().unwrap();
        wait(&cancelled);
        // Bound serialization before emitting any successful JSON. The
        // configured connection still has enough room for a typed refusal.
        let small = runtime
            .server()
            .bind("127.0.0.1:0".parse().unwrap())
            .api_key(Some("secret".into()))
            .limits(zerogravity_serving::ingress::IngressLimits {
                output_bytes: 1024,
                ..Default::default()
            })
            .build()
            .unwrap()
            .start()
            .unwrap();
        let (status, error) = request(
            small.local_addr(),
            "POST",
            "/v1/embeddings",
            &json!({"model":"fixture","input":vec!["hello";16]}),
            true,
            false,
        );
        assert_eq!(status, 413, "{error}");
        assert_eq!(error["error"]["code"], "output_too_large");
        small.shutdown().unwrap();
        // Shutdown joins HTTP jobs, not the caller's runtime.
        let reply = runtime
            .request_embeddings(EmbeddingRequest {
                model: "fixture".into(),
                inputs: vec![EmbeddingInput::Text("hello".into())],
            })
            .unwrap();
        assert_eq!(
            reply
                .events
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .input_tokens(),
            3
        );
        drop(reply);
        for snapshot in runtime.models().snapshot().unwrap() {
            runtime
                .models()
                .unload_registration(&snapshot.aliases[0], snapshot.registration_id)
                .unwrap();
        }
        assert!(runtime.models().snapshot().unwrap().is_empty());
        assert_eq!(host.snapshot().unwrap().1, baseline);
    }
}
#[cfg(feature = "serve")]
use wire::exercise as wire_fixture;
