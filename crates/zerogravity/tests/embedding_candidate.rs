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
    runtime.shutdown().unwrap();
}
