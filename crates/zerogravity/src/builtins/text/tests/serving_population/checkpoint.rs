//! One original tiny GGUF/Qwen CPU path, not trained-checkpoint qualification.
//! All successful text comes from the real fixture and approved tokenizer.
use super::{HOST_MIB, MAX_CONTEXT, MAX_OUTPUT, fixture, reexec, request};
use crate::{
    Runtime,
    run::{OutputStream, TaskInput, TaskRequest},
    runtime::{
        TextConfig,
        text::{RequestEvents, ResponseEvent, event_channel::mpsc},
    },
};
use std::{sync::Arc, time::Duration};

const WAIT: Duration = Duration::from_secs(30);
type Outcome = (String, usize, usize, &'static str);

fn terminal(event: ResponseEvent) -> Outcome {
    match event {
        ResponseEvent::Done {
            output,
            prompt_tokens,
            completion_tokens,
            finish_reason,
        } => (output, prompt_tokens, completion_tokens, finish_reason),
        ResponseEvent::Error(error) => panic!("real fixture failed: {error}"),
        _ => panic!("expected one terminal event"),
    }
}

fn direct(runtime: &Runtime, alias: &str) -> Outcome {
    let mut request = request();
    request.model = alias.into();
    let receipt: RequestEvents = runtime.request_text(request).expect("SDK request_text");
    assert!(receipt.id > 0);
    assert_eq!(receipt.model, alias);
    let outcome = terminal(
        receipt
            .events
            .recv_timeout(WAIT)
            .expect("direct completion"),
    );
    assert!(
        matches!(
            receipt.events.recv_timeout(WAIT),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ),
        "non-streaming receipt has exactly one terminal and no deltas"
    );
    outcome
}

fn streamed(mut stream: Box<dyn OutputStream>) -> Outcome {
    let mut deltas = String::new();
    let mut done = None;
    while let Some(event) = stream.next().expect("SDK stream event") {
        assert!(done.is_none(), "event after terminal");
        match event {
            ResponseEvent::Text(text) => deltas.push_str(&text),
            event => done = Some(terminal(event)),
        }
    }
    let outcome = done.expect("stream terminal");
    assert_eq!(
        deltas, outcome.0,
        "actual decoded deltas match terminal output"
    );
    outcome
}

#[test]
#[ignore = "requires ZG_APPROVED_TOKENIZER; serve additionally requires private loopback"]
fn source_bound_sdk_http_lifecycle() {
    const TEST: &str =
        "builtins::text::tests::serving_population::checkpoint::source_bound_sdk_http_lifecycle";
    const DONE: &str = "SDK_CHECKPOINT_CHILD_OK";
    if reexec("ZG_SDK_CHECKPOINT_CHILD", TEST, DONE) {
        return;
    }
    // OutputStream::next and lifecycle joins have no caller deadline. Bound
    // the entire isolated child (including teardown) without installing signals.
    let (finished, deadline) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if matches!(
            deadline.recv_timeout(Duration::from_secs(180)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ) {
            eprintln!("SDK checkpoint child exceeded 180 seconds");
            std::process::exit(124);
        }
    });
    let root = fixture("sdk-checkpoint");
    let runtime = Runtime::builder().build().unwrap();
    let clone = runtime.clone();
    let model = root.join("model.gguf");
    let declared = vec![("tiny".into(), model.clone()), ("alias".into(), model)];
    #[cfg(not(feature = "serve"))]
    let load = crate::models::ModelLoadRequest::CpuText {
        declared,
        config: TextConfig {
            workers: 1,
            active_per_worker: 1,
            max_context: MAX_CONTEXT,
            max_output: MAX_OUTPUT,
            host_mib: HOST_MIB,
            max_pending: Some(4),
        },
    };
    #[cfg(feature = "serve")]
    let load = crate::models::ModelLoadRequest::ServingText {
        declared,
        profile: crate::serving::TextProfile {
            workers: 1,
            active_per_worker: 1,
            max_context: MAX_CONTEXT,
            max_output: MAX_OUTPUT,
            host_mib: HOST_MIB,
            max_pending_requests: Some(4),
            backend: "cpu".into(),
            source_profile: "gguf".into(),
            native_artifacts: None,
            provider_mib: 2048,
            device_mib: 4096,
            batch_mode: "interleaved".into(),
            batch_rows: 2,
            native_batch_rows: 0,
        },
    };
    runtime
        .models()
        .load(load)
        .expect("public source-bound CPU load");
    #[cfg(feature = "serve")]
    {
        use crate::serving::TextService;
        let signal = runtime.charge_ingress("caller-signal", 64).unwrap();
        assert_eq!(runtime.model_ids().unwrap(), vec!["alias", "tiny"]);
        assert_eq!(runtime.max_output().unwrap(), MAX_OUTPUT);
        runtime.require_open().unwrap();
        drop(signal);
    }
    assert_eq!(clone.status().loaded_models, 2);
    assert_eq!(clone.status().execution_adapters, 1);
    let models = clone.models().list().unwrap();
    assert_eq!(models.len(), 2);
    assert!(models[0].shares_source(&models[1]));
    {
        let named = clone.models().resolve("tiny").unwrap();
        let alias = clone.models().resolve("alias").unwrap();
        assert!(named.shares_source(&alias));
        assert!(named.shares_source(&models[0]));
        assert!(clone.models().resolve("absent").is_err());
        let error = clone
            .models()
            .load(crate::models::ModelLoadRequest::CpuText {
                declared: vec![(
                    "never-opened".into(),
                    "/source-must-not-open-on-reload".into(),
                )],
                config: TextConfig::default(),
            })
            .unwrap_err();
        assert_eq!(error.code, "runtime_configured");
    }
    let source = Arc::downgrade(models[0].prepared());
    let owner = models[0].prepared().tokenizer();
    assert!(Arc::ptr_eq(&owner, &models[1].prepared().tokenizer()));
    let prompt = owner.prompt_tokens("!", false).unwrap();
    assert_eq!(prompt, vec![0], "approved tokenizer's actual fixture input");
    assert_eq!(owner.decode(&[0], true).unwrap(), "!");
    let tokenizer = Arc::downgrade(&owner);
    drop((owner, models));
    // Keep only weak observations: test-held source/tokenizer owners must not
    // mask a runtime lifetime bug during cancellation, server stop or shutdown.
    let check_source = || {
        let models = clone.models().list().unwrap();
        assert_eq!(models.len(), 2);
        assert!(models[0].shares_source(&models[1]));
        assert!(Arc::ptr_eq(
            models[0].prepared(),
            &source.upgrade().expect("live source")
        ));
        let owner = tokenizer.upgrade().expect("live admitted tokenizer");
        assert!(Arc::ptr_eq(&owner, &models[0].prepared().tokenizer()));
        assert_eq!(owner.prompt_tokens("!", false).unwrap(), prompt);
    };
    let expected = ("!".to_owned(), prompt.len(), 1, "length");
    assert_eq!(direct(&runtime, "tiny"), expected);
    let task = TaskRequest {
        model: "alias".into(),
        input: TaskInput::Text("!".into()),
        max_output: MAX_OUTPUT.try_into().unwrap(),
    };
    assert_eq!(streamed(clone.run(&task).unwrap()), expected);

    // Phase timing is deliberately not claimed: these may be queued, active
    // or already finished. One active slot makes later completion meaningful.
    drop(runtime.request_text(request()).unwrap());
    let mut cancelled = clone.run(&task).unwrap();
    cancelled.cancel().unwrap();
    assert!(cancelled.next().unwrap().is_none());
    drop(cancelled);
    drop(clone.run(&task).unwrap());
    check_source();
    assert_eq!(direct(&clone, "alias"), expected);

    #[cfg(feature = "serve")]
    http_checkpoint(&runtime, &task, &expected);
    check_source();
    assert!(!clone.status().closed);
    assert_eq!(
        direct(&clone, "tiny"),
        expected,
        "later SDK request (after server stop when serve is enabled)"
    );

    runtime.shutdown().unwrap();
    for handle in [&runtime, &clone] {
        assert!(handle.status().closed);
        assert_eq!(handle.status().loaded_models, 0);
        assert_eq!(handle.status().execution_adapters, 0);
        let error = handle
            .request_text(request())
            .err()
            .expect("closed admission");
        assert_eq!((error.status, error.code), (503, "runtime_closed"));
        assert!(matches!(handle.run(&task), Err(crate::Error::State(_))));
        assert!(handle.models().list().is_err());
        #[cfg(feature = "serve")]
        assert!(handle.server().build().is_err());
        handle.shutdown().unwrap();
    }
    assert!(
        source.upgrade().is_none(),
        "runtime retired the shared source"
    );
    assert!(
        tokenizer.upgrade().is_none(),
        "original tokenizer owner retired"
    );
    drop((runtime, clone));
    std::fs::remove_dir_all(root).unwrap();
    finished.send(()).unwrap();
    watchdog.join().unwrap();
    println!("{DONE}");
}

#[cfg(feature = "serve")]
fn http_checkpoint(runtime: &Runtime, task: &TaskRequest, expected: &Outcome) {
    use crate::serving::{ServerBuilder, ingress::IngressLimits};
    use std::{
        io::{Read, Write},
        net::{SocketAddr, TcpStream},
    };

    const WIRE_LIMIT: usize = 16 * 1024;
    fn exchange(address: SocketAddr, alias: &str, streaming: bool) -> String {
        let mut body = serde_json::json!({
            "model": alias, "prompt": "!", "max_tokens": MAX_OUTPUT, "stream": streaming,
        });
        if streaming {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }
        let body = body.to_string();
        let mut socket = TcpStream::connect_timeout(&address, WAIT).unwrap();
        socket.set_read_timeout(Some(WAIT)).unwrap();
        socket.set_write_timeout(Some(WAIT)).unwrap();
        write!(
            socket,
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        )
        .unwrap();
        let mut raw = String::new();
        socket
            .take((WIRE_LIMIT + 1) as u64)
            .read_to_string(&mut raw)
            .unwrap();
        assert!(raw.len() <= WIRE_LIMIT);
        assert!(raw.starts_with("HTTP/1.1 200 OK\r\n"), "{raw}");
        raw
    }

    let builder: ServerBuilder = runtime.server();
    let server = builder
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(IngressLimits {
            connections: 4,
            body_bytes: 1024,
            output_bytes: WIRE_LIMIT,
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    assert!(address.ip().is_loopback());
    assert_ne!(address.port(), 0);
    let usage = serde_json::json!({
        "prompt_tokens": expected.1, "completion_tokens": expected.2,
        "total_tokens": expected.1 + expected.2,
    });
    let raw = exchange(address, "tiny", false);
    let (headers, body) = raw.split_once("\r\n\r\n").unwrap();
    assert!(headers.contains("Content-Type: application/json"));
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(json["model"], "tiny");
    assert_eq!(json["choices"].as_array().unwrap().len(), 1);
    assert_eq!(json["choices"][0]["text"], expected.0);
    assert_eq!(json["choices"][0]["finish_reason"], expected.3);
    assert_eq!(json["usage"], usage);

    let raw = exchange(address, "alias", true);
    let (headers, mut chunks) = raw.split_once("\r\n\r\n").unwrap();
    assert!(headers.contains("Content-Type: text/event-stream"));
    assert!(headers.contains("Transfer-Encoding: chunked"));
    let mut payload = String::new();
    while chunks != "0\r\n\r\n" {
        let (size, rest) = chunks.split_once("\r\n").unwrap();
        let size = usize::from_str_radix(size, 16).unwrap();
        assert!(size > 0, "exactly one final HTTP chunk, no trailers");
        let (data, rest) = rest.split_at(size);
        payload.push_str(data);
        chunks = rest.strip_prefix("\r\n").expect("chunk boundary");
    }
    let frames: Vec<_> = payload
        .split_terminator("\n\n")
        // Keep-alive comments depend on CPU scheduling, not model output.
        .filter(|frame| !frame.starts_with(':'))
        .map(|frame| frame.strip_prefix("data: ").expect("SSE data frame"))
        .collect();
    assert_eq!(frames.len(), 4, "one delta, terminal, usage, and DONE");
    assert_eq!(frames[3], "[DONE]");
    let events: Vec<serde_json::Value> = frames[..3]
        .iter()
        .map(|frame| serde_json::from_str(frame).unwrap())
        .collect();
    assert!(events.iter().all(|event| event["model"] == "alias"));
    assert_eq!(events[0]["choices"][0]["text"], expected.0);
    assert!(events[0]["choices"][0]["finish_reason"].is_null());
    assert_eq!(events[1]["choices"][0]["text"], "");
    assert_eq!(events[1]["choices"][0]["finish_reason"], expected.3);
    assert_eq!(events[2]["choices"], serde_json::json!([]));
    assert_eq!(events[2]["usage"], usage);

    // A caller-owned SDK stream survives stopping/joining only HTTP ingress.
    // It may already have completed; no race-dependent active-phase assertion.
    let held = runtime.run(task).unwrap();
    server.stop();
    server.join().unwrap();
    assert!(!runtime.status().closed);
    assert_eq!(&streamed(held), expected);
}
