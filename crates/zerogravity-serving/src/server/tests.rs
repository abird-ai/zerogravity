//! Socket/lifecycle controls only. Scripted events do not qualify model math.
use super::*;
use crate::{
    service::{Request, RequestEvents, ServiceError, ServingCharge},
    test_support::{self, HostBytes},
};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    sync::Mutex,
    time::{Duration, Instant},
};
use zerogravity_engine::router::{
    ChatMessage, ChatRole, ResponseEvent, SemanticRequest, TextInput,
    event_channel::mpsc,
    pending::{CancelToken, ServingHost},
};

fn wait(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !predicate() {
        assert!(
            Instant::now() < deadline,
            "transport ownership boundary timed out"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn exchange(address: SocketAddr, request: &str) -> String {
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let request = request.replacen("\r\n\r\n", "\r\nConnection: close\r\n\r\n", 1);
    socket.write_all(request.as_bytes()).unwrap();
    let mut raw = String::new();
    socket.read_to_string(&mut raw).unwrap();
    raw
}
fn get(path: &str, auth: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}\r\n")
}
fn post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}
fn document(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw.split_once("\r\n\r\n").unwrap().1).unwrap()
}

#[test]
fn cold_build_live_port_zero_explicit_auth_errors_and_metrics() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let cold = ServerBuilder::from_service(test_support::idle(&host))
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .build()
        .unwrap();
    assert_eq!(cold.address().port(), 0);
    assert_eq!(host.serving_bytes(), baseline);
    let server = cold.start().unwrap();
    let address = server.local_addr();
    assert_ne!(address.port(), 0);
    let unauthorized = exchange(address, &get("/health", ""));
    assert!(unauthorized.starts_with("HTTP/1.1 401"));
    assert_eq!(document(&unauthorized)["error"]["code"], "unauthorized");
    let auth = "Authorization: Bearer secret\r\n";
    let health = exchange(address, &get("/health", auth));
    assert!(health.starts_with("HTTP/1.1 200"));
    assert_eq!(
        document(&health)["admission"]["kind"],
        "transport_test_only"
    );
    let models = exchange(address, &get("/v1/models", auth));
    assert_eq!(document(&models)["data"], serde_json::json!([]));
    let missing = exchange(address, &get("/v1/responses", auth));
    assert_eq!(document(&missing)["error"]["code"], "not_found");
    wait(|| server.metrics().active_connections == 0);
    assert_eq!(server.metrics().accepted_requests, 4);
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
    assert!(TcpStream::connect(address).is_err());
}

#[test]
fn configured_connection_and_body_limits_are_real_and_drop_joins() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let server = ServerBuilder::from_service(test_support::idle(&host))
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(IngressLimits {
            connections: 1,
            body_bytes: 8,
            output_bytes: 4096,
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    // Only headers: declared length must be refused before any body read.
    let too_large = exchange(
        address,
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 9\r\n\r\n",
    );
    assert!(too_large.starts_with("HTTP/1.1 413"));
    // Allow the finished handler to be reaped before filling the sole slot.
    wait(|| server.metrics().active_connections == 0);
    std::thread::sleep(Duration::from_millis(60));
    let mut idle = TcpStream::connect(address).unwrap();
    wait(|| server.metrics().active_connections == 1);
    let mut excess = TcpStream::connect(address).unwrap();
    excess
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut response = String::new();
    excess.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 503"));
    assert_eq!(document(&response)["error"]["code"], "connections_full");
    drop(server);
    idle.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    assert_eq!(idle.read(&mut [0]).unwrap(), 0);
    assert_eq!(host.serving_bytes(), baseline);
}

struct Script {
    host: ServingHost,
    requests: Mutex<Vec<(SemanticRequest, Vec<ServingCharge>)>>,
    cancellations: Mutex<Vec<CancelToken>>,
    raw_only: bool,
    failure: Option<ServiceError>,
    fail_after_delta: bool,
    finish_reason: &'static str,
    cached_prompt_tokens: usize,
}
impl Script {
    fn new(host: ServingHost) -> Arc<Self> {
        Arc::new(Self {
            host,
            requests: Mutex::new(Vec::new()),
            cancellations: Mutex::new(Vec::new()),
            raw_only: false,
            failure: None,
            fail_after_delta: false,
            finish_reason: "stop",
            cached_prompt_tokens: 0,
        })
    }
}
impl TextService for Script {
    fn validate_chat(&self, _: &str) -> std::result::Result<(), ServiceError> {
        if self.raw_only {
            Err(ServiceError::new(400, "unsupported_chat", "raw text only"))
        } else {
            Ok(())
        }
    }
    fn charge_ingress(
        &self,
        session: &str,
        bytes: u64,
    ) -> std::result::Result<ServingCharge, ServiceError> {
        self.host.charge(
            "scripted-transport",
            session,
            zerogravity_core::resource_arbiter::AllocationClass::Scratch,
            bytes,
        )
    }
    fn model_ids(&self) -> std::result::Result<Vec<String>, ServiceError> {
        Ok(vec!["fixture".into()])
    }
    fn max_output(&self) -> std::result::Result<usize, ServiceError> {
        Ok(4)
    }
    fn admission(&self) -> std::result::Result<serde_json::Value, ServiceError> {
        Ok(serde_json::json!({"kind":"scripted_transport_only"}))
    }
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        self.request_semantic_admitted(request.into(), ingress)
    }
    fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        if request.input.is_chat() {
            self.validate_chat(&request.model)?;
        }
        if request.model != "fixture" {
            return Err(ServiceError::new(
                404,
                "model_not_found",
                "fixture alias missing",
            ));
        }
        let (tx, rx) = mpsc::try_sync_channel(&self.host, "fixture", "events", 4)?;
        if let Some(error) = &self.failure {
            if request.stream && self.fail_after_delta {
                tx.try_send(ResponseEvent::Text("partial text".into()))
                    .unwrap_or_else(|_| panic!("fixture channel"));
            }
            tx.try_send(ResponseEvent::Error(error.clone()))
                .unwrap_or_else(|_| panic!("fixture channel"));
            return Ok(RequestEvents::new(
                12,
                request.model,
                rx,
                test_support::cancel(&self.host),
                None,
                None,
            ));
        }
        let observations = request.token_metrics.clone();
        let (output, completion_tokens) = if observations.is_some() {
            for (index, id, text) in [(0, 4, ""), (1, 5, "é")] {
                tx.try_send(ResponseEvent::TokenText {
                    text: text.into(),
                    token: zerogravity_engine::router::TokenObservation {
                        index,
                        id,
                        commit_observed_at: Instant::now(),
                        gpu: None,
                    },
                })
                .unwrap_or_else(|_| panic!("fixture channel"));
            }
            ("é", 2)
        } else {
            if request.stream {
                tx.try_send(ResponseEvent::Text("transport fixture".into()))
                    .unwrap_or_else(|_| panic!("fixture channel"));
            }
            ("transport fixture", 1)
        };
        tx.try_send(ResponseEvent::Done {
            output: output.into(),
            prompt_tokens: 3,
            cached_prompt_tokens: self.cached_prompt_tokens,
            completion_tokens,
            finish_reason: self.finish_reason,
        })
        .unwrap_or_else(|_| panic!("fixture channel"));
        let cancel = test_support::cancel(&self.host);
        self.cancellations.lock().unwrap().push(cancel.clone());
        let events = RequestEvents::new(12, request.model.clone(), rx, cancel, None, None)
            .observed(observations);
        self.requests.lock().unwrap().push((request, ingress));
        Ok(events)
    }
}

#[test]
fn actual_http_and_sse_error_details_are_opt_in_authenticated_and_logs_are_safe() {
    use zerogravity_engine::router::FailureDetails;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    assert!(matches!(
        ServerBuilder::from_service(test_support::idle(&host))
            .enable_error_details(true)
            .build(),
        Err(Error::InvalidInput(_))
    ));
    for enabled in [false, true] {
        let mut service = Script::new(host.clone());
        Arc::get_mut(&mut service).unwrap().failure =
            Some(ServiceError::inference(Error::Resource(
                zerogravity_core::ResourceLimit::new("PRIVATE-PATH", "model admission", 4096, 2048)
                    .with_scope("PRIVATE-SECRET"),
            )));
        let observations = Arc::new(Mutex::new(Vec::<FailureDetails>::new()));
        let observed = Arc::clone(&observations);
        let server = ServerBuilder::from_service(service)
            .bind("127.0.0.1:0".parse().unwrap())
            .api_key(Some("fixture-key".into()))
            .enable_error_details(enabled)
            .observe_errors(move |details| observed.lock().unwrap().push(details))
            .build()
            .unwrap()
            .start()
            .unwrap();
        let address = server.local_addr();
        let unauthorized = exchange(address, &get("/health", ""));
        assert!(unauthorized.starts_with("HTTP/1.1 401"));
        assert!(document(&unauthorized)["error"].get("details").is_none());
        let before_auth = exchange(
            address,
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 999999999\r\n\r\n",
        );
        assert!(document(&before_auth)["error"].get("details").is_none());
        for stream in [false, true] {
            let body = format!(
                r#"{{"model":"fixture","prompt":"hello","max_tokens":1,"stream":{stream}}}"#
            );
            let request = post("/v1/completions", &body).replacen(
                "Host: localhost\r\n",
                "Host: localhost\r\nAuthorization: Bearer fixture-key\r\n",
                1,
            );
            let response = exchange(address, &request);
            let error = if stream {
                assert!(response.starts_with("HTTP/1.1 200"));
                let line = response
                    .lines()
                    .find(|line| line.starts_with("data: ") && line.contains("\"error\""))
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(&line[6..]).unwrap()["error"].clone()
            } else {
                assert!(response.starts_with("HTTP/1.1 503"));
                document(&response)["error"].clone()
            };
            assert_eq!(error["code"], "resource_exhausted");
            assert_eq!(error["type"], "server_error");
            assert!(!error["message"].as_str().unwrap().contains("PRIVATE"));
            if enabled {
                assert_eq!(error["details"]["status"], 503);
                assert_eq!(error["details"]["requested_bytes"], 4096);
                assert_eq!(error["details"]["available_bytes"], 2048);
                assert!(
                    error["details"]["diagnostic"]
                        .as_str()
                        .unwrap()
                        .contains("PRIVATE-PATH")
                );
            } else {
                assert!(error.get("details").is_none());
                assert!(!response.contains("PRIVATE"));
            }
        }
        server.shutdown().unwrap();
        let entries = observations.lock().unwrap();
        assert_eq!(
            entries
                .iter()
                .filter(|d| d.code == "resource_exhausted")
                .count(),
            2
        );
        assert!(
            entries
                .iter()
                .filter(|d| d.code == "resource_exhausted")
                .all(|d| d.requested_bytes == Some(4096) && d.available_bytes == Some(2048))
        );
        assert!(
            !serde_json::to_string(&*entries)
                .unwrap()
                .contains("PRIVATE")
        );
        assert_eq!(host.serving_bytes(), baseline);
    }
}

#[test]
fn framed_chat_preserves_source_refusal_before_enqueue() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let mut service = Script::new(host.clone());
    Arc::get_mut(&mut service).unwrap().raw_only = true;
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap();
    let body = r#"{"model":"fixture","messages":[{"role":"user","content":"hello"}]}"#;
    let response = exchange(server.local_addr(), &post("/v1/chat/completions", body));
    assert!(response.starts_with("HTTP/1.1 400"));
    assert_eq!(document(&response)["error"]["code"], "unsupported_chat");
    assert!(service.requests.lock().unwrap().is_empty());
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn configured_prefix_scope_refuses_unaware_text_service() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let service = Script::new(host);
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .prefix_sharing_scope(Some(
            zerogravity_engine::state_cache::PrefixSharingScope::new(7).unwrap(),
        ))
        .build()
        .unwrap()
        .start()
        .unwrap();
    let body = r#"{"model":"fixture","prompt":"hello","max_tokens":1}"#;
    let response = exchange(server.local_addr(), &post("/v1/completions", body));
    assert!(response.starts_with("HTTP/1.1 400"));
    assert_eq!(
        document(&response)["error"]["code"],
        "prefix_sharing_unsupported"
    );
    assert!(service.requests.lock().unwrap().is_empty());
    server.shutdown().unwrap();
}

#[test]
fn actual_json_and_chunked_sse_use_one_service_and_preserve_chat_framing() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    let body = r#"{"model":"fixture","prompt":"hello"}"#;
    let json = exchange(address, &post("/v1/completions", body));
    assert!(json.starts_with("HTTP/1.1 200"));
    let json = document(&json);
    assert_eq!(json["choices"][0]["text"], "transport fixture");
    assert_eq!(json["usage"]["total_tokens"], 4);
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    let chat = r#"{"model":"fixture","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}"#;
    let sse = exchange(address, &post("/v1/chat/completions", chat));
    assert!(sse.contains("Transfer-Encoding: chunked\r\n"));
    assert!(sse.contains("data: [DONE]\n\n"));
    assert!(sse.ends_with("0\r\n\r\n"));
    let mut chunks = sse.split_once("\r\n\r\n").unwrap().1;
    while !chunks.starts_with("0\r\n") {
        let (size, rest) = chunks.split_once("\r\n").unwrap();
        let size = usize::from_str_radix(size, 16).unwrap();
        assert_eq!(&rest[size..size + 2], "\r\n");
        assert!(rest[..size].starts_with("data: "));
        chunks = &rest[size + 2..];
    }
    let requests = service.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].0.max_tokens, 4);
    assert_eq!(
        requests[1].0.input,
        TextInput::Conversation(vec![ChatMessage {
            role: ChatRole::User,
            text: "hello".into()
        }])
    );
    drop(requests);
    let refusal = exchange(
        address,
        &post("/v1/completions", r#"{"model":"missing","prompt":"x"}"#),
    );
    assert_eq!(document(&refusal)["error"]["code"], "model_not_found");
    server.shutdown().unwrap();
    assert!(
        service
            .cancellations
            .lock()
            .unwrap()
            .iter()
            .all(CancelToken::is_cancelled)
    );
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn output_byte_limit_closes_json_and_sse_and_cancels_original_receipt() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    for streaming in [false, true] {
        let service = Script::new(host.clone());
        let server = ServerBuilder::from_service(service.clone())
            .bind("127.0.0.1:0".parse().unwrap())
            .limits(IngressLimits {
                output_bytes: 256,
                ..Default::default()
            })
            .build()
            .unwrap()
            .start()
            .unwrap();
        let body =
            serde_json::json!({"model":"fixture", "prompt":"x", "stream":streaming}).to_string();
        let response = exchange(server.local_addr(), &post("/v1/completions", &body));
        assert!(response.len() <= 256);
        assert!(!response.contains("[DONE]"));
        wait(|| {
            let cancellations = service.cancellations.lock().unwrap();
            cancellations.len() == 1 && cancellations.iter().all(CancelToken::is_cancelled)
        });
        server.shutdown().unwrap();
        drop(service);
        assert_eq!(host.serving_bytes(), baseline);
    }
}

#[test]
fn failed_bind_releases_all_transport_inventory_without_stopping_service() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let service = test_support::idle(&host);
    assert!(
        ServerBuilder::from_service(service.clone())
            .bind(listener.local_addr().unwrap())
            .build()
            .unwrap()
            .start()
            .is_err()
    );
    assert_eq!(host.serving_bytes(), baseline);
    let server = ServerBuilder::from_service(service)
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap();
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
}

/// Read exactly one ordinary HTTP response without waiting for connection
/// retirement. This intentionally does not assume one socket per response.
fn read_response(reader: &mut BufReader<TcpStream>) -> (String, String) {
    let mut head = String::new();
    loop {
        let mut line = String::new();
        assert_ne!(
            reader.read_line(&mut line).unwrap(),
            0,
            "short response headers"
        );
        head.push_str(&line);
        assert!(head.len() <= 8192);
        if line == "\r\n" {
            break;
        }
    }
    let lower = head.to_ascii_lowercase();
    let mut body = Vec::new();
    if lower.contains("transfer-encoding: chunked\r\n") {
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let size = usize::from_str_radix(line.trim(), 16).unwrap();
            if size == 0 {
                let mut end = [0; 2];
                reader.read_exact(&mut end).unwrap();
                assert_eq!(&end, b"\r\n");
                break;
            }
            let start = body.len();
            assert!(start + size <= 1 << 20);
            body.resize(start + size, 0);
            reader.read_exact(&mut body[start..]).unwrap();
            let mut end = [0; 2];
            reader.read_exact(&mut end).unwrap();
            assert_eq!(&end, b"\r\n");
        }
    } else {
        let len = lower
            .lines()
            .find_map(|line| line.strip_prefix("content-length: "))
            .expect("bounded length")
            .parse::<usize>()
            .unwrap();
        assert!(len <= 1 << 20);
        body.resize(len, 0);
        reader.read_exact(&mut body).unwrap();
    }
    (head, String::from_utf8(body).unwrap())
}

#[test]
fn persistent_socket_delivers_split_json_token_sse_and_next_request() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let stream = TcpStream::connect(server.local_addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut client = BufReader::new(stream);

    client
        .get_mut()
        .write_all(get("/v1/models", "").as_bytes())
        .unwrap();
    let (head, body) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert!(!head.to_ascii_lowercase().contains("connection: close"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["data"][0]["id"],
        "fixture"
    );

    // Header and a partial body arrive together, then the remaining bytes.
    let request = post("/v1/completions", r#"{"model":"fixture","prompt":"hello"}"#);
    let split = request.find("\r\n\r\n").unwrap() + 6;
    client
        .get_mut()
        .write_all(&request.as_bytes()[..split])
        .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(service.requests.lock().unwrap().is_empty());
    client
        .get_mut()
        .write_all(&request.as_bytes()[split..])
        .unwrap();
    let (head, body) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["choices"][0]["text"],
        "transport fixture"
    );

    let chat = r#"{"model":"fixture","messages":[{"role":"user","content":"hello"}],"stream":true,"return_token_ids":true}"#;
    let request = post("/v1/chat/completions", chat).replacen(
        "Host: localhost\r\n",
        "Host: localhost\r\nX-Request-ID: run:request-0\r\n",
        1,
    );
    // A next request may be buffered while streaming; it must not interrupt
    // the original response or be mistaken for a disconnect.
    let request = request + &get("/health", "");
    client.get_mut().write_all(request.as_bytes()).unwrap();
    let (head, body) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert!(
        head.to_ascii_lowercase()
            .contains("x-request-id: run:request-0\r\n")
    );
    assert!(body.ends_with("data: [DONE]\n\n"));
    let events = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(events[0]["object"], "chat.completion.chunk");
    assert_eq!(events[0]["id"], "chatcmpl-zg-12");
    let ids = events
        .iter()
        .flat_map(|event| event["choices"].as_array().unwrap())
        .flat_map(|choice| choice["token_ids"].as_array().into_iter().flatten())
        .map(|id| id.as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids, [4, 5]);
    let usage = events
        .iter()
        .find(|event| event.get("usage").is_some())
        .unwrap();
    let metrics = &usage["zerogravity_token_metrics"];
    assert_eq!(metrics["provider_response_id"], "chatcmpl-zg-12");
    assert_eq!(metrics["wire_request_id"], "run:request-0");
    assert_eq!(metrics["completion_tokens"], 2);
    let mut last = 0.0;
    for token in metrics["tokens"].as_array().unwrap() {
        let commit = token["commit_observed_at_seconds"].as_f64().unwrap();
        let received = token["output_received_at_seconds"].as_f64().unwrap();
        let written = token["http_write_completed_at_seconds"].as_f64().unwrap();
        assert!(last <= received && commit <= received && received <= written);
        last = written;
    }
    assert_eq!(
        events
            .iter()
            .flat_map(|event| event["choices"].as_array().unwrap())
            .filter(|choice| !choice["finish_reason"].is_null())
            .count(),
        1
    );
    let (head, body) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["status"],
        "ok"
    );
    assert_eq!(server.metrics().accepted_requests, 4);
    assert_eq!(server.metrics().active_connections, 1);
    drop(client);
    wait(|| server.metrics().active_connections == 0);
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn actual_framing_refuses_ambiguity_and_preauth_body_without_admitting() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .build()
        .unwrap()
        .start()
        .unwrap();
    let addr = server.local_addr();
    let unauthorized = exchange(
        addr,
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 32\r\n\r\n",
    );
    assert!(unauthorized.starts_with("HTTP/1.1 401"));
    assert_eq!(document(&unauthorized)["error"]["code"], "unauthorized");
    for raw in [
        "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n",
        "POST / HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n",
        "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: -1\r\n\r\n",
        "GET /health HTTP/1.1\r\nHost: localhost\r\nHost: other\r\n\r\n",
        "POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n",
        "GET /health HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nAuthorization: Bearer other\r\n\r\n",
        "POST / HTTP/1.1\r\nHost: localhost\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n",
    ] {
        let response = exchange(addr, raw);
        assert!(response.starts_with("HTTP/1.1 400"), "{raw:?}: {response}");
        assert!(!response.contains("100 Continue"));
    }
    let response = exchange(
        addr,
        "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1048577\r\n\r\n",
    );
    assert!(response.starts_with("HTTP/1.1 413"));
    let response = exchange(
        addr,
        &format!(
            "GET /health HTTP/1.1\r\nHost: localhost\r\nX-Large: {}\r\n\r\n",
            "a".repeat(8192)
        ),
    );
    assert!(response.starts_with("HTTP/1.1 431") || response.starts_with("HTTP/1.1 400"));
    let mut short = TcpStream::connect(addr).unwrap();
    short
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    short.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{").unwrap();
    short.shutdown(std::net::Shutdown::Write).unwrap();
    let mut response = String::new();
    let _ = short.read_to_string(&mut response);
    assert!(response.is_empty() || response.starts_with("HTTP/1.1 400"));
    assert!(service.requests.lock().unwrap().is_empty());
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

fn chunked_head(path: &str, auth: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n{auth}\r\n"
    )
}

fn write_chunk(socket: &mut TcpStream, bytes: &[u8]) {
    write!(socket, "{:x}\r\n", bytes.len()).unwrap();
    socket.write_all(bytes).unwrap();
    socket.write_all(b"\r\n").unwrap();
}

#[test]
fn chunked_text_routes_admit_only_after_complete_body_and_reuse_connection() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .limits(IngressLimits {
            body_bytes: 128,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let auth = "Authorization: Bearer secret\r\n";
    let stream = TcpStream::connect(server.local_addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut client = BufReader::new(stream);
    let mut raw = br#"{"model":"fixture","prompt":"hello","max_tokens":2}"#.to_vec();
    // Exactly the decoded byte cap, not the HTTP chunk framing bytes.
    raw.resize(128, b' ');
    client
        .get_mut()
        .write_all(chunked_head("/v1/completions", auth).as_bytes())
        .unwrap();
    write_chunk(client.get_mut(), &raw[..17]);
    wait(|| server.metrics().accepted_requests == 1);
    assert!(service.requests.lock().unwrap().is_empty());
    for piece in raw[17..].chunks(19) {
        write_chunk(client.get_mut(), piece);
    }
    client.get_mut().write_all(b"0\r\n\r\n").unwrap();
    client
        .get_mut()
        .write_all(get("/v1/models", auth).as_bytes())
        .unwrap();
    let (head, result) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert!(!head.to_ascii_lowercase().contains("connection: close"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&result).unwrap()["choices"][0]["text"],
        "transport fixture"
    );
    let (head, result) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&result).unwrap()["data"][0]["id"],
        "fixture"
    );

    // Split a UTF-8 code point across separate decoded frames. The same
    // original chat lowering and SSE path must receive the full JSON.
    let chat = r#"{"model":"fixture","messages":[{"role":"user","content":"héllo"}],"stream":true,"max_tokens":2}"#;
    let split = chat.find('é').unwrap() + 1;
    client
        .get_mut()
        .write_all(chunked_head("/v1/chat/completions", auth).as_bytes())
        .unwrap();
    write_chunk(client.get_mut(), &chat.as_bytes()[..split]);
    write_chunk(client.get_mut(), &chat.as_bytes()[split..]);
    client.get_mut().write_all(b"0\r\n\r\n").unwrap();
    let (head, result) = read_response(&mut client);
    assert!(head.starts_with("HTTP/1.1 200"));
    assert!(result.contains("transport fixture") && result.ends_with("data: [DONE]\n\n"));
    let requests = service.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].0.input, TextInput::Raw("hello".into()));
    assert!(!requests[0].0.input.is_chat() && !requests[0].0.stream);
    assert_eq!(
        requests[1].0.input,
        TextInput::Conversation(vec![ChatMessage {
            role: ChatRole::User,
            text: "héllo".into()
        }])
    );
    assert!(requests[1].0.input.is_chat() && requests[1].0.stream);
    drop(requests);
    assert_eq!(server.metrics().active_connections, 1);
    assert_eq!(server.metrics().accepted_requests, 3);
    drop(client);
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn chunked_ingress_refuses_bad_framing_auth_and_bounds_before_model_admission() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .limits(IngressLimits {
            body_bytes: 128,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    let auth = "Authorization: Bearer secret\r\n";
    let idle = host.serving_bytes();
    let head = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{auth}"
    );
    // These must fail in the actual Hyper parser, before the service sees a
    // normalized head, even when both lengths agree or TE occurs first.
    for framing in [
        "Content-Length: 64\r\nTransfer-Encoding: chunked\r\n",
        "Transfer-Encoding: chunked\r\nContent-Length: 64\r\n",
        "Transfer-Encoding: chunked\r\nContent-Length: invalid\r\n",
        "Content-Length: 64\r\nContent-Length: 64\r\n",
        "Content-Length: 64\r\nContent-Length: 65\r\n",
    ] {
        let response = exchange(address, &format!("{head}{framing}\r\n"));
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "{framing:?}: {response}"
        );
    }
    assert_eq!(server.metrics().accepted_requests, 0);
    for framing in [
        "Transfer-Encoding: gzip, chunked\r\n",
        "Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n",
        "Transfer-Encoding: chunked\r\nTrailer: X-Trace\r\n",
        "Transfer-Encoding: chunked\r\nExpect: 100-continue\r\n",
    ] {
        let response = exchange(address, &format!("{head}{framing}\r\n"));
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "{framing:?}: {response}"
        );
        assert!(!response.contains("100 Continue"));
    }
    let response = exchange(address, &chunked_head("/v1/completions", ""));
    assert!(response.starts_with("HTTP/1.1 401"));
    assert_eq!(document(&response)["error"]["code"], "unauthorized");
    let response = exchange(address, &chunked_head("/v1/audio/transcriptions", auth));
    assert!(response.starts_with("HTTP/1.1 400"));

    // Actual decoded aggregate overflow, malformed chunks and trailers
    // cannot publish a request, even after receiving valid initial bytes.
    let head = chunked_head("/v1/completions", auth);
    let oversized = format!(
        "{head}40\r\n{}\r\n41\r\n{}\r\n0\r\n\r\n",
        "x".repeat(64),
        "x".repeat(65)
    );
    let response = exchange(address, &oversized);
    assert!(response.starts_with("HTTP/1.1 413"), "{response}");
    for suffix in [
        "not-a-size\r\n",
        "4\r\nabcd\r\n0\r\nX-Trace: ignored\r\n\r\n",
    ] {
        let response = exchange(address, &format!("{head}{suffix}"));
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    }
    wait(|| server.metrics().active_connections == 0);
    wait(|| host.serving_bytes() == idle);
    assert!(service.requests.lock().unwrap().is_empty());

    // A partial upload owns only bounded staging, not a model request.
    let mut client = TcpStream::connect(address).unwrap();
    client.write_all(head.as_bytes()).unwrap();
    write_chunk(&mut client, b"partial");
    wait(|| server.metrics().active_connections == 1 && host.serving_bytes() > idle);
    drop(client);
    wait(|| server.metrics().active_connections == 0);
    wait(|| host.serving_bytes() == idle);
    assert!(service.requests.lock().unwrap().is_empty());
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn persistent_wire_budget_is_not_reset_by_next_request() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let server = ServerBuilder::from_service(test_support::idle(&host))
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(IngressLimits {
            output_bytes: 512,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let mut client = TcpStream::connect(server.local_addr()).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    client
        .write_all(get("/health", "").repeat(8).as_bytes())
        .unwrap();
    let mut raw = Vec::new();
    let _ = client.read_to_end(&mut raw);
    assert!(raw.starts_with(b"HTTP/1.1 200") && raw.len() <= 512);
    wait(|| server.metrics().active_connections == 0);
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn multipart_foundation_does_not_publish_media_or_bypass_text_content_policy() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .build()
        .unwrap()
        .start()
        .unwrap();
    let form = "--example\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.bin\"\r\n\r\nhello\r\n--example--\r\n";
    for (path, status) in [("/v1/audio/transcriptions", 404), ("/v1/completions", 415)] {
        let raw = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: multipart/form-data; boundary=example\r\nContent-Length: {}\r\n\r\n{form}",
            form.len()
        );
        let response = exchange(server.local_addr(), &raw);
        assert!(
            response.starts_with(&format!("HTTP/1.1 {status}")),
            "{response}"
        );
    }
    assert!(service.requests.lock().unwrap().is_empty());
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

// Responses is a protocol projection over the same original typed receipt.
fn responses_request(
    client: &mut BufReader<TcpStream>,
    value: &serde_json::Value,
    chunked: bool,
) -> (String, serde_json::Value) {
    let data = value.to_string();
    let socket = client.get_mut();
    if chunked {
        write!(socket, "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        for part in data.as_bytes().chunks(17) {
            write!(socket, "{:x}\r\n", part.len()).unwrap();
            socket.write_all(part).unwrap();
            socket.write_all(b"\r\n").unwrap();
        }
        socket.write_all(b"0\r\n\r\n").unwrap();
    } else {
        write!(socket, "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{data}", data.len()).unwrap();
    }
    let (head, body) = read_response(client);
    (head, serde_json::from_str(&body).unwrap())
}
fn response_client(address: std::net::SocketAddr) -> BufReader<TcpStream> {
    let socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    BufReader::new(socket)
}
fn correlation(head: &str) -> &str {
    head.lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("x-request-id")
                .then(|| value.trim())
        })
        .expect("server request correlation")
}
fn request_response_events(address: std::net::SocketAddr) -> (String, Vec<serde_json::Value>) {
    let mut client = response_client(address);
    let request = r#"{"model":"fixture","input":"hello","stream":true,"store":false}"#;
    write!(client.get_mut(), "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{request}", request.len()).unwrap();
    let (head, body) = read_response(&mut client);
    assert!(
        head.starts_with("HTTP/1.1 200") && head.to_ascii_lowercase().contains("text/event-stream")
    );
    let events = body
        .split("\n\n")
        .filter(|event| !event.is_empty())
        .enumerate()
        .map(|(sequence, event)| {
            let (kind, data) = event.split_once("\ndata: ").expect("named Responses event");
            let value: serde_json::Value = serde_json::from_str(data).unwrap();
            assert_eq!(kind.strip_prefix("event: "), value["type"].as_str());
            assert_eq!(value["sequence_number"], sequence);
            value
        })
        .collect();
    (head, events)
}

#[test]
fn models_retrieve_and_stateless_responses_share_original_semantic_admission() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let mut service = Script::new(host.clone());
    Arc::get_mut(&mut service).unwrap().cached_prompt_tokens = 2;
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap();
    let mut client = response_client(server.local_addr());
    client
        .get_mut()
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let (head, list) = read_response(&mut client);
    let first_id = correlation(&head).to_owned();
    let list: serde_json::Value = serde_json::from_str(&list).unwrap();
    client
        .get_mut()
        .write_all(b"GET /v1/models/fixture HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let (head, model) = read_response(&mut client);
    assert_ne!(correlation(&head), first_id);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&model).unwrap(),
        list["data"][0]
    );
    let request = json!({
        "model":"fixture","instructions":"Brief.","input":[
            {"role":"user","content":[{"type":"input_text","text":"hé"},{"type":"input_text","text":"llo"}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi","annotations":[],"logprobs":[]}],"status":"completed","id":"msg_explicit"},
            {"role":"user","content":"again"}
        ],"store":false,"max_output_tokens":2,"temperature":0,"top_p":1,
        "text":{"format":{"type":"text"}},"tools":[],"tool_choice":"none","metadata":{"label":"test"}
    });
    let (head, response) = responses_request(&mut client, &request, true);
    assert!(head.starts_with("HTTP/1.1 200"), "{response}");
    assert!(correlation(&head).starts_with("req_zg_"));
    assert_eq!(
        response["id"].as_str().unwrap(),
        correlation(&head).replacen("req_", "resp_", 1)
    );
    assert_eq!(response["object"], "response");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["store"], false);
    assert_eq!(response["instructions"], "Brief.");
    assert_eq!(response["metadata"], request["metadata"]);
    assert_eq!(
        response["output"][0]["content"][0]["text"],
        "transport fixture"
    );
    assert_eq!(
        response["usage"],
        json!({"input_tokens":3,"input_tokens_details":{"cached_tokens":2},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":4})
    );
    let requests = service.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].0.input,
        TextInput::Conversation(vec![
            ChatMessage {
                role: ChatRole::Developer,
                text: "Brief.".into()
            },
            ChatMessage {
                role: ChatRole::User,
                text: "héllo".into()
            },
            ChatMessage {
                role: ChatRole::Assistant,
                text: "Hi".into()
            },
            ChatMessage {
                role: ChatRole::User,
                text: "again".into()
            },
        ])
    );
    drop(requests);
    drop(client);
    let missing = exchange(
        server.local_addr(),
        "GET /v1/models/missing HTTP/1.1\r\nHost: localhost\r\n\r\n",
    );
    assert!(missing.starts_with("HTTP/1.1 404"));
    assert_eq!(document(&missing)["error"]["param"], "model");
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn responses_stream_projects_named_monotonic_events_and_truthful_terminal_states() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    for finish in ["stop", "length", "error"] {
        let mut service = Script::new(host.clone());
        let fixture = Arc::get_mut(&mut service).unwrap();
        fixture.finish_reason = finish;
        if finish == "error" {
            fixture.failure = Some(ServiceError::new(
                500,
                "inference_error",
                "private backend detail",
            ));
            fixture.fail_after_delta = true;
        }
        let server = ServerBuilder::from_service(service.clone())
            .bind("127.0.0.1:0".parse().unwrap())
            .build()
            .unwrap()
            .start()
            .unwrap();
        let (head, events) = request_response_events(server.local_addr());
        let id = correlation(&head).replacen("req_", "resp_", 1);
        let kinds = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect::<Vec<_>>();
        let mut expected = vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
        ];
        if finish != "error" {
            expected.extend([
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
            ]);
        }
        expected.push(match finish {
            "stop" => "response.completed",
            "length" => "response.incomplete",
            _ => "response.failed",
        });
        assert_eq!(kinds, expected);
        assert_eq!(events[0]["response"]["id"], id);
        let terminal = &events.last().unwrap()["response"];
        assert_eq!(terminal["id"], id);
        assert_eq!(events[2]["item"]["id"], terminal["output"][0]["id"]);
        assert_eq!(events[4]["item_id"], terminal["output"][0]["id"]);
        assert_eq!(
            events[4]["delta"],
            terminal["output"][0]["content"][0]["text"]
        );
        if finish == "length" {
            assert_eq!(terminal["status"], "incomplete");
            assert_eq!(
                terminal["incomplete_details"]["reason"],
                "max_output_tokens"
            );
            assert_eq!(terminal["output"][0]["status"], "incomplete");
        } else if finish == "error" {
            assert_eq!(terminal["status"], "failed");
            assert_eq!(terminal["error"]["code"], "server_error");
            assert!(terminal["usage"].is_null());
            assert!(!terminal.to_string().contains("private backend detail"));
        } else {
            assert_eq!(terminal["status"], "completed");
            assert_eq!(terminal["usage"]["total_tokens"], 4);
        }
        server.shutdown().unwrap();
        drop(service);
        assert_eq!(host.serving_bytes(), baseline);
    }
}

#[test]
fn responses_refuses_unsupported_state_controls_and_content_before_enqueue() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = Script::new(host.clone());
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap();
    for (key, value) in [
        ("store", json!(true)),
        ("background", json!(true)),
        ("previous_response_id", json!("resp_prior")),
        ("conversation", json!("conv_prior")),
        ("temperature", json!(0.7)),
        ("reasoning", json!({"effort":"high"})),
        ("tools", json!([{"type":"web_search"}])),
        ("text", json!({"format":{"type":"json_schema"}})),
        ("max_output_tokens", json!(5)),
        (
            "input",
            json!([{"role":"user","content":[{"type":"input_image","image_url":"not-a-local-resource"}]}]),
        ),
    ] {
        let mut request = json!({"model":"fixture","input":"hello"});
        request[key] = value;
        let (head, body) =
            responses_request(&mut response_client(server.local_addr()), &request, false);
        assert!(head.starts_with("HTTP/1.1 400"), "{body}");
        assert_eq!(body["error"]["param"], key);
        assert!(correlation(&head).starts_with("req_zg_"));
    }
    assert!(service.requests.lock().unwrap().is_empty());
    server.shutdown().unwrap();
    drop(service);
    assert_eq!(host.serving_bytes(), baseline);
}
