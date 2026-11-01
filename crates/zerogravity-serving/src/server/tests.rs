//! Socket/lifecycle controls only. Scripted events do not qualify model math.
use super::*;
use crate::{
    service::{Request, RequestEvents, ServiceError, ServingCharge},
    test_support::{self, HostBytes},
};
use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Mutex,
    time::{Duration, Instant},
};
use zerogravity_engine::router::{
    ResponseEvent,
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
    requests: Mutex<Vec<(Request, Vec<ServingCharge>)>>,
    cancellations: Mutex<Vec<CancelToken>>,
}
impl Script {
    fn new(host: ServingHost) -> Arc<Self> {
        Arc::new(Self {
            host,
            requests: Mutex::new(Vec::new()),
            cancellations: Mutex::new(Vec::new()),
        })
    }
}
impl TextService for Script {
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
        if request.model != "fixture" {
            return Err(ServiceError::new(
                404,
                "model_not_found",
                "fixture alias missing",
            ));
        }
        let (tx, rx) = mpsc::try_sync_channel(&self.host, "fixture", "events", 2)?;
        if request.stream {
            tx.try_send(ResponseEvent::Text("transport fixture".into()))
                .unwrap_or_else(|_| panic!("fixture channel"));
        }
        tx.try_send(ResponseEvent::Done {
            output: "transport fixture".into(),
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: "stop",
        })
        .unwrap_or_else(|_| panic!("fixture channel"));
        let cancel = test_support::cancel(&self.host);
        self.cancellations.lock().unwrap().push(cancel.clone());
        let events = RequestEvents::new(12, request.model.clone(), rx, cancel, None, None);
        self.requests.lock().unwrap().push((request, ingress));
        Ok(events)
    }
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
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
    assert!(!requests[1].0.chat);
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
