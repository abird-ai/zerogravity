//! Actual async transport ownership tests. The queue/events below deliberately
//! hold original cancellation receipts; they do not qualify model arithmetic.
use crate::{
    ServerBuilder, ServerHandle, TextService,
    service::{Request, RequestEvents, SemanticRequest, ServiceError, ServingCharge},
    test_support::{self, HostBytes},
};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use zerogravity_core::resource_arbiter::AllocationClass;
use zerogravity_engine::router::{
    ResponseEvent,
    event_channel::mpsc,
    pending::{CancelToken, PendingQueue, Pop, ServingHost},
};

fn wait_until(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "control did not reach its ownership boundary"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn start(service: Arc<dyn TextService>) -> ServerHandle {
    ServerBuilder::from_service(service)
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap()
}

fn assert_closed(mut client: TcpStream) {
    client
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    let mut bytes = Vec::new();
    match client.read_to_end(&mut bytes) {
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe
            ) => {}
        other => panic!("socket survived joined shutdown: {other:?}"),
    }
}

#[test]
fn idle_listener_stops_without_a_wakeup_connection() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let server = start(test_support::idle(&host));
    let address = server.local_addr();
    assert_ne!(address.port(), 0);
    assert!(host.serving_bytes() > baseline);
    server.stop();
    server.join().unwrap();
    assert!(TcpStream::connect(address).is_err());
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn shutdown_interrupts_idle_partial_header_and_partial_body_then_joins_tasks() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let server = start(test_support::idle(&host));
    let address = server.local_addr();
    let idle = TcpStream::connect(address).unwrap();
    let mut header = TcpStream::connect(address).unwrap();
    header
        .write_all(b"POST /v1/completions HTTP/1.1\r\nHost:")
        .unwrap();
    let mut body = TcpStream::connect(address).unwrap();
    body.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 32\r\nContent-Type: application/json\r\n\r\n{\"").unwrap();
    wait_until(|| {
        server.metrics().active_connections == 3 && server.metrics().accepted_requests == 1
    });
    assert!(host.serving_bytes() >= baseline + 3 * super::transport::CONNECTION_HEADROOM);
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
    assert!(TcpStream::connect(address).is_err());
    for client in [idle, header, body] {
        assert_closed(client);
    }
}

type Held = (CancelToken, mpsc::SyncSender<ResponseEvent>);
struct PendingService {
    host: ServingHost,
    queue: PendingQueue<Held>,
    receipts: Mutex<Vec<CancelToken>>,
}
impl TextService for PendingService {
    fn count_input_tokens(
        &self,
        model: &str,
        input: &zerogravity_engine::router::TextInput,
    ) -> Result<usize, ServiceError> {
        assert_eq!(model, "fixture");
        // Test-only observation of exact semantic context, never model math.
        Ok(input.content_bytes().expect("bounded input"))
    }

    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge, ServiceError> {
        self.host
            .charge("shutdown-control", session, AllocationClass::Scratch, bytes)
    }
    fn model_ids(&self) -> Result<Vec<String>, ServiceError> {
        Ok(vec!["fixture".into()])
    }
    fn max_output(&self) -> Result<usize, ServiceError> {
        Ok(4)
    }
    fn admission(&self) -> Result<serde_json::Value, ServiceError> {
        Ok(serde_json::json!({}))
    }
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_semantic_admitted(request.into(), ingress)
    }
    fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        _ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        assert_ne!(request.model, "panic", "injected protocol-task panic");
        let (sender, receiver) = mpsc::try_sync_channel(&self.host, "control", "events", 2)?;
        let cancel = self.queue.cancel_token(None);
        let mut receipts = self.receipts.lock().unwrap();
        let id = receipts.len() as u64 + 1;
        self.queue
            .enqueue(self.queue.reserve()?, (cancel.clone(), sender))
            .map_err(|(error, _)| error)?;
        receipts.push(cancel.clone());
        Ok(RequestEvents::new(
            id,
            request.model,
            receiver,
            cancel,
            None,
            None,
        ))
    }
}
impl PendingService {
    fn open(host: ServingHost) -> Arc<Self> {
        Arc::new(Self {
            queue: PendingQueue::new(host.clone(), Some(8)).unwrap(),
            host,
            receipts: Mutex::new(Vec::new()),
        })
    }
    fn receipt(&self, index: usize) -> CancelToken {
        wait_until(|| self.receipts.lock().unwrap().len() > index);
        self.receipts.lock().unwrap()[index].clone()
    }
}

fn request(address: std::net::SocketAddr, streaming: bool) -> TcpStream {
    let mut socket = TcpStream::connect(address).unwrap();
    let body = format!(r#"{{"model":"fixture","prompt":"x","stream":{streaming}}}"#);
    write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}", body.len()).unwrap();
    socket
}

#[test]
fn shutdown_cancels_original_queued_and_active_streaming_and_nonstream_receipts() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let mut active = Vec::new();
    let mut clients = Vec::new();
    let mut receipts = Vec::new();
    for (i, streaming) in [false, true, false, true].into_iter().enumerate() {
        clients.push(request(server.local_addr(), streaming));
        receipts.push(service.receipt(i));
        if i < 2 {
            match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
                Pop::Item(job) => active.push(job),
                _ => panic!("active fixture missing"),
            }
        }
    }
    assert_eq!(service.queue.snapshot().pending, 2);
    server.shutdown().unwrap();
    assert!(receipts.iter().all(CancelToken::is_cancelled));
    assert_eq!(
        service
            .queue
            .sweep(|(cancel, _)| cancel.is_cancelled())
            .unwrap(),
        2
    );
    assert_eq!(service.queue.snapshot().pending, 0);
    for client in clients {
        assert_closed(client);
    }
    drop(active);
    service.queue.close();
    assert!(matches!(
        consumer.try_pop(|(cancel, _)| cancel.is_cancelled()),
        Pop::Closed
    ));
    drop((receipts, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn disconnect_cancels_only_its_original_request_before_first_token() {
    let _lock = test_support::lock();
    for (streaming, trailing) in [(false, false), (true, false), (false, true), (true, true)] {
        let host = test_support::host();
        let baseline = host.serving_bytes();
        let service = PendingService::open(host.clone());
        let consumer = service.queue.worker().unwrap();
        let server = start(service.clone());
        let mut first = request(server.local_addr(), streaming);
        let first_token = service.receipt(0);
        let second = request(server.local_addr(), !streaming);
        let second_token = service.receipt(1);
        if trailing {
            // A buffered next request cannot mask a peer close indefinitely.
            first
                .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                .unwrap();
        }
        drop(first);
        wait_until(|| first_token.is_cancelled());
        assert!(!second_token.is_cancelled());
        assert_eq!(
            service
                .queue
                .sweep(|(cancel, _)| cancel.is_cancelled())
                .unwrap(),
            1
        );
        server.shutdown().unwrap();
        assert!(second_token.is_cancelled());
        assert_closed(second);
        drop((first_token, second_token, consumer, service));
        assert_eq!(host.serving_bytes(), baseline);
    }
}

#[test]
fn responses_disconnect_and_shutdown_retire_original_receipts_and_projection() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let connect = |stream: bool| {
        let mut socket = TcpStream::connect(server.local_addr()).unwrap();
        let body = format!(r#"{{"model":"fixture","input":"x","stream":{stream},"store":true}}"#);
        write!(socket, "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}", body.len()).unwrap();
        socket
    };
    let first = connect(true);
    let first_token = service.receipt(0);
    let second = connect(false);
    let second_token = service.receipt(1);
    drop(first);
    wait_until(|| first_token.is_cancelled());
    assert!(!second_token.is_cancelled());
    assert_eq!(
        service
            .queue
            .sweep(|(cancel, _)| cancel.is_cancelled())
            .unwrap(),
        1
    );
    server.shutdown().unwrap();
    assert!(second_token.is_cancelled());
    assert_closed(second);
    drop((first_token, second_token, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn shutdown_joins_a_backpressured_stream_without_waiting_for_client() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(crate::ingress::IngressLimits {
            output_bytes: 32 * 1024 * 1024,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let client = request(server.local_addr(), true);
    let receipt = service.receipt(0);
    let active = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("active fixture missing"),
    };
    // An unread socket eventually blocks the original bounded event channel.
    // No unlimited producer or artificial sleep in the transport itself.
    let mut blocked = false;
    for _ in 0..64 {
        let mut event = ResponseEvent::Text("x".repeat(256 * 1024));
        let deadline = Instant::now() + Duration::from_millis(200);
        loop {
            match active.1.try_send(event) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(returned)) => {
                    if Instant::now() >= deadline {
                        blocked = true;
                        break;
                    }
                    event = returned;
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    panic!("request retired before backpressure")
                }
            }
        }
        if blocked {
            break;
        }
    }
    assert!(blocked && !receipt.is_cancelled());
    let stopped = Instant::now();
    server.shutdown().unwrap();
    assert!(stopped.elapsed() < Duration::from_secs(3));
    assert!(receipt.is_cancelled());
    assert_closed(client);
    drop((active, receipt, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn panicked_connection_is_reported_and_other_tasks_are_joined() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let addr = server.local_addr();
    let client = request(addr, false);
    let receipt = service.receipt(0);
    let mut panic_client = TcpStream::connect(addr).unwrap();
    let body = r#"{"model":"panic","prompt":"x"}"#;
    write!(panic_client, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}", body.len()).unwrap();
    wait_until(|| server.is_finished());
    assert!(server.join().is_err());
    assert!(receipt.is_cancelled());
    assert!(TcpStream::connect(addr).is_err());
    assert_closed(client);
    assert_closed(panic_client);
    drop((receipt, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

// Every exchange uses a fresh connection; the create socket owns no
// background execution lifetime.
fn background_exchange(
    address: std::net::SocketAddr,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
) -> (u16, serde_json::Value) {
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let data = body.map(serde_json::Value::to_string).unwrap_or_default();
    write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}", data.len()).unwrap();
    let mut raw = String::new();
    socket.read_to_string(&mut raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").expect("HTTP response");
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap())
}

#[test]
fn background_poll_cancel_completion_and_shutdown_share_original_receipts() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let address = server.local_addr();

    let submit = |store: bool| {
        let (status, response) = background_exchange(
            address,
            "POST",
            "/v1/responses",
            Some(&json!({
                "model":"fixture","input":"hello","background":true,"store":store,
                "background_options":{"completion_window":"soonest"}
            })),
        );
        assert_eq!(status, 200, "{response}");
        assert_eq!(response["background"], true);
        assert_eq!(response["store"], store);
        assert_eq!(response["status"], "in_progress");
        assert!(response["usage"].is_null());
        let path = format!("/v1/responses/{}", response["id"].as_str().unwrap());
        assert_eq!(
            background_exchange(address, "GET", &path, None),
            (200, response)
        );
        path
    };

    let first = submit(true);
    let first_token = service.receipt(0);
    assert!(!first_token.is_cancelled(), "create connection closed");
    let first_job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("original active receipt missing"),
    };
    let second = submit(false);
    let second_token = service.receipt(1);
    let second_job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("original active receipt missing"),
    };
    assert_eq!(background_exchange(address, "DELETE", &first, None).0, 409);
    let (status, items) =
        background_exchange(address, "GET", &format!("{first}/input_items"), None);
    assert_eq!(status, 200);
    assert_eq!(items["data"][0]["content"][0]["text"], "hello");
    let cancel_path = format!("{first}/cancel");
    let (status, cancelled) = background_exchange(address, "POST", &cancel_path, None);
    assert_eq!(status, 200);
    assert_eq!(cancelled["status"], "cancelled");
    assert!(cancelled["usage"].is_null());
    assert!(first_token.is_cancelled());
    assert!(!second_token.is_cancelled());
    assert_eq!(
        background_exchange(address, "POST", &cancel_path, None),
        (200, cancelled.clone())
    );
    // A late worker terminal cannot overwrite an accepted cancellation.
    let _ = first_job.1.try_send(ResponseEvent::Done {
        output: "late".into(),
        prompt_tokens: 3,
        cached_prompt_tokens: 0,
        completion_tokens: 1,
        finish_reason: "stop",
    });
    second_job
        .1
        .try_send(ResponseEvent::Done {
            output: "done".into(),
            prompt_tokens: 3,
            cached_prompt_tokens: 1,
            completion_tokens: 2,
            finish_reason: "length",
        })
        .unwrap_or_else(|_| panic!("unrelated original worker disconnected"));
    let mut terminal = serde_json::Value::Null;
    wait_until(|| {
        let (status, value) = background_exchange(address, "GET", &second, None);
        assert_eq!(status, 200);
        terminal = value;
        terminal["status"] != "in_progress"
    });
    assert_eq!(terminal["status"], "incomplete");
    assert_eq!(
        terminal["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(terminal["output"][0]["content"][0]["text"], "done");
    assert_eq!(
        terminal["usage"]["input_tokens_details"]["cached_tokens"],
        1
    );
    assert_eq!(
        background_exchange(address, "POST", &format!("{second}/cancel"), None),
        (200, terminal.clone())
    );
    assert_eq!(
        background_exchange(address, "GET", &first, None),
        (200, cancelled)
    );
    // Temporary store=false has polling semantics, not continuation.
    let (status, error) = background_exchange(
        address,
        "POST",
        "/v1/responses",
        Some(&json!({
            "model":"fixture","input":"again","previous_response_id":terminal["id"],
        })),
    );
    assert_eq!(status, 400);
    assert_eq!(error["error"]["param"], "previous_response_id");

    assert_eq!(background_exchange(address, "DELETE", &first, None).0, 200);
    assert_eq!(background_exchange(address, "GET", &first, None).0, 404);
    let (status, error) =
        background_exchange(address, "GET", &format!("{second}?stream=true"), None);
    assert_eq!(status, 400);
    assert_eq!(error["error"]["param"], "stream");
    let (status, error) = background_exchange(
        address,
        "POST",
        "/v1/responses",
        Some(
            &json!({"model":"fixture","input":"hello","background":true,"stream":true,
            "background_options":{"completion_window":"2h"}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(error["error"]["param"], "background_options");

    let _third = submit(true);
    let third_token = service.receipt(2);
    assert!(!third_token.is_cancelled());
    assert_eq!(service.queue.snapshot().pending, 1);
    server.shutdown().unwrap();
    assert!(third_token.is_cancelled());
    assert_eq!(
        service
            .queue
            .sweep(|(cancel, _)| cancel.is_cancelled())
            .unwrap(),
        1
    );
    drop((
        first_job,
        second_job,
        first_token,
        second_token,
        third_token,
        consumer,
        service,
    ));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn background_worker_failure_and_output_limit_publish_safe_terminal_records() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(crate::ingress::IngressLimits {
            output_bytes: 4096,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    for (event, expected) in [
        (
            ResponseEvent::Error(ServiceError::new(400, "invalid_prompt", "private-path")),
            "invalid_prompt",
        ),
        (
            ResponseEvent::Done {
                output: "x".repeat(8192),
                prompt_tokens: 3,
                cached_prompt_tokens: 0,
                completion_tokens: 2,
                finish_reason: "length",
            },
            "server_error",
        ),
    ] {
        let (status, response) = background_exchange(
            address,
            "POST",
            "/v1/responses",
            Some(&json!({
                "model":"fixture","input":"hello","background":true,"store":true,
            })),
        );
        assert_eq!(status, 200, "{response}");
        let path = format!("/v1/responses/{}", response["id"].as_str().unwrap());
        let job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
            Pop::Item(job) => job,
            _ => panic!("original active receipt missing"),
        };
        job.1
            .try_send(event)
            .unwrap_or_else(|_| panic!("original worker disconnected"));
        wait_until(|| {
            let (status, result) = background_exchange(address, "GET", &path, None);
            assert_eq!(status, 200);
            if result["status"] == "in_progress" {
                return false;
            }
            assert_eq!(result["status"], "failed");
            assert_eq!(result["error"]["code"], expected);
            assert!(result["usage"].is_null());
            assert!(!result.to_string().contains("private-path"));
            true
        });
        assert_eq!(background_exchange(address, "DELETE", &path, None).0, 200);
        drop(job);
    }
    server.shutdown().unwrap();
    drop((consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn conversation_background_cancel_and_terminal_append_are_atomic() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let address = server.local_addr();
    let call = |method: &str, path: &str, body: Option<&serde_json::Value>| {
        background_exchange(address, method, path, body)
    };
    let (status, conv) = call(
        "POST",
        "/v1/conversations",
        Some(&json!({
            "items":[{"role":"user","content":"original"}]
        })),
    );
    assert_eq!(status, 200, "{conv}");
    let id = conv["id"].as_str().unwrap();
    let path = format!("/v1/conversations/{id}");
    let items = format!("{path}/items");
    let initial = call("GET", &items, None);
    let request = json!({"model":"fixture","conversation":id,"input":"pending","background":true,"store":false});
    let (status, response) = call("POST", "/v1/responses", Some(&request));
    assert_eq!(status, 200, "{response}");
    let first = format!("/v1/responses/{}", response["id"].as_str().unwrap());
    let token = service.receipt(0);
    let job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("original active receipt missing"),
    };
    assert_eq!(call("GET", &items, None), initial);
    let count_request =
        json!({"model":"fixture","conversation":id,"input":"next","instructions":"brief"});
    let count = || call("POST", "/v1/responses/input_tokens", Some(&count_request));
    // Reads observe the last committed history, not the pending new input.
    assert_eq!(
        count(),
        (
            200,
            json!({"object":"response.input_tokens","input_tokens":17})
        )
    );
    assert_eq!(service.receipts.lock().unwrap().len(), 1);
    assert_eq!(call("GET", &items, None), initial);
    assert_eq!(call("POST", "/v1/responses", Some(&request)).0, 409);
    assert_eq!(
        call(
            "POST",
            &items,
            Some(&json!({"items":[{"role":"user","content":"race"}]}))
        )
        .0,
        409
    );
    assert_eq!(
        call("POST", &path, Some(&json!({"metadata":{"race":"true"}}))).0,
        409
    );
    assert_eq!(call("DELETE", &path, None).0, 409);
    assert_eq!(
        call(
            "DELETE",
            &format!("{items}/{}", initial.1["first_id"].as_str().unwrap()),
            None
        )
        .0,
        409
    );
    let (status, cancelled) = call("POST", &format!("{first}/cancel"), None);
    assert_eq!(status, 200);
    assert_eq!(cancelled["status"], "cancelled");
    assert!(token.is_cancelled());
    let _ = job.1.try_send(ResponseEvent::Done {
        output: "late".into(),
        prompt_tokens: 3,
        cached_prompt_tokens: 0,
        completion_tokens: 1,
        finish_reason: "stop",
    });
    assert_eq!(call("GET", &items, None), initial);
    // Lease release is tied to the joined consumer, not the cancellation flag.
    let mut second = serde_json::Value::Null;
    wait_until(|| {
        let (status, response) = call("POST", "/v1/responses", Some(&request));
        if status == 409 {
            return false;
        }
        assert_eq!(status, 200, "{response}");
        second = response;
        true
    });
    let second_path = format!("/v1/responses/{}", second["id"].as_str().unwrap());
    let second_job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("second original receipt missing"),
    };
    assert_eq!(call("GET", &items, None), initial);
    second_job
        .1
        .try_send(ResponseEvent::Done {
            output: "complete".into(),
            prompt_tokens: 3,
            cached_prompt_tokens: 0,
            completion_tokens: 1,
            finish_reason: "length",
        })
        .unwrap_or_else(|_| panic!("worker disconnected"));
    let mut terminal = serde_json::Value::Null;
    wait_until(|| {
        let (status, value) = call("GET", &second_path, None);
        assert_eq!(status, 200);
        terminal = value;
        terminal["status"] != "in_progress"
    });
    assert_eq!(terminal["status"], "incomplete");
    let (status, page) = call("GET", &format!("{items}?order=asc"), None);
    assert_eq!(status, 200);
    let data = page["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    assert_eq!(data[1]["content"][0]["text"], "pending");
    assert_eq!(data[2]["content"][0]["text"], "complete");
    assert_eq!(data[2]["id"], terminal["output"][0]["id"]);
    assert_eq!(data[2]["status"], "incomplete");
    assert_eq!(
        count(),
        (
            200,
            json!({"object":"response.input_tokens","input_tokens":32})
        )
    );
    assert_eq!(service.receipts.lock().unwrap().len(), 2);

    assert_eq!(
        call("POST", &format!("{second_path}/cancel"), None),
        (200, terminal)
    );
    assert_eq!(
        call("GET", &format!("{items}?order=asc"), None),
        (200, page)
    );
    assert_eq!(call("GET", &first, None), (200, cancelled));
    // Active shutdown must retire the lease and original receipt.
    assert_eq!(call("POST", "/v1/responses", Some(&request)).0, 200);
    let third = service.receipt(2);
    server.shutdown().unwrap();
    assert!(third.is_cancelled());
    assert_eq!(
        service
            .queue
            .sweep(|(cancel, _)| cancel.is_cancelled())
            .unwrap(),
        1
    );
    drop((job, second_job, token, third, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn foreground_conversation_disconnect_releases_lease_without_append() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let address = server.local_addr();
    let (status, conv) =
        background_exchange(address, "POST", "/v1/conversations", Some(&json!({})));
    assert_eq!(status, 200);
    let id = conv["id"].as_str().unwrap();
    let path = format!("/v1/conversations/{id}/items");
    for stream in [true, false] {
        let mut socket = TcpStream::connect(address).unwrap();
        let body =
            json!({"model":"fixture","conversation":id,"input":"disconnected","stream":stream})
                .to_string();
        write!(socket, "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}", body.len()).unwrap();
        let receipt = service.receipt(usize::from(!stream));
        assert!(!receipt.is_cancelled());
        let (status, page) = background_exchange(address, "GET", &path, None);
        assert_eq!(status, 200);
        assert_eq!(page["data"], json!([]));
        drop(socket);
        wait_until(|| receipt.is_cancelled());
        assert_eq!(
            service
                .queue
                .sweep(|(cancel, _)| cancel.is_cancelled())
                .unwrap(),
            1
        );
        // An edit proves the exact mutation lease was released. Remove the
        // edit to keep the same empty context for the next disconnect case.
        let mut added = serde_json::Value::Null;
        wait_until(|| {
            let (status, value) = background_exchange(
                address,
                "POST",
                &path,
                Some(&json!({"items":[{"role":"user","content":"edit"}]})),
            );
            if status == 409 {
                return false;
            }
            assert_eq!(status, 200, "{value}");
            added = value;
            true
        });
        assert_eq!(added["data"].as_array().unwrap().len(), 1);
        let item = added["first_id"].as_str().unwrap();
        assert_eq!(
            background_exchange(address, "DELETE", &format!("{path}/{item}"), None).0,
            200
        );
        assert_eq!(
            background_exchange(address, "GET", &path, None).1["data"],
            json!([])
        );
    }
    server.shutdown().unwrap();
    drop((consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

// Read SSE incrementally over actual chunked HTTP; a TCP/HTTP chunk need
// not coincide with an event. Preserve original event bytes for replay.
struct Sse {
    reader: BufReader<TcpStream>,
    pending: Vec<u8>,
    closed: bool,
}
impl Sse {
    fn open(
        address: std::net::SocketAddr,
        method: &str,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> Self {
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let data = body.map(serde_json::Value::to_string).unwrap_or_default();
        write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}", data.len()).unwrap();
        let mut reader = BufReader::new(socket);
        let mut head = String::new();
        loop {
            let mut line = String::new();
            assert_ne!(reader.read_line(&mut line).unwrap(), 0);
            head.push_str(&line);
            if line == "\r\n" {
                break;
            }
        }
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        assert!(
            head.to_ascii_lowercase()
                .contains("transfer-encoding: chunked"),
            "{head}"
        );
        assert!(
            head.to_ascii_lowercase()
                .contains("content-type: text/event-stream")
        );
        Self {
            reader,
            pending: Vec::new(),
            closed: false,
        }
    }
    fn next(&mut self) -> Option<String> {
        loop {
            if let Some(end) = self.pending.windows(2).position(|w| w == b"\n\n") {
                let frame = String::from_utf8(self.pending.drain(..end + 2).collect()).unwrap();
                if frame.starts_with(':') {
                    continue;
                }
                return Some(frame);
            }
            if self.closed {
                assert!(self.pending.is_empty(), "truncated SSE event");
                return None;
            }
            let mut line = String::new();
            assert_ne!(
                self.reader.read_line(&mut line).unwrap(),
                0,
                "truncated chunked response"
            );
            let size = usize::from_str_radix(line.trim(), 16).unwrap();
            let start = self.pending.len();
            assert!(start + size <= 1 << 20);
            self.pending.resize(start + size, 0);
            self.reader.read_exact(&mut self.pending[start..]).unwrap();
            let mut end = [0; 2];
            self.reader.read_exact(&mut end).unwrap();
            assert_eq!(&end, b"\r\n");
            self.closed = size == 0;
        }
    }
    fn rest(&mut self) -> Vec<String> {
        std::iter::from_fn(|| self.next()).collect()
    }
}
fn event(frame: &str) -> serde_json::Value {
    let (kind, data) = frame.trim_end().split_once("\ndata: ").unwrap();
    let value: serde_json::Value = serde_json::from_str(data).unwrap();
    assert_eq!(kind.strip_prefix("event: ").unwrap(), value["type"]);
    value
}

#[test]
fn background_stream_resumes_exact_events_and_appends_conversation_once() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = start(service.clone());
    let address = server.local_addr();
    let (status, conversation) = background_exchange(
        address,
        "POST",
        "/v1/conversations",
        Some(&json!({"items":[{"role":"user","content":"original"}]})),
    );
    assert_eq!(status, 200);
    let cid = conversation["id"].as_str().unwrap();
    let items = format!("/v1/conversations/{cid}/items?order=asc");
    let initial_items = background_exchange(address, "GET", &items, None);
    let request = json!({"model":"fixture","input":"new","conversation":cid,
        "background":true,"stream":true,"store":false});
    let mut create = Sse::open(address, "POST", "/v1/responses", Some(&request));
    let mut prefix = (0..4).map(|_| create.next().unwrap()).collect::<Vec<_>>();
    let opening = event(&prefix[0]);
    assert_eq!(opening["type"], "response.created");
    assert_eq!(opening["response"]["background"], true);
    let id = opening["response"]["id"].as_str().unwrap();
    let path = format!("/v1/responses/{id}");
    let token = service.receipt(0);
    let job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("original active receipt missing"),
    };
    job.1
        .try_send(ResponseEvent::Text("hé".into()))
        .unwrap_or_else(|_| panic!("worker disconnected"));
    prefix.push(create.next().unwrap());
    assert_eq!(event(&prefix[4])["delta"], "hé");
    drop(create);
    wait_until(|| server.metrics().active_connections == 0);
    assert!(
        !token.is_cancelled(),
        "creation socket owns no background receipt"
    );
    assert_eq!(
        background_exchange(address, "GET", &path, None).1,
        opening["response"]
    );
    assert_eq!(
        background_exchange(address, "GET", &items, None),
        initial_items
    );

    for query in [
        "stream=true&starting_after=5",
        "stream=true&starting_after=-1",
        "stream=false&starting_after=0",
        "stream=true&starting_after=0&starting_after=1",
        "stream=true&include_obfuscation=true",
    ] {
        assert_eq!(
            background_exchange(address, "GET", &format!("{path}?{query}"), None).0,
            400,
            "{query}"
        );
    }
    let mut resumed = Sse::open(
        address,
        "GET",
        &format!("{path}?starting_after=4&stream=true&include_obfuscation=false"),
        None,
    );
    let mut whole = Sse::open(address, "GET", &format!("{path}?stream=true"), None);
    for original in &prefix {
        assert_eq!(&whole.next().unwrap(), original);
    }
    job.1
        .try_send(ResponseEvent::Text("llo".into()))
        .unwrap_or_else(|_| panic!("worker disconnected"));
    let delta = resumed.next().unwrap();
    assert_eq!(event(&delta)["sequence_number"], 5);
    assert_eq!(event(&delta)["delta"], "llo");
    assert_eq!(whole.next().unwrap(), delta);
    job.1
        .try_send(ResponseEvent::Done {
            output: "héllo".into(),
            prompt_tokens: 3,
            cached_prompt_tokens: 1,
            completion_tokens: 2,
            finish_reason: "stop",
        })
        .unwrap_or_else(|_| panic!("worker disconnected"));
    let tail = resumed.rest();
    assert_eq!(whole.rest(), tail);
    assert_eq!(tail.len(), 4);
    let terminal = event(tail.last().unwrap());
    assert_eq!(terminal["type"], "response.completed");
    assert_eq!(
        terminal["response"]["output"][0]["content"][0]["text"],
        "héllo"
    );
    assert_eq!(
        terminal["response"]["usage"]["input_tokens_details"]["cached_tokens"],
        1
    );
    let all = prefix
        .into_iter()
        .chain([delta])
        .chain(tail)
        .collect::<Vec<_>>();
    for (i, frame) in all.iter().enumerate() {
        assert_eq!(event(frame)["sequence_number"], i);
    }
    assert_eq!(
        background_exchange(address, "GET", &path, None),
        (200, terminal["response"].clone())
    );
    let page = background_exchange(address, "GET", &items, None).1;
    assert_eq!(page["data"].as_array().unwrap().len(), 3);
    assert_eq!(page["data"][1]["content"][0]["text"], "new");
    assert_eq!(
        page["data"][2]["id"],
        terminal["response"]["output"][0]["id"]
    );
    assert_eq!(page["data"][2]["content"][0]["text"], "héllo");
    assert_eq!(
        Sse::open(address, "GET", &format!("{path}?stream=true"), None).rest(),
        all
    );
    assert_eq!(
        Sse::open(
            address,
            "GET",
            &format!("{path}?stream=true&starting_after=7"),
            None
        )
        .rest(),
        all[8..]
    );
    assert!(
        Sse::open(
            address,
            "GET",
            &format!("{path}?stream=true&starting_after=9"),
            None
        )
        .rest()
        .is_empty()
    );
    assert_eq!(background_exchange(address, "GET", &items, None).1, page);
    assert_eq!(service.receipts.lock().unwrap().len(), 1);
    assert_eq!(
        background_exchange(
            address,
            "POST",
            "/v1/responses",
            Some(&json!({
        "model":"fixture","input":"again","previous_response_id":id}))
        )
        .0,
        400
    );
    assert_eq!(background_exchange(address, "DELETE", &path, None).0, 200);
    assert_eq!(
        background_exchange(address, "GET", &format!("{path}?stream=true"), None).0,
        404
    );
    server.shutdown().unwrap();
    drop((resumed, whole, job, token, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn background_stream_cancel_overflow_and_shutdown_preserve_terminal_authority() {
    use serde_json::json;
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = PendingService::open(host.clone());
    let consumer = service.queue.worker().unwrap();
    let server = ServerBuilder::from_service(service.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(crate::ingress::IngressLimits {
            output_bytes: 8192,
            ..Default::default()
        })
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    let (_, conversation) =
        background_exchange(address, "POST", "/v1/conversations", Some(&json!({})));
    let cid = conversation["id"].as_str().unwrap();
    let items = format!("/v1/conversations/{cid}/items");
    let initial = background_exchange(address, "GET", &items, None);
    let request = json!({"model":"fixture","input":"x","conversation":cid,
        "background":true,"stream":true,"store":true});
    let submit = || {
        let mut stream = Sse::open(address, "POST", "/v1/responses", Some(&request));
        let prefix = (0..4).map(|_| stream.next().unwrap()).collect::<Vec<_>>();
        let first = event(&prefix[0]);
        let id = first["response"]["id"].as_str().unwrap();
        (stream, prefix, format!("/v1/responses/{id}"))
    };

    let (mut stream, mut frames, path) = submit();
    let token = service.receipt(0);
    let job = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("original active receipt missing"),
    };
    job.1
        .try_send(ResponseEvent::Text("partial".into()))
        .unwrap_or_else(|_| panic!("worker disconnected"));
    frames.push(stream.next().unwrap());
    let mut resumed = Sse::open(
        address,
        "GET",
        &format!("{path}?stream=true&starting_after=4"),
        None,
    );
    let (status, cancelled) = background_exchange(address, "POST", &format!("{path}/cancel"), None);
    assert_eq!(status, 200);
    assert_eq!(cancelled["status"], "cancelled");
    assert!(token.is_cancelled());
    let _ = job.1.try_send(ResponseEvent::Done {
        output: "late".into(),
        prompt_tokens: 3,
        cached_prompt_tokens: 0,
        completion_tokens: 1,
        finish_reason: "stop",
    });
    assert!(stream.rest().is_empty());
    assert!(resumed.rest().is_empty());
    assert_eq!(
        Sse::open(address, "GET", &format!("{path}?stream=true"), None).rest(),
        frames
    );
    assert_eq!(
        background_exchange(address, "POST", &format!("{path}/cancel"), None),
        (200, cancelled)
    );
    assert_eq!(background_exchange(address, "GET", &items, None), initial);
    assert_eq!(background_exchange(address, "DELETE", &path, None).0, 200);

    // This admitted delta fits the log, but the entire success terminal
    // group cannot fit. Never expose a partially successful terminal.
    let (mut stream, mut frames, path) = submit();
    let token2 = service.receipt(1);
    let job2 = match consumer.try_pop(|(cancel, _)| cancel.is_cancelled()) {
        Pop::Item(job) => job,
        _ => panic!("original active receipt missing"),
    };
    let text = "x".repeat(3500);
    job2.1
        .try_send(ResponseEvent::Text(text.clone()))
        .unwrap_or_else(|_| panic!("worker disconnected"));
    frames.push(stream.next().unwrap());
    assert_eq!(event(&frames[4])["delta"], text);
    job2.1
        .try_send(ResponseEvent::Done {
            output: text,
            prompt_tokens: 3,
            cached_prompt_tokens: 0,
            completion_tokens: 1,
            finish_reason: "stop",
        })
        .unwrap_or_else(|_| panic!("worker disconnected"));
    let tail = stream.rest();
    assert_eq!(tail.len(), 1);
    let failed = event(&tail[0]);
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["sequence_number"], 5);
    assert_eq!(failed["response"]["status"], "failed");
    assert!(failed["response"]["usage"].is_null());
    frames.extend(tail);
    assert!(token2.is_cancelled());
    assert_eq!(
        background_exchange(address, "GET", &path, None),
        (200, failed["response"].clone())
    );
    assert_eq!(background_exchange(address, "GET", &items, None), initial);
    assert_eq!(
        Sse::open(address, "GET", &format!("{path}?stream=true"), None).rest(),
        frames
    );
    assert_eq!(background_exchange(address, "DELETE", &path, None).0, 200);
    // Active subscriber and job are both joined at shutdown.
    let (stream, _, _) = submit();
    let token3 = service.receipt(2);
    server.shutdown().unwrap();
    assert!(token3.is_cancelled());
    assert_eq!(
        service
            .queue
            .sweep(|(cancel, _)| cancel.is_cancelled())
            .unwrap(),
        1
    );
    assert_closed(stream.reader.into_inner());
    drop((job, job2, token, token2, token3, consumer, service));
    assert_eq!(host.serving_bytes(), baseline);
}
