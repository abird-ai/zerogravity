//! Real sockets over controlled original receipts. Learned parity is separate.
use super::*;
use crate::service::{Request, RequestEvents, SemanticRequest, ServingCharge};
use serde_json::{Value, json};
use std::sync::Mutex;
use zerogravity_core::resource_arbiter::AllocationClass;
use zerogravity_engine::{
    router::{
        ChatMessage, ChatRole, ResponseEvent, TextInput,
        event_channel::mpsc,
        pending::{CancelToken, ServingHost},
    },
    state_cache::PrefixSharingScope,
};

struct Probe {
    host: ServingHost,
    registration: AtomicUsize,
    requests: Mutex<Vec<Vec<ChatMessage>>>,
    held: Mutex<Vec<(mpsc::SyncSender<ResponseEvent>, CancelToken)>>,
}
impl TextService for Probe {
    fn request_admitted(
        &self,
        _: Request,
        _: Vec<ServingCharge>,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        Err(ServiceError::new(
            400,
            "unsupported_task",
            "fixture only accepts realtime",
        ))
    }
    fn realtime_text_registration(&self, alias: &str) -> std::result::Result<u64, ServiceError> {
        if alias != "fixture" {
            return Err(ServiceError::new(404, "not_found", "fixture unavailable"));
        }
        Ok(self.registration.load(Ordering::Acquire) as u64)
    }
    fn request_realtime_text_admitted(
        &self,
        request: SemanticRequest,
        registration: u64,
        ingress: Vec<ServingCharge>,
        sharing: Option<PrefixSharingScope>,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        if self.realtime_text_registration(&request.model)? != registration {
            return Err(ServiceError::new(
                409,
                "model_changed",
                "fixture registration changed",
            ));
        }
        assert!(request.stream && request.token_metrics.is_none() && sharing.is_none());
        assert_eq!(ingress.len(), 1);
        let TextInput::Conversation(turns) = request.input else {
            panic!("conversation intent")
        };
        let last = turns.last().unwrap().text.clone();
        self.requests.lock().unwrap().push(turns);
        let (tx, rx) = mpsc::try_sync_channel(&self.host, "fixture", "realtime", 4)?;
        let cancel = test_support::cancel(&self.host);
        let text = if last == "hold" { "partial" } else { "hello" };
        tx.try_send(ResponseEvent::Text(text.into()))
            .unwrap_or_else(|_| panic!("delta"));
        if last == "hold" {
            self.held.lock().unwrap().push((tx, cancel.clone()));
        } else if last == "model-error" {
            tx.try_send(ResponseEvent::Error(ServiceError::new(
                400,
                "invalid_input",
                "fixture refusal",
            )))
            .unwrap_or_else(|_| panic!("error terminal"));
        } else {
            tx.try_send(ResponseEvent::Done {
                output: if last == "bad" {
                    "different".into()
                } else {
                    text.into()
                },
                prompt_tokens: 5,
                cached_prompt_tokens: 2,
                completion_tokens: 2,
                finish_reason: if last == "length" { "length" } else { "stop" },
            })
            .unwrap_or_else(|_| panic!("terminal"));
        }
        Ok(RequestEvents::new(
            1,
            request.model,
            rx,
            cancel,
            ingress.into_iter().next(),
            None,
        ))
    }
    fn charge_ingress(
        &self,
        session: &str,
        bytes: u64,
    ) -> std::result::Result<ServingCharge, ServiceError> {
        self.host
            .charge("fixture", session, AllocationClass::Scratch, bytes)
    }
    fn model_ids(&self) -> std::result::Result<Vec<String>, ServiceError> {
        Ok(vec!["fixture".into()])
    }
    fn max_output(&self) -> std::result::Result<usize, ServiceError> {
        Ok(8)
    }
    fn admission(&self) -> std::result::Result<Value, ServiceError> {
        Ok(json!({"kind":"realtime_control_fixture"}))
    }
}
fn send(client: &mut WebSocket<TcpStream>, value: Value) {
    client
        .send(Message::Text(value.to_string().into()))
        .unwrap();
}
fn read(client: &mut WebSocket<TcpStream>, kind: &str) -> Value {
    for _ in 0..20 {
        let event: Value =
            serde_json::from_str(client.read().unwrap().into_text().unwrap().as_str()).unwrap();
        assert!(
            event["event_id"]
                .as_str()
                .is_some_and(|id| id.starts_with("event_"))
        );
        if event["type"] == kind {
            return event;
        }
        assert_ne!(event["type"], "error", "{event}");
    }
    panic!("missing {kind}")
}
fn user(client: &mut WebSocket<TcpStream>, text: &str) {
    send(
        client,
        json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":text}]}}),
    );
    let item = read(client, "conversation.item.done");
    assert_eq!(item["item"]["content"][0]["text"], text);
}
fn respond(client: &mut WebSocket<TcpStream>) -> Value {
    send(client, json!({"type":"response.create"}));
    read(client, "response.created")
}
#[test]
fn bounded_realtime_turns_use_original_receipts_and_retire() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let probe = Arc::new(Probe {
        host: host.clone(),
        registration: AtomicUsize::new(7),
        requests: Mutex::new(Vec::new()),
        held: Mutex::new(Vec::new()),
    });
    let observations = Arc::new(Mutex::new(
        Vec::<zerogravity_engine::router::FailureDetails>::new(),
    ));
    let observed = observations.clone();
    let server = ServerBuilder::from_service(probe.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .enable_error_details(true)
        .observe_errors(move |details| observed.lock().unwrap().push(details))
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    assert!(handshake(address, PATH, false, KEY, 0).starts_with("HTTP/1.1 401"));
    assert!(
        handshake(address, "/v1/realtime?model=missing", true, KEY, 0).starts_with("HTTP/1.1 404")
    );
    let mut client = upgrade(address);
    let created = read(&mut client, "session.created");
    assert_eq!(created["session"]["output_modalities"], json!(["text"]));
    send(
        &mut client,
        json!({"type":"session.update","session":{"type":"realtime","output_modalities":["audio"]}}),
    );
    let refusal = read(&mut client, "error");
    assert_eq!(refusal["error"]["code"], "unsupported_parameter");
    assert!(refusal["error"].get("details").is_none());
    client
        .send(Message::Text(
            r#"{"type":"response.create","type":"response.cancel"}"#.into(),
        ))
        .unwrap();
    assert_eq!(read(&mut client, "error")["error"]["code"], "invalid_json");
    send(
        &mut client,
        json!({"type":"input_audio_buffer.append","audio":"AAA="}),
    );
    assert_eq!(
        read(&mut client, "error")["error"]["code"],
        "unsupported_parameter"
    );
    send(
        &mut client,
        json!({"type":"session.update","session":{"type":"realtime","instructions":"brief","output_modalities":["text"],"max_output_tokens":4}}),
    );
    assert_eq!(
        read(&mut client, "session.updated")["session"]["max_output_tokens"],
        4
    );
    user(&mut client, "first");
    let response = respond(&mut client);
    let delta = read(&mut client, "response.output_text.delta");
    assert_eq!(delta["response_id"], response["response"]["id"]);
    assert_eq!(delta["delta"], "hello");
    let done = read(&mut client, "response.done");
    assert_eq!(done["response"]["status"], "completed");
    assert_eq!(done["response"]["usage"]["total_tokens"], 7);
    assert_eq!(
        done["response"]["usage"]["input_token_details"]["cached_tokens"],
        2
    );
    user(&mut client, "length");
    respond(&mut client);
    let done = read(&mut client, "response.done");
    assert_eq!(done["response"]["status"], "incomplete");
    assert_eq!(
        done["response"]["status_details"]["reason"],
        "max_output_tokens"
    );
    user(&mut client, "bad");
    respond(&mut client);
    let done = read(&mut client, "response.done");
    assert_eq!(done["response"]["status"], "failed");
    assert_eq!(
        done["response"]["status_details"]["error"]["code"],
        "invalid_terminal"
    );
    user(&mut client, "model-error");
    respond(&mut client);
    let done = read(&mut client, "response.done");
    assert_eq!(done["response"]["status"], "failed");
    assert_eq!(
        done["response"]["status_details"]["error"]["type"],
        "invalid_request_error"
    );
    assert_eq!(
        done["response"]["status_details"]["error"]["code"],
        "invalid_input"
    );
    user(&mut client, "hold");
    let response = respond(&mut client);
    read(&mut client, "response.output_text.delta");
    send(
        &mut client,
        json!({"type":"response.cancel","response_id":"wrong"}),
    );
    read(&mut client, "error");
    assert!(!probe.held.lock().unwrap()[0].1.is_cancelled());
    send(
        &mut client,
        json!({"type":"response.cancel","response_id":response["response"]["id"]}),
    );
    let done = read(&mut client, "response.done");
    assert_eq!(done["response"]["status"], "cancelled");
    assert!(done["response"]["usage"].is_null());
    wait(|| probe.held.lock().unwrap()[0].1.is_cancelled());
    {
        let requests = probe.requests.lock().unwrap();
        assert_eq!(
            requests[0],
            vec![
                ChatMessage {
                    role: ChatRole::System,
                    text: "brief".into()
                },
                ChatMessage {
                    role: ChatRole::User,
                    text: "first".into()
                }
            ]
        );
        let roles: Vec<_> = requests
            .last()
            .unwrap()
            .iter()
            .map(|m| (m.role, m.text.as_str()))
            .collect();
        assert_eq!(
            roles,
            [
                (ChatRole::System, "brief"),
                (ChatRole::User, "first"),
                (ChatRole::Assistant, "hello"),
                (ChatRole::User, "length"),
                (ChatRole::User, "bad"),
                (ChatRole::User, "model-error"),
                (ChatRole::User, "hold")
            ]
        );
    }
    probe.registration.store(8, Ordering::Release);
    user(&mut client, "stale");
    send(&mut client, json!({"type":"response.create"}));
    assert_eq!(read(&mut client, "error")["error"]["code"], "model_changed");
    assert_eq!(probe.requests.lock().unwrap().len(), 5);
    drop(client);
    let mut client = upgrade(address);
    read(&mut client, "session.created");
    user(&mut client, "hold");
    respond(&mut client);
    read(&mut client, "response.output_text.delta");
    drop(client);
    wait(|| probe.held.lock().unwrap()[1].1.is_cancelled());
    let mut client = upgrade(address);
    read(&mut client, "session.created");
    user(&mut client, "hold");
    respond(&mut client);
    read(&mut client, "response.output_text.delta");
    server.shutdown().unwrap();
    assert!(probe.held.lock().unwrap()[2].1.is_cancelled());
    assert!(client.read().is_err());
    let entries = observations.lock().unwrap();
    for (code, status) in [
        ("unsupported_parameter", 400),
        ("invalid_terminal", 500),
        ("invalid_input", 400),
    ] {
        assert!(
            entries
                .iter()
                .any(|e| e.code == code && e.status == Some(status)),
            "{code}"
        );
    }
    drop(entries);
    drop(probe);
    assert_eq!(host.serving_bytes(), baseline);
}
