//! The only installed realtime adapter here is a transport fixture. No
//! model, event protocol or public realtime capability is advertised.
use super::*;
use crate::{
    service::ServiceError,
    test_support::{self, HostBytes},
};
use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};
use tokio_tungstenite::tungstenite::{
    self, Message, WebSocket, client::IntoClientRequest, http::HeaderValue,
};

fn wait(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !ready() {
        assert!(Instant::now() < deadline, "upgrade owner did not retire");
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn connect(address: SocketAddr) -> TcpStream {
    let socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    socket
}
fn handshake(address: SocketAddr, path: &str, bearer: bool, key: &str, length: usize) -> String {
    let mut socket = connect(address);
    let auth = if bearer {
        "Authorization: Bearer secret\r\n"
    } else {
        ""
    };
    write!(socket, "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {key}\r\nContent-Length: {length}\r\n{auth}\r\n").unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    response
}
fn upgrade(address: SocketAddr) -> WebSocket<TcpStream> {
    let mut request = "ws://localhost/v1/realtime?model=fixture"
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("authorization", HeaderValue::from_static("Bearer secret"));
    let (client, response) = tungstenite::client(request, connect(address)).unwrap();
    assert_eq!(response.status().as_u16(), 101);
    client
}
struct Receipt(Arc<AtomicUsize>);
impl Drop for Receipt {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
fn fixture(
    host: &zerogravity_engine::router::pending::ServingHost,
    limits: IngressLimits,
) -> (ServerHandle, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let admitted = Arc::new(AtomicUsize::new(0));
    let live = Arc::new(AtomicUsize::new(0));
    let mut builder = ServerBuilder::from_service(test_support::idle(host))
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .limits(limits);
    let admissions = admitted.clone();
    let owners = live.clone();
    builder.realtime = Some(Arc::new(move |path| {
        if path != "/v1/realtime?model=fixture" {
            return Err(ServiceError::new(404, "model_not_found", "unknown fixture"));
        }
        admissions.fetch_add(1, Ordering::AcqRel);
        owners.fetch_add(1, Ordering::AcqRel);
        let receipt = Receipt(owners.clone());
        Ok(Box::new(move |mut socket| {
            Box::pin(async move {
                let _receipt = receipt;
                while let Some(message) = socket.receive().await? {
                    socket.send(message).await?;
                }
                Ok(())
            })
        }))
    }));
    (builder.build().unwrap().start().unwrap(), admitted, live)
}
const KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";
const PATH: &str = "/v1/realtime?model=fixture";

#[test]
fn ordinary_server_does_not_advertise_or_upgrade_realtime() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let server = ServerBuilder::from_service(test_support::idle(&host))
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .build()
        .unwrap()
        .start()
        .unwrap();
    assert!(handshake(server.local_addr(), PATH, true, KEY, 0).starts_with("HTTP/1.1 404"));
    assert!(handshake(server.local_addr(), "/v1/models", true, KEY, 0).starts_with("HTTP/1.1 400"));
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn actual_upgrade_checks_auth_handshake_and_adapter_before_switching() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let (server, admitted, live) = fixture(&host, IngressLimits::default());
    let address = server.local_addr();
    assert!(handshake(address, PATH, false, KEY, 0).starts_with("HTTP/1.1 401"));
    assert!(
        handshake(address, PATH, true, "bm90LWEtMTYtYnl0ZS1rZXk=", 0).starts_with("HTTP/1.1 400")
    );
    assert!(handshake(address, PATH, true, KEY, 1).starts_with("HTTP/1.1 400"));
    assert!(
        handshake(address, "/v1/realtime?model=missing", true, KEY, 0).starts_with("HTTP/1.1 404")
    );
    wait(|| server.metrics().active_connections == 0);
    assert_eq!(admitted.load(Ordering::Acquire), 0);
    assert_eq!(live.load(Ordering::Acquire), 0);
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn actual_upgrade_retains_connection_slot_and_retires_on_close_and_stop() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let (server, admitted, live) = fixture(
        &host,
        IngressLimits {
            connections: 1,
            body_bytes: 64,
            output_bytes: 4096,
        },
    );
    let address = server.local_addr();
    let idle = host.serving_bytes();
    let mut client = upgrade(address);
    assert_eq!(live.load(Ordering::Acquire), 1);
    assert_eq!(server.metrics().active_connections, 1);
    assert!(host.serving_bytes() > idle);
    client.send(Message::Text("hello".into())).unwrap();
    assert_eq!(client.read().unwrap(), Message::Text("hello".into()));
    client
        .send(Message::Binary(vec![0, 255, 4].into()))
        .unwrap();
    assert_eq!(
        client.read().unwrap(),
        Message::Binary(vec![0, 255, 4].into())
    );
    client.send(Message::Ping(vec![7].into())).unwrap();
    assert_eq!(client.read().unwrap(), Message::Pong(vec![7].into()));
    let mut excess = connect(address);
    let mut refusal = String::new();
    excess.read_to_string(&mut refusal).unwrap();
    assert!(refusal.starts_with("HTTP/1.1 503"));
    client.close(None).unwrap();
    assert!(matches!(client.read(), Ok(Message::Close(_))));
    wait(|| live.load(Ordering::Acquire) == 0 && server.metrics().active_connections == 0);
    // Joining completed connection tasks restores the same ledger.
    wait(|| host.serving_bytes() == idle);

    let client = upgrade(address);
    drop(client);
    wait(|| live.load(Ordering::Acquire) == 0 && server.metrics().active_connections == 0);
    wait(|| host.serving_bytes() == idle);

    let mut client = upgrade(address);
    assert_eq!(admitted.load(Ordering::Acquire), 3);
    server.shutdown().unwrap();
    assert_eq!(live.load(Ordering::Acquire), 0);
    assert!(client.read().is_err());
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn actual_upgrade_rejects_oversize_frames_and_keeps_original_wire_budget() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let (server, admitted, live) = fixture(
        &host,
        IngressLimits {
            connections: 2,
            body_bytes: 64,
            output_bytes: 512,
        },
    );
    let address = server.local_addr();
    let mut client = upgrade(address);
    client.send(Message::Binary(vec![0; 65].into())).unwrap();
    assert!(client.read().is_err());
    wait(|| live.load(Ordering::Acquire) == 0);

    // Each frame fits individually, but the reassembled message does not.
    use tungstenite::protocol::frame::{
        Frame,
        coding::{Data, OpCode},
    };
    let mut client = upgrade(address);
    client
        .send(Message::Frame(Frame::message(
            vec![0; 40],
            OpCode::Data(Data::Binary),
            false,
        )))
        .unwrap();
    client
        .send(Message::Frame(Frame::message(
            vec![0; 40],
            OpCode::Data(Data::Continue),
            true,
        )))
        .unwrap();
    assert!(client.read().is_err());
    wait(|| live.load(Ordering::Acquire) == 0);

    let mut client = upgrade(address);
    let message = Message::Binary(vec![0; 64].into());
    let mut echoes = 0;
    for _ in 0..12 {
        if client.send(message.clone()).is_err() {
            break;
        }
        match client.read() {
            Ok(reply) => {
                assert_eq!(reply, message);
                echoes += 1;
            }
            Err(_) => break,
        }
    }
    assert!(
        (1..=6).contains(&echoes),
        "upgrade reset wire budget: {echoes}"
    );
    wait(|| live.load(Ordering::Acquire) == 0);
    assert_eq!(admitted.load(Ordering::Acquire), 3);
    server.shutdown().unwrap();
    assert_eq!(host.serving_bytes(), baseline);
}
