//! Model-free HTTP controls; no numerical/device qualification.

use super::*;
use crate::serve::{engine, join_router};
use pending::{CancelToken, PendingQueue, Pop, ServingHost};
use std::sync::mpsc as control;

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

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

#[derive(Clone)]
struct Ingress(ServingHost);
impl IngressCharge for Ingress {
    fn charge_http_ingress(
        &self,
        session: &str,
        bytes: u64,
    ) -> Result<ServingCharge, ServiceError> {
        self.0.charge(
            "shutdown-control",
            session,
            zerogravity::resource_arbiter::AllocationClass::Scratch,
            bytes,
        )
    }
}
fn ingress() -> Ingress {
    Ingress(ServingHost::isolated(256).unwrap())
}

#[test]
fn idle_listener_stops_without_a_wakeup_connection() {
    let (router, host) = engine::Router::transport_test();
    let router = Arc::new(Router::Cpu(router));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&shutdown);
    let serving = Arc::clone(&router);
    let (ready, started) = control::channel();
    let (done, finished) = control::channel();
    let server = std::thread::spawn(move || {
        ready.send(()).unwrap();
        let result = serve(listener, serving, None, false, shutdown);
        done.send(result).unwrap();
    });
    started.recv_timeout(Duration::from_secs(3)).unwrap();
    std::thread::sleep(SHUTDOWN_POLL * 2);
    stop.store(true, Ordering::Release);
    finished
        .recv_timeout(Duration::from_secs(3))
        .unwrap()
        .unwrap();
    server.join().unwrap();
    join_router(router).unwrap();
    assert_eq!(host.serving_bytes(), 0);
}

#[test]
fn shutdown_interrupts_idle_partial_header_and_partial_body_then_joins_workers() {
    let (router, host) = engine::Router::transport_test();
    let router = Arc::new(Router::Cpu(router));
    let baseline = host.serving_bytes();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let serving = Arc::clone(&router);
    let stop = Arc::clone(&shutdown);
    let (done, finished) = control::channel();
    let server = std::thread::spawn(move || {
        done.send(serve(listener, serving, None, false, shutdown))
            .unwrap();
    });
    let idle = TcpStream::connect(address).unwrap();
    let mut header = TcpStream::connect(address).unwrap();
    header
        .write_all(b"POST /v1/completions HTTP/1.1\r\nHost:")
        .unwrap();
    let mut body = TcpStream::connect(address).unwrap();
    body.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 32\r\nContent-Type: application/json\r\n\r\n{\"").unwrap();
    wait_until(|| host.serving_bytes() >= baseline + 3 * HTTP_HEADER_ENVELOPE);
    stop.store(true, Ordering::Release);
    finished
        .recv_timeout(Duration::from_secs(3))
        .unwrap()
        .unwrap();
    server.join().unwrap();
    join_router(router).unwrap();
    assert_eq!(
        host.serving_bytes(),
        0,
        "HTTP/worker/table owners retired after joins"
    );
    assert!(
        TcpStream::connect(address).is_err(),
        "listener closed before return"
    );
    for mut client in [idle, header, body] {
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut byte = [0];
        match client.read(&mut byte) {
            Ok(0) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe
                ) => {}
            other => panic!("client remained open after joined shutdown: {other:?}"),
        }
    }
}

#[test]
fn shutdown_cancels_original_queued_and_active_streaming_and_nonstream_receipts() {
    let ingress = ingress();
    let queue = PendingQueue::new(ingress.0.clone(), Some(8)).unwrap();
    let consumer = queue.worker().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut connections = Connections::new(&ingress, Arc::clone(&shutdown)).unwrap();
    let mut active = Vec::new();
    let mut clients = Vec::new();
    let mut tokens = Vec::new();
    let (begun, started) = control::channel();
    for (id, streaming, running) in [
        (1, false, true),
        (2, true, true),
        (3, false, false),
        (4, true, false),
    ] {
        let cancel = queue.cancel_token(None);
        let (sender, receiver) =
            mpsc::try_sync_channel(&ingress.0, "control", "events", 2).unwrap();
        queue
            .enqueue(queue.reserve().unwrap(), (cancel.clone(), sender))
            .unwrap_or_else(|_| panic!("publish control"));
        if running {
            match consumer.try_pop(|(cancel, _): &(CancelToken, _)| cancel.is_cancelled()) {
                Pop::Item(job) => active.push(job),
                _ => panic!("active control missing"),
            }
        }
        let request =
            RequestEvents::new(id, "no-model".into(), receiver, cancel.clone(), None, None);
        tokens.push(cancel);
        let (client, mut stream) = pair();
        let interrupt = stream.try_clone().unwrap();
        let stop = Arc::clone(&shutdown);
        let begun = begun.clone();
        let charge = ingress.charge_http_ingress("connection", 128).unwrap();
        let handle = std::thread::spawn(move || {
            begun.send(()).unwrap();
            if streaming {
                assert!(stream_events(&mut stream, request, false, false, &stop).is_err());
            } else {
                assert_eq!(
                    completion_events(&mut stream, request, false, &stop)
                        .unwrap_err()
                        .code,
                    "shutting_down"
                );
            }
        });
        connections.entries.push(Connection {
            handle,
            interrupt,
            _charge: charge,
        });
        clients.push(client);
    }
    for _ in 0..4 {
        started.recv_timeout(Duration::from_secs(3)).unwrap();
    }
    assert_eq!(queue.snapshot().pending, 2);
    connections.finish().unwrap();
    assert!(tokens.iter().all(CancelToken::is_cancelled));
    assert_eq!(queue.sweep(|(cancel, _)| cancel.is_cancelled()).unwrap(), 2);
    assert_eq!(queue.snapshot().pending, 0);
    drop(active);
    queue.close();
    assert!(matches!(
        consumer.try_pop(|(cancel, _)| cancel.is_cancelled()),
        Pop::Closed
    ));
    drop((connections, tokens, clients, consumer, queue));
    assert_eq!(ingress.0.serving_bytes(), 0);
}

#[test]
fn disconnect_cancels_generation_before_the_first_token() {
    for streaming in [false, true] {
        let host = ServingHost::isolated(256).unwrap();
        let (sender, receiver) = mpsc::try_sync_channel(&host, "control", "events", 2).unwrap();
        let cancel = CancelToken::standalone();
        let token = cancel.clone();
        let request = RequestEvents::new(1, "no-model".into(), receiver, cancel, None, None);
        let (client, mut stream) = pair();
        let handle = std::thread::spawn(move || {
            let stop = AtomicBool::new(false);
            if streaming {
                assert!(stream_events(&mut stream, request, false, false, &stop).is_err());
            } else {
                assert_eq!(
                    completion_events(&mut stream, request, false, &stop)
                        .unwrap_err()
                        .status,
                    499
                );
            }
        });
        drop(client);
        wait_until(|| token.is_cancelled());
        handle.join().unwrap();
        drop(sender);
        assert_eq!(host.serving_bytes(), 0);
    }
}

fn blocked_reader(
    connections: &mut Connections,
    ingress: &Ingress,
) -> (TcpStream, control::Receiver<()>) {
    let (client, mut stream) = pair();
    let interrupt = stream.try_clone().unwrap();
    let source = ingress.clone();
    let charge = ingress.charge_http_ingress("connection", 128).unwrap();
    let (done, finished) = control::channel();
    let handle = std::thread::spawn(move || {
        assert!(read_request(&mut stream, &source).is_err());
        done.send(()).unwrap();
    });
    connections.entries.push(Connection {
        handle,
        interrupt,
        _charge: charge,
    });
    (client, finished)
}

#[test]
fn listener_failure_interrupts_and_joins_existing_partial_clients() {
    let ingress = ingress();
    let mut connections = Connections::new(&ingress, Arc::new(AtomicBool::new(false))).unwrap();
    let (_client, finished) = blocked_reader(&mut connections, &ingress);
    let error = accept_connections(
        &mut connections,
        || Err(std::io::Error::other("injected listener failure")),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "injected listener failure");
    connections.finish().unwrap();
    finished.recv_timeout(Duration::from_secs(3)).unwrap();
    drop(connections);
    assert_eq!(ingress.0.serving_bytes(), 0);
}

#[test]
fn handler_spawn_failure_interrupts_and_joins_existing_clients() {
    let ingress = ingress();
    let mut connections = Connections::new(&ingress, Arc::new(AtomicBool::new(false))).unwrap();
    let (_blocked_client, finished) = blocked_reader(&mut connections, &ingress);
    let (_new_client, stream) = pair();
    let mut accepted = Some(stream);
    let error = accept_connections(
        &mut connections,
        || Ok(accepted.take().expect("one connection before spawn error")),
        |_, _| Err(std::io::Error::other("injected handler spawn failure")),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "injected handler spawn failure");
    connections.finish().unwrap();
    finished.recv_timeout(Duration::from_secs(3)).unwrap();
    drop(connections);
    assert_eq!(ingress.0.serving_bytes(), 0);
}

#[test]
fn shutdown_interrupts_a_backpressured_writer_before_join() {
    let ingress = ingress();
    let mut connections = Connections::new(&ingress, Arc::new(AtomicBool::new(false))).unwrap();
    let (_client, mut stream) = pair();
    stream.set_nonblocking(true).unwrap();
    let data = [0u8; 16 * 1024];
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        assert!(
            Instant::now() < deadline,
            "socket did not reach backpressure"
        );
        match stream.write(&data) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            other => panic!("cannot establish backpressured control: {other:?}"),
        }
    }
    stream.set_nonblocking(false).unwrap();
    let interrupt = stream.try_clone().unwrap();
    let (begun, started) = control::channel();
    let handle = std::thread::spawn(move || {
        begun.send(()).unwrap();
        loop {
            if stream.write_all(&data).is_err() {
                break;
            }
        }
    });
    connections.entries.push(Connection {
        handle,
        interrupt,
        _charge: ingress.charge_http_ingress("connection", 128).unwrap(),
    });
    started.recv_timeout(Duration::from_secs(3)).unwrap();
    connections.finish().unwrap();
    drop(connections);
    assert_eq!(ingress.0.serving_bytes(), 0);
}

#[test]
fn unwind_interrupts_and_joins_every_existing_handler() {
    let ingress = ingress();
    let mut connections = Connections::new(&ingress, Arc::new(AtomicBool::new(false))).unwrap();
    let (_client, finished) = blocked_reader(&mut connections, &ingress);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _owned = connections;
        panic!("startup/accept unwind after handler creation");
    }));
    assert!(result.is_err());
    finished.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(ingress.0.serving_bytes(), 0);
}

#[test]
fn panicked_handler_is_a_failure_and_does_not_skip_other_joins() {
    let ingress = ingress();
    let mut connections = Connections::new(&ingress, Arc::new(AtomicBool::new(false))).unwrap();
    let (_panicked_client, stream) = pair();
    let handle = std::thread::spawn(|| panic!("injected handler panic"));
    connections.entries.push(Connection {
        handle,
        interrupt: stream,
        _charge: ingress.charge_http_ingress("connection", 128).unwrap(),
    });
    let (_blocked_client, finished) = blocked_reader(&mut connections, &ingress);
    assert!(connections.finish().is_err());
    finished.recv_timeout(Duration::from_secs(3)).unwrap();
    drop(connections);
    assert_eq!(ingress.0.serving_bytes(), 0);
}
