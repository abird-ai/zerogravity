//! Actual decoded Hyper frames over a socket. The fixture exercises the
//! private staging primitive, not an advertised chunked public endpoint.
use super::*;
use crate::{
    http::body,
    test_support::{self, HostBytes},
};
use hyper::{Request, service::service_fn};
use hyper_util::rt::TokioIo;
use serde_json::json;
use std::convert::Infallible;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

fn exchange(
    declared: Option<usize>,
    cap: usize,
    head: &str,
    chunks: &[&[u8]],
    expected: Result<&[u8], u16>,
    deadline: Duration,
) {
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let service = test_support::idle(&host);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = async {
            let (socket, _) = listener.accept().await.unwrap();
            let fixture = service_fn(|request: Request<Incoming>| {
                let service = service.clone();
                let host = host.clone();
                async move {
                    let result = match Reader::new(request.into_body(), declared, cap, deadline) {
                        Ok(reader) => collect(reader, service.as_ref()).await,
                        Err(error) => Err(error),
                    };
                    let status = match (result, expected) {
                        (Ok(body), Ok(expected)) => {
                            let (backing, used, charge) = body.into_parts();
                            assert_eq!(&backing[..used], expected);
                            assert!(backing.len() <= cap);
                            if let Some(length) = declared {
                                assert_eq!(backing.len(), length);
                            }
                            // Compare the full charged lease, including the
                            // arbiter's bookkeeping, not just payload bytes.
                            let retained = host.serving_bytes();
                            let reference = service
                                .charge_ingress("http-body", backing.len() as u64)
                                .unwrap();
                            assert_eq!(host.serving_bytes() - retained, retained - baseline);
                            drop(reference);
                            drop(backing);
                            drop(charge);
                            200
                        }
                        (Err(error), Err(status)) => {
                            assert_eq!(error.status, status);
                            status
                        }
                        (Ok(_), Err(status)) => panic!("expected refusal {status}"),
                        (Err(error), Ok(_)) => panic!("unexpected refusal: {}", error.code),
                    };
                    assert_eq!(host.serving_bytes(), baseline);
                    Ok::<_, Infallible>(body::json_response(
                        status,
                        &json!({"status":status}),
                        None,
                    ))
                }
            });
            // The reader consumes decoded frames, not a second HTTP parser.
            let _ = hyper::server::conn::http1::Builder::new()
                .half_close(true)
                .keep_alive(false)
                .max_buf_size(8192)
                .serve_connection(TokioIo::new(socket), fixture)
                .await;
        };
        let client = async {
            let mut socket = TcpStream::connect(address).await.unwrap();
            socket.write_all(head.as_bytes()).await.unwrap();
            for chunk in chunks {
                socket.write_all(chunk).await.unwrap();
                // Separate deliveries exercise the incremental backing.
                time::sleep(Duration::from_millis(2)).await;
            }
            let mut raw = Vec::new();
            socket.read_to_end(&mut raw).await.unwrap();
            let status = expected.err().unwrap_or(200);
            assert!(
                raw.starts_with(format!("HTTP/1.1 {status}").as_bytes()),
                "unexpected HTTP status: {}",
                String::from_utf8_lossy(&raw)
            );
        };
        time::timeout(Duration::from_secs(3), async {
            tokio::join!(server, client);
        })
        .await
        .unwrap();
    });
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn actual_fixed_and_unknown_length_staging_charge_and_release() {
    let _lock = test_support::lock();
    let head = "POST /fixture HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\nConnection: close\r\n\r\n";
    exchange(
        Some(5),
        5,
        head,
        &[b"he", b"llo"],
        Ok(b"hello"),
        Duration::from_secs(1),
    );

    let head = "POST /fixture HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    let a = vec![b'a'; 4096];
    let b = vec![b'b'; 5000];
    let mut first = b"1000\r\n".to_vec();
    first.extend_from_slice(&a);
    first.extend_from_slice(b"\r\n");
    let mut second = b"1388\r\n".to_vec();
    second.extend_from_slice(&b);
    second.extend_from_slice(b"\r\n0\r\n\r\n");
    let mut expected = a;
    expected.extend_from_slice(&b);
    exchange(
        None,
        expected.len(),
        head,
        &[&first, &second],
        Ok(&expected),
        Duration::from_secs(1),
    );
    exchange(
        None,
        16,
        head,
        &[b"0\r\n\r\n"],
        Ok(b""),
        Duration::from_secs(1),
    );
}

#[test]
fn actual_unknown_length_overflow_trailer_and_deadline_release_staging() {
    let _lock = test_support::lock();
    let head = "POST /fixture HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    exchange(
        None,
        8,
        head,
        &[b"4\r\nabcd\r\n", b"5\r\nefghi\r\n0\r\n\r\n"],
        Err(413),
        Duration::from_secs(1),
    );
    exchange(
        None,
        8,
        head,
        &[b"4\r\nabcd\r\n0\r\nX-Trailer: ignored\r\n\r\n"],
        Err(400),
        Duration::from_secs(1),
    );
    exchange(
        None,
        8,
        head,
        &[b"4\r\nabcd\r\n"],
        Err(408),
        Duration::from_millis(80),
    );
}
