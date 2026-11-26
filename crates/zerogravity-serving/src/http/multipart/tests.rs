//! Real TCP fixtures for the private form decoder; no public media route.
use super::*;
use crate::{
    http::{body, transport},
    test_support::{self, HostBytes},
};
use hyper::{Request, body::Incoming, service::service_fn};
use hyper_util::rt::TokioIo;
use serde_json::json;
use std::{convert::Infallible, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

const CONTENT_TYPE: &str = "multipart/form-data; boundary=zg-boundary";
const PART: &str = "--zg-boundary\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhello";
const END: &str = "\r\n--zg-boundary--\r\n";

#[derive(Clone, Copy)]
struct Fixture {
    cap: usize,
    limits: Limits,
    deadline: Duration,
    available: Option<u64>,
}
impl Default for Fixture {
    fn default() -> Self {
        Self {
            cap: 64 * 1024,
            limits: Limits {
                parts: 4,
                part_bytes: 16 * 1024,
            },
            deadline: Duration::from_secs(2),
            available: None,
        }
    }
}

struct Expected<'a> {
    name: &'a str,
    filename: Option<&'a str>,
    content_type: Option<&'a str>,
    data: &'a [u8],
}

fn head(content_type: &str, length: Option<usize>, authorized: bool) -> String {
    let framing = length.map_or_else(
        || "Transfer-Encoding: chunked\r\n".to_owned(),
        |n| format!("Content-Length: {n}\r\n"),
    );
    let auth = if authorized {
        "Authorization: Bearer secret\r\n"
    } else {
        ""
    };
    format!(
        "POST /fixture HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\n{framing}{auth}Connection: close\r\n\r\n"
    )
}

fn chunked(data: &[u8], width: usize, end: &[u8]) -> Vec<Vec<u8>> {
    let mut chunks = data
        .chunks(width)
        .map(|data| {
            let mut out = format!("{:x}\r\n", data.len()).into_bytes();
            out.extend_from_slice(data);
            out.extend_from_slice(b"\r\n");
            out
        })
        .collect::<Vec<_>>();
    if !end.is_empty() {
        chunks.push(end.to_vec());
    }
    chunks
}

async fn receive(
    request: Request<Incoming>,
    service: &dyn TextService,
    fixture: Fixture,
) -> Result<Form, ServiceError> {
    let (parts, incoming) = request.into_parts();
    let length = transport::validate_headers(&parts, fixture.cap, true)?;
    transport::authorize(&parts.headers, Some("secret"))?;
    assert_eq!(parts.method, hyper::Method::POST);
    assert_eq!(parts.uri.path(), "/fixture");
    let content_type = parts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(invalid)?;
    collect(
        Reader::new(incoming, length, fixture.cap, fixture.deadline)?,
        content_type,
        service,
        fixture.limits,
    )
    .await
}

fn exchange(
    fixture: Fixture,
    header: &str,
    chunks: &[&[u8]],
    expected: Result<&[Expected<'_>], u16>,
) {
    let host = test_support::host();
    let service = test_support::idle(&host);
    let _pressure = fixture.available.map(|available| {
        service
            .charge_ingress(
                "fixture-pressure",
                host.snapshot().unwrap().2.saturating_sub(available),
            )
            .unwrap()
    });
    let baseline = host.serving_bytes();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = async {
            let (socket, _) = listener.accept().await.unwrap();
            let fixture_service = service_fn(|request| {
                let service = service.clone();
                let host = host.clone();
                let expected = expected.as_ref();
                async move {
                    let outcome = receive(request, service.as_ref(), fixture).await;
                    let status = match (outcome, expected) {
                        (Ok(form), Ok(expected)) => {
                            assert_eq!(form.parts().len(), expected.len());
                            for (part, expected) in form.parts().iter().zip(expected.iter()) {
                                assert_eq!(part.name, expected.name);
                                assert_eq!(part.filename.as_deref(), expected.filename);
                                assert_eq!(part.content_type.as_deref(), expected.content_type);
                                assert_eq!(part.bytes(), expected.data);
                            }
                            assert!(host.serving_bytes() > baseline);
                            drop(form);
                            200
                        }
                        (Err(error), Err(status)) => {
                            assert_eq!(error.status, *status, "{}", error.code);
                            *status
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
            let _ = hyper::server::conn::http1::Builder::new()
                .half_close(true)
                .keep_alive(false)
                .max_buf_size(8192)
                .max_header_size(8192)
                .max_headers(32)
                .serve_connection(TokioIo::new(socket), fixture_service)
                .await;
        };
        let client = async {
            let mut socket = TcpStream::connect(address).await.unwrap();
            socket.write_all(header.as_bytes()).await.unwrap();
            for chunk in chunks {
                socket.write_all(chunk).await.unwrap();
                if chunks.len() > 1 {
                    time::sleep(Duration::from_millis(1)).await;
                }
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
        time::timeout(Duration::from_secs(6), async {
            tokio::join!(server, client);
        })
        .await
        .unwrap();
    });
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn actual_fixed_and_chunked_forms_preserve_parts_and_retire_admission() {
    let _lock = test_support::lock();
    let mut data = PART.as_bytes().to_vec();
    data.extend_from_slice(b"\r\n--zg-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n");
    let binary = [0, 255, 1, 13, 10, 2].repeat(1000);
    data.extend_from_slice(&binary);
    data.extend_from_slice(
        b"\r\n--zg-boundary\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\n",
    );
    data.extend_from_slice(END.as_bytes());
    let expected = [
        Expected {
            name: "prompt",
            filename: None,
            content_type: None,
            data: b"hello",
        },
        Expected {
            name: "file",
            filename: Some("audio.bin"),
            content_type: Some("application/octet-stream"),
            data: &binary,
        },
        Expected {
            name: "file",
            filename: None,
            content_type: None,
            data: b"",
        },
    ];
    let cfg = Fixture {
        cap: data.len(),
        ..Default::default()
    };
    exchange(
        cfg,
        &head(CONTENT_TYPE, Some(data.len()), true),
        &[&data],
        Ok(&expected),
    );
    let chunks = chunked(&data, 137, b"0\r\n\r\n");
    let refs = chunks.iter().map(Vec::as_slice).collect::<Vec<_>>();
    exchange(cfg, &head(CONTENT_TYPE, None, true), &refs, Ok(&expected));
    let simple = format!("{PART}{END}");
    let pieces = chunked(simple.as_bytes(), 1, b"0\r\n\r\n");
    let refs = pieces.iter().map(Vec::as_slice).collect::<Vec<_>>();
    exchange(
        Fixture::default(),
        &head(CONTENT_TYPE, None, true),
        &refs,
        Ok(&expected[..1]),
    );
    let empty = b"--zg-boundary--";
    exchange(
        Fixture::default(),
        &head(CONTENT_TYPE, Some(empty.len()), true),
        &[empty],
        Ok(&[]),
    );
}

#[test]
fn actual_malformed_and_oversized_forms_never_publish() {
    let _lock = test_support::lock();
    let cfg = Fixture::default();
    for data in [
        PART.to_owned(),
        format!("{PART}\r\n--zg-boundary--garbage"),
        format!("{PART}{END}epilogue"),
        format!("{PART}\r\n--zg-boundary--\r"),
        format!("preamble\r\n{PART}{END}"),
        "--zg-boundary\r\nContent-Disposition: form-data; name=\"x\"".into(),
        format!(
            "--zg-boundary\r\nContent-Disposition: form-data; name=\"x\"\r\nContent-Disposition: form-data; name=\"y\"\r\n\r\nx{END}"
        ),
        format!("--zg-boundary\r\nContent-Disposition: attachment; name=\"x\"\r\n\r\nx{END}"),
    ] {
        exchange(
            cfg,
            &head(CONTENT_TYPE, Some(data.len()), true),
            &[data.as_bytes()],
            Err(400),
        );
    }
    let data = format!("{PART}{END}");
    let part_bound = Fixture {
        limits: Limits {
            parts: 4,
            part_bytes: 4,
        },
        ..cfg
    };
    exchange(
        part_bound,
        &head(CONTENT_TYPE, Some(data.len()), true),
        &[data.as_bytes()],
        Err(413),
    );
    let two = format!("{PART}\r\n{PART}{END}");
    let count_bound = Fixture {
        limits: Limits {
            parts: 1,
            part_bytes: 16 * 1024,
        },
        ..cfg
    };
    exchange(
        count_bound,
        &head(CONTENT_TYPE, Some(two.len()), true),
        &[two.as_bytes()],
        Err(413),
    );
    let aggregate_bound = Fixture {
        cap: data.len() - 1,
        ..cfg
    };
    exchange(
        aggregate_bound,
        &head(CONTENT_TYPE, Some(data.len()), true),
        &[],
        Err(413),
    );
    let wire = chunked(data.as_bytes(), data.len(), b"0\r\n\r\n").concat();
    exchange(
        aggregate_bound,
        &head(CONTENT_TYPE, None, true),
        &[&wire],
        Err(413),
    );
}

#[test]
fn actual_form_requires_authorization_and_complete_http_body() {
    let _lock = test_support::lock();
    let cfg = Fixture::default();
    exchange(
        Fixture {
            available: Some(32 * 1024),
            ..cfg
        },
        &head(CONTENT_TYPE, None, true),
        &[],
        Err(503),
    );
    exchange(cfg, &head(CONTENT_TYPE, None, false), &[], Err(401));
    exchange(cfg, &head("application/json", None, true), &[], Err(415));
    exchange(cfg, &head("multipart/form-data", None, true), &[], Err(400));
    let data = format!("{PART}{END}");
    let wire = chunked(
        data.as_bytes(),
        data.len(),
        b"0\r\nX-Trace: ignored\r\n\r\n",
    )
    .concat();
    exchange(cfg, &head(CONTENT_TYPE, None, true), &[&wire], Err(400));
    let wire = chunked(data.as_bytes(), data.len(), b"").concat();
    exchange(
        Fixture {
            deadline: Duration::from_millis(100),
            ..cfg
        },
        &head(CONTENT_TYPE, None, true),
        &[&wire],
        Err(408),
    );
}

#[test]
fn cancelled_partial_form_releases_parser_and_staged_parts() {
    let _lock = test_support::lock();
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
        let mut client = TcpStream::connect(address).await.unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let fixture = service_fn(|request| {
            let service = service.clone();
            async move {
                let form = receive(request, service.as_ref(), Fixture::default()).await;
                assert!(form.is_err(), "partial form published");
                Ok::<_, Infallible>(body::json_response(400, &json!({}), None))
            }
        });
        let mut connection = Box::pin(
            hyper::server::conn::http1::Builder::new()
                .keep_alive(false)
                .max_buf_size(8192)
                .serve_connection(TokioIo::new(socket), fixture),
        );
        client
            .write_all(head(CONTENT_TYPE, None, true).as_bytes())
            .await
            .unwrap();
        let raw = format!("{PART}\r\n{PART}");
        let wire = chunked(raw.as_bytes(), raw.len(), b"").concat();
        client.write_all(&wire).await.unwrap();
        let admitted = async {
            while host.serving_bytes() == baseline {
                time::sleep(Duration::from_millis(2)).await;
            }
            time::sleep(Duration::from_millis(10)).await;
        };
        tokio::select! {
            _ = &mut connection => panic!("partial connection ended before cancellation"),
            result = time::timeout(Duration::from_secs(1), admitted) => result.unwrap(),
        }
        assert!(host.serving_bytes() > baseline);
        // Same ownership boundary as stopping the enclosing connection task.
        drop(connection);
        drop(client);
        assert_eq!(host.serving_bytes(), baseline);
    });
    assert_eq!(host.serving_bytes(), baseline);
}
