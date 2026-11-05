//! Public loopback HTTP on the same runtime/worker; supervised private netns only.
#[path = "qwen_hip_http/progress.rs"]
mod progress;

use super::{population::Harness, *};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    io::{BufRead, BufReader, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    sync::Barrier,
};

fn connect(address: SocketAddr, path: &str, body: &Value, id: &str) -> TcpStream {
    let body = serde_json::to_vec(body).unwrap();
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(1200)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    write!(socket, "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nX-Request-ID: {id}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
    socket.write_all(&body).unwrap();
    socket
}
fn headers(reader: &mut impl BufRead) -> String {
    let mut result = String::new();
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        assert!(result.len() + line.len() <= 8192);
        result.push_str(&line);
        if line == "\r\n" {
            return result;
        }
    }
}
fn chunk(reader: &mut impl BufRead) -> Option<String> {
    let mut size = String::new();
    reader.read_line(&mut size).unwrap();
    let n = usize::from_str_radix(size.trim(), 16).unwrap();
    assert!(n <= 64 * 1024);
    let mut body = vec![0; n];
    reader.read_exact(&mut body).unwrap();
    let mut end = [0; 2];
    reader.read_exact(&mut end).unwrap();
    assert_eq!(end, *b"\r\n");
    (n != 0).then(|| String::from_utf8(body).unwrap())
}
fn verify(document: &Value, expected: &[u32], id: &str, prompt_tokens: usize) -> u64 {
    assert!(document.get("error").is_none(), "{document}");
    assert_eq!(document["usage"]["prompt_tokens"], prompt_tokens);
    assert_eq!(document["usage"]["completion_tokens"], expected.len());
    assert_eq!(
        document["usage"]["total_tokens"],
        prompt_tokens + expected.len()
    );
    let metrics = &document["zerogravity_token_metrics"];
    assert_eq!(metrics["wire_request_id"], id);
    let tokens = metrics["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), expected.len());
    let first = &tokens[0]["gpu"]["producer_root"];
    for (i, token) in tokens.iter().enumerate() {
        assert_eq!(token["token_id"], expected[i]);
        assert_eq!(token["gpu"]["producer_frontier"], prompt_tokens + i);
        assert_eq!(
            token["gpu"]["producer_root"]["session_id"],
            first["session_id"]
        );
        assert_eq!(
            token["gpu"]["producer_root"]["session_epoch"],
            first["session_epoch"]
        );
        assert!(token["gpu"]["native_cohort"].is_null());
        if i != 0 {
            assert_ne!(
                token["gpu"]["producer_root"]["root"],
                tokens[i - 1]["gpu"]["producer_root"]["root"]
            );
        }
    }
    first["session_id"].as_u64().unwrap()
}
fn completion(
    address: SocketAddr,
    body: Value,
    expected: &[u32],
    text: &str,
    id: &str,
    disconnect: bool,
    prompt_tokens: usize,
) -> Option<(String, u64)> {
    let streaming = body["stream"] == true;
    let socket = connect(address, "/v1/completions", &body, id);
    let mut reader = BufReader::new(socket);
    let head = headers(&mut reader);
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    if !streaming {
        let mut raw = Vec::new();
        reader.take(64 * 1024 + 1).read_to_end(&mut raw).unwrap();
        assert!(raw.len() <= 64 * 1024);
        let doc: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(doc["choices"][0]["text"], text);
        assert_eq!(doc["choices"][0]["token_ids"], json!(expected));
        let session = verify(&doc, expected, id, prompt_tokens);
        return Some((doc["id"].as_str().unwrap().into(), session));
    }
    assert!(head.contains("Transfer-Encoding: chunked\r\n"));
    let mut ids = Vec::new();
    let mut output = String::new();
    let mut terminal = None;
    let mut done = 0;
    let mut total = 0;
    while let Some(data) = chunk(&mut reader) {
        total += data.len();
        assert!(total <= 64 * 1024);
        if data.starts_with(':') {
            continue;
        }
        let data = data.strip_prefix("data: ").unwrap().trim_end();
        if data == "[DONE]" {
            done += 1;
            continue;
        }
        let doc: Value = serde_json::from_str(data).unwrap();
        assert!(doc.get("error").is_none(), "{doc}");
        if let Some(tokens) = doc["choices"][0]["token_ids"].as_array() {
            ids.extend(tokens.iter().map(|v| v.as_u64().unwrap() as u32));
            output.push_str(doc["choices"][0]["text"].as_str().unwrap());
            if disconnect {
                assert_eq!(ids, expected[..ids.len()]);
                reader.get_ref().shutdown(Shutdown::Both).unwrap();
                return None;
            }
        }
        if doc.get("usage").is_some() {
            assert!(terminal.is_none());
            let session = verify(&doc, expected, id, prompt_tokens);
            terminal = Some((doc["id"].as_str().unwrap().into(), session));
        }
    }
    assert_eq!(done, 1);
    assert_eq!(ids, expected);
    assert_eq!(output, text);
    Some(terminal.expect("one terminal usage/root record"))
}

#[test]
#[ignore = "approved gfx1151/source, private loopback netns, fresh release process"]
fn public_source_http_population_json_sse_disconnect_and_replay() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let clients: usize = std::env::var("ZG_QWEN_HIP_CLIENTS")
        .unwrap()
        .parse()
        .unwrap();
    let mode = std::env::var("ZG_QWEN_HIP_POPULATION").unwrap();
    assert!([1, 4, 8, 16, 32, 64].contains(&clients));
    assert!(matches!(mode.as_str(), "complete" | "disconnect"));
    // The launch wrapper verifies namespace ownership before enabling only lo.
    assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
    let h = Harness::open(&label, clients.min(8), 514, 3, clients + 1);
    let (prompt, expected) = h.canonical();
    let expected_text = h.tokenizer.decode(&expected, true).unwrap();
    let server = h
        .runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(zerogravity::serving::ingress::IngressLimits {
            connections: clients,
            body_bytes: 64 * 1024,
            output_bytes: 64 * 1024,
        })
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    // HTTP's already-framed chat must not bypass the original raw-text policy.
    let mut refused = BufReader::new(connect(
        address,
        "/v1/chat/completions",
        &json!({"model":"source", "messages":[{"role":"user","content":"hi"}],"max_tokens":3}),
        "chat-refusal",
    ));
    assert!(headers(&mut refused).starts_with("HTTP/1.1 400"));
    let mut body = String::new();
    refused.take(64 * 1024).read_to_string(&mut body).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "unsupported_chat"
    );
    let barrier = Barrier::new(clients);
    let outcomes = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for i in 0..clients {
            let disconnect = mode == "disconnect" && i % 3 == 0;
            let stream = disconnect || i % 2 != 0;
            let body = json!({"model":if i%2==0 {"source"} else {"alias"},"prompt":prompt,"max_tokens":3,"stream":stream,"return_token_ids":true});
            let (barrier, expected, text) = (&barrier, &expected, &expected_text);
            handles.push(scope.spawn(move || {
                barrier.wait();
                completion(
                    address,
                    body,
                    expected,
                    text,
                    &format!("population-{i}"),
                    disconnect,
                    47,
                )
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut responses = BTreeSet::new();
    let mut sessions = BTreeSet::new();
    let mut disconnects = 0;
    for outcome in outcomes {
        if let Some((response, session)) = outcome {
            assert!(responses.insert(response));
            assert!(sessions.insert(session));
        } else {
            disconnects += 1;
        }
    }
    assert_eq!(responses.len() + disconnects, clients);
    let metrics = server.metrics();
    server.shutdown().unwrap();
    assert!(TcpStream::connect(address).is_err());
    assert!(!h.runtime.status().closed);
    let replay = complete(
        h.runtime.request_text(request("alias", &prompt)).unwrap(),
        47,
        h.tokenizer.as_ref(),
    );
    assert_eq!(replay.ids, expected);
    assert!(sessions.insert(replay.roots[0].producer_root.unwrap().session_id));
    let admission = h.admission();
    assert_eq!(admission["failed_gpu_quanta"], 0);
    assert_eq!(admission["gpu_fused_batches"], 0);
    println!(
        "SOURCE_HTTP_OUTCOME model={label} mode={mode} offered={clients} completed_readers={} disconnect_requested={disconnects} active_limit={} accepted_http_requests={} native_cohorts=false original_ids=true sdk_replay=true full_float_comparisons=0 admission={admission}",
        responses.len(),
        clients.min(8),
        metrics.accepted_requests
    );
    h.retire();
    println!("SOURCE_HTTP_PASS model={label} mode={mode} offered={clients}");
}

#[test]
#[ignore = "approved populated original, gfx1151, private loopback, fresh release process"]
fn public_source_populated_http_json_sse_disconnect_and_replay() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let capacity: usize = std::env::var("ZG_QWEN_CONTEXT").unwrap().parse().unwrap();
    let case = [1024, 2048, 4096, 8192, 16384]
        .iter()
        .position(|n| *n == capacity)
        .unwrap();
    let mode = std::env::var("ZG_QWEN_HIP_POPULATION").unwrap();
    assert!(matches!(mode.as_str(), "complete" | "disconnect"));
    assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
    let h = Harness::open(&label, 2, capacity, 3, 3);
    let reference = super::contexts::ContextReference::open(&label);
    let (prompt, expected) = reference.trajectory(case, capacity, h.tokenizer.as_ref());
    let expected_text = h.tokenizer.decode(&expected, true).unwrap();
    let server = h
        .runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(zerogravity::serving::ingress::IngressLimits {
            connections: 2,
            body_bytes: 128 * 1024,
            output_bytes: 64 * 1024,
        })
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    // Long transport framing here; SDK/context and population cover concurrency.
    let outcomes = (0..2).map(|i| {
        let body = json!({"model":if i == 0 {"source"} else {"alias"},"prompt":prompt,"max_tokens":3,"stream":i==0,"return_token_ids":true});
        assert!(serde_json::to_vec(&body).unwrap().len() <= 128*1024);
        completion(address,body,&expected,&expected_text,&format!("context-{i}"),mode == "disconnect" && i == 0,capacity-2)
    }).collect::<Vec<_>>();
    let mut sessions = BTreeSet::new();
    let mut responses = BTreeSet::new();
    let mut disconnects = 0;
    for outcome in outcomes {
        if let Some((response, session)) = outcome {
            assert!(responses.insert(response));
            assert!(sessions.insert(session));
        } else {
            disconnects += 1;
        }
    }
    assert_eq!(disconnects, usize::from(mode == "disconnect"));
    assert_eq!(sessions.len() + disconnects, 2);
    server.shutdown().unwrap();
    assert!(TcpStream::connect(address).is_err());
    assert!(!h.runtime.status().closed);
    let replay = complete(
        h.runtime.request_text(request("source", &prompt)).unwrap(),
        capacity - 2,
        h.tokenizer.as_ref(),
    );
    assert_eq!(replay.ids, expected);
    assert!(sessions.insert(replay.roots[0].producer_root.unwrap().session_id));
    let admission = h.admission();
    assert_eq!(admission["failed_gpu_quanta"], 0);
    assert_eq!(admission["queued_requests"], 0);
    assert_eq!(admission["gpu_fused_batches"], 0);
    println!(
        "SOURCE_CONTEXT_HTTP_OUTCOME model={label} context={capacity} mode={mode} offered=2 concurrent_clients=1 completed_readers={} disconnect_requested={disconnects} prompt_tokens={} original_ids=true sdk_replay=true native_cohorts=false full_float_comparisons=0 admission={admission}",
        responses.len(),
        capacity - 2
    );
    drop(reference);
    h.retire();
    println!("SOURCE_CONTEXT_HTTP_PASS model={label} context={capacity} mode={mode}");
}
