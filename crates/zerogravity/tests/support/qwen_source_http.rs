//! Raw-text HTTP protocol checks shared by source-backed GPU diagnostics.
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    time::Duration,
};

pub(super) fn connect(address: SocketAddr, path: &str, body: &Value, id: &str) -> TcpStream {
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
pub(super) fn headers(reader: &mut impl BufRead) -> String {
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
pub(super) fn chunk(reader: &mut impl BufRead) -> Option<String> {
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
pub(super) fn verify(document: &Value, expected: &[u32], id: &str, prompt_tokens: usize) -> u64 {
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
pub(super) fn completion(
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

pub(super) fn refuse_chat(address: SocketAddr) {
    // HTTP framing must not bypass the original raw-text policy.
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
}
