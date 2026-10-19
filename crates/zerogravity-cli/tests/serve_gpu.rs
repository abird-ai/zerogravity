//! Ignored release-only proof over actual CUDA HTTP requests and original weights.
//! Run serially on the qualified CUDA host; no source acquisition or speed claim.
#![cfg(all(feature = "serve", feature = "cuda"))]

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use zerogravity::qwen35::{LoadedQwenModel, QwenDenseCpu};
use zerogravity::tokenize::QwenTokenizer;

type Proof<T> = Result<T, String>;
const REQUEST_LIMIT: Duration = Duration::from_secs(180);
const BODY_LIMIT: usize = 4 * 1024 * 1024;
static GPU_HTTP_LOCK: Mutex<()> = Mutex::new(());

struct OwnedServer {
    child: Child,
    address: SocketAddr,
    aliases: [String; 2],
    stderr: Arc<Mutex<Vec<u8>>>,
    stderr_reader: Option<JoinHandle<()>>,
}

impl OwnedServer {
    fn start(source: &Path) -> Proof<Self> {
        Self::start_rows(source, 2)
    }

    fn start_rows(source: &Path, rows: usize) -> Proof<Self> {
        Self::start_configured(source, rows, false)
    }

    fn start_configured(source: &Path, rows: usize, metrics: bool) -> Proof<Self> {
        if !matches!(rows, 2 | 4 | 8) {
            return Err("physical HTTP server requires native rows2/4/8".into());
        }
        for _ in 0..3 {
            let reserved = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
            let address = reserved.local_addr().map_err(|e| e.to_string())?;
            let aliases = [
                format!("http-a-{}", address.port()),
                format!("http-b-{}", address.port()),
            ];
            drop(reserved);
            let mut command = Command::new(env!("CARGO_BIN_EXE_zerogravity"));
            command.arg("serve");
            if metrics {
                command.arg("--enable-per-request-metrics");
            }
            let mut child = command
                .arg("--backend")
                .arg("cuda")
                .arg("--batch-mode")
                .arg("fused")
                .arg("--batch-rows")
                .arg(rows.to_string())
                .arg("--listen")
                .arg(address.to_string())
                .arg("--workers")
                .arg("1")
                .arg("--active-per-worker")
                .arg(if rows == 2 { "4" } else { "8" })
                .arg("--max-context")
                .arg("512")
                .arg("--max-output")
                .arg("8")
                .arg("--provider-mib")
                .arg("4096")
                .arg("--device-mib")
                .arg("5120")
                .arg("--model")
                .arg(format!("{}={}", aliases[0], source.display()))
                .arg("--model")
                .arg(format!("{}={}", aliases[1], source.display()))
                .env_remove("ZEROGRAVITY_API_KEY")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| e.to_string())?;
            let mut pipe = child
                .stderr
                .take()
                .ok_or("owned server missing stderr pipe")?;
            let stderr = Arc::new(Mutex::new(Vec::<u8>::new()));
            let captured = Arc::clone(&stderr);
            let stderr_reader = thread::spawn(move || {
                let mut chunk = [0u8; 4096];
                while let Ok(count) = pipe.read(&mut chunk) {
                    if count == 0 {
                        break;
                    }
                    let mut log = captured.lock().expect("owned server diagnostic mutex");
                    log.extend_from_slice(&chunk[..count]);
                    if log.len() > 64 * 1024 {
                        let excess = log.len() - 64 * 1024;
                        drop(log.drain(..excess));
                    }
                }
            });
            let mut server = Self {
                child,
                address,
                aliases,
                stderr,
                stderr_reader: Some(stderr_reader),
            };
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                if let Some(status) = server.child.try_wait().map_err(|e| e.to_string())? {
                    server.stop();
                    let log = server.diagnostics();
                    if log.to_ascii_lowercase().contains("address already in use") {
                        break;
                    }
                    return Err(format!(
                        "owned HTTP server exited before readiness: {status}\n{log}"
                    ));
                }
                if let Ok(health) = server.health()
                    && health["status"] == "ok"
                    && health["admission"]["batch_mode"] == "fused"
                    && health["admission"]["native_batch_rows_max"] == rows
                    && let Ok(models) = get_json(server.address, "/v1/models")
                    && models["data"].as_array().is_some_and(|models| {
                        models.len() == 2
                            && server
                                .aliases
                                .iter()
                                .all(|alias| models.iter().any(|m| m["id"] == *alias))
                    })
                {
                    return Ok(server);
                }
                if Instant::now() >= deadline {
                    return Err(format!(
                        "owned HTTP readiness deadline\n{}",
                        server.diagnostics()
                    ));
                }
                thread::sleep(Duration::from_millis(25));
            }
        }
        Err("owned HTTP ephemeral port lost three bind races".into())
    }

    fn diagnostics(&self) -> String {
        String::from_utf8_lossy(&self.stderr.lock().expect("diagnostic mutex")).into_owned()
    }

    fn stop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        if let Some(reader) = self.stderr_reader.take() {
            let _ = reader.join();
        }
    }

    fn health(&self) -> Proof<Value> {
        get_json(self.address, "/health")
    }

    fn wait_health(&self, predicate: impl Fn(&Value) -> bool) -> Proof<Value> {
        self.wait_health_for(REQUEST_LIMIT, predicate)
    }

    fn wait_health_for(&self, limit: Duration, predicate: impl Fn(&Value) -> bool) -> Proof<Value> {
        let deadline = Instant::now() + limit;
        let mut last = Value::Null;
        while Instant::now() < deadline {
            if let Ok(health) = self.health() {
                if predicate(&health["admission"]) {
                    return Ok(health["admission"].clone());
                }
                last = health;
            }
            thread::sleep(Duration::from_millis(10));
        }
        Err(format!("HTTP health condition deadline; last={last}"))
    }
}

impl Drop for OwnedServer {
    fn drop(&mut self) {
        self.stop();
        if thread::panicking() {
            eprintln!("owned HTTP server stderr:\n{}", self.diagnostics());
        }
    }
}

struct Response {
    reader: BufReader<TcpStream>,
    status: u16,
    chunked: bool,
    length: Option<usize>,
}

impl Response {
    fn read(stream: TcpStream) -> Proof<Self> {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let status = line
            .split_whitespace()
            .nth(1)
            .ok_or("missing HTTP status")?
            .parse()
            .map_err(|e| format!("invalid HTTP status: {e}"))?;
        let mut chunked = false;
        let mut length = None;
        let mut headers = line.len();
        loop {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("EOF in HTTP headers".into());
            }
            headers += line.len();
            if headers > 8192 {
                return Err("HTTP response headers exceed bound".into());
            }
            if line == "\r\n" {
                break;
            }
            let (name, value) = line.split_once(':').ok_or("invalid HTTP response header")?;
            if name.eq_ignore_ascii_case("transfer-encoding") {
                chunked = value.trim().eq_ignore_ascii_case("chunked");
            }
            if name.eq_ignore_ascii_case("content-length") {
                length = Some(value.trim().parse::<usize>().map_err(|e| e.to_string())?);
            }
        }
        Ok(Self {
            reader,
            status,
            chunked,
            length,
        })
    }

    fn chunk(&mut self) -> Proof<Option<Vec<u8>>> {
        if !self.chunked {
            return Err("expected actual chunked HTTP transport".into());
        }
        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if line.len() > 128 || !line.ends_with("\r\n") {
            return Err("bad chunk size framing".into());
        }
        let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or_default(), 16)
            .map_err(|e| e.to_string())?;
        if size > BODY_LIMIT {
            return Err("HTTP chunk exceeds bound".into());
        }
        if size == 0 {
            loop {
                line.clear();
                let count = self
                    .reader
                    .read_line(&mut line)
                    .map_err(|e| e.to_string())?;
                if count == 0 {
                    return Err("EOF in HTTP chunk trailers".into());
                }
                if line == "\r\n" {
                    break;
                }
                if line.len() > 8192 {
                    return Err("HTTP trailer exceeds bound".into());
                }
            }
            return Ok(None);
        }
        let mut data = vec![0; size];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| e.to_string())?;
        let mut ending = [0; 2];
        self.reader
            .read_exact(&mut ending)
            .map_err(|e| e.to_string())?;
        if ending != *b"\r\n" {
            return Err("bad HTTP chunk terminator".into());
        }
        Ok(Some(data))
    }

    fn body(&mut self) -> Proof<Vec<u8>> {
        let mut body = Vec::new();
        if self.chunked {
            while let Some(chunk) = self.chunk()? {
                if body.len() + chunk.len() > BODY_LIMIT {
                    return Err("HTTP body exceeds bound".into());
                }
                body.extend(chunk);
            }
        } else if let Some(length) = self.length {
            if length > BODY_LIMIT {
                return Err("HTTP body exceeds bound".into());
            }
            body.resize(length, 0);
            self.reader
                .read_exact(&mut body)
                .map_err(|e| e.to_string())?;
        } else {
            self.reader
                .by_ref()
                .take(BODY_LIMIT as u64 + 1)
                .read_to_end(&mut body)
                .map_err(|e| e.to_string())?;
            if body.len() > BODY_LIMIT {
                return Err("HTTP body exceeds bound".into());
            }
        }
        Ok(body)
    }

    fn disconnect(&self) {
        let _ = self.reader.get_ref().shutdown(Shutdown::Both);
    }
}

fn open_request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> Proof<TcpStream> {
    open_bound_request(address, method, path, body, None)
}

fn open_bound_request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
    wire_id: Option<&str>,
) -> Proof<TcpStream> {
    let mut stream =
        TcpStream::connect_timeout(&address, Duration::from_secs(2)).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(REQUEST_LIMIT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    let body = body
        .map(serde_json::to_vec)
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let identity = wire_id
        .map(|id| format!("X-Request-ID: {id}\r\n"))
        .unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Type: application/json\r\n{identity}Content-Length: {}\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(&body))
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

fn get_json(address: SocketAddr, path: &str) -> Proof<Value> {
    let stream = open_request(address, "GET", path, None)?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| e.to_string())?;
    let mut response = Response::read(stream)?;
    let bytes = response.body()?;
    if response.status != 200 {
        return Err(format!(
            "GET {path}: {} {}",
            response.status,
            String::from_utf8_lossy(&bytes)
        ));
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn payload(alias: &str, prompt: &str, max_tokens: usize, stream: bool) -> Value {
    let mut body = json!({"model":alias,"prompt":prompt,"max_tokens":max_tokens,
        "temperature":0,"top_p":1,"stream":stream});
    if stream {
        body["stream_options"] = json!({"include_usage":true});
    }
    body
}

#[derive(Clone)]
struct Oracle {
    prompt: String,
    prompt_tokens: usize,
    ids: Vec<u32>,
    stopped: bool,
}

impl Oracle {
    fn compute(
        source: &Arc<LoadedQwenModel>,
        tokenizer: &QwenTokenizer,
        prompt: &str,
    ) -> Proof<Self> {
        let tokens = tokenizer
            .prompt_tokens(prompt, false)
            .map_err(|e| e.to_string())?;
        let mut cpu = QwenDenseCpu::from_shared(Arc::clone(source)).map_err(|e| e.to_string())?;
        let mut logits = cpu
            .prefill_transaction(&tokens)
            .map_err(|e| e.to_string())?;
        let mut ids = Vec::new();
        let mut stopped = false;
        for step in 0..8 {
            if logits.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite independent CPU logits".into());
            }
            let id = (0..logits.len())
                .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
                .ok_or("empty independent CPU vocabulary")? as u32;
            if id == tokenizer.eos() {
                stopped = true;
                break;
            }
            ids.push(id);
            if step != 7 {
                logits = cpu.forward(id).map_err(|e| e.to_string())?;
            }
        }
        Ok(Self {
            prompt: prompt.into(),
            prompt_tokens: tokens.len(),
            ids,
            stopped,
        })
    }

    fn expected(
        &self,
        tokenizer: &QwenTokenizer,
        limit: usize,
    ) -> Proof<(String, usize, &'static str)> {
        let count = self.ids.len().min(limit);
        let text = tokenizer
            .decode(&self.ids[..count], true)
            .map_err(|e| e.to_string())?;
        let reason = if self.stopped && self.ids.len() < limit {
            "stop"
        } else {
            "length"
        };
        Ok((text, count, reason))
    }
}

fn find_source_eos(source: &Arc<LoadedQwenModel>, tokenizer: &QwenTokenizer) -> Proof<Oracle> {
    // Same bounded source-real search as the internal physical worker proof.
    for prefix in [
        "<|im_start|>user\nSay only OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nOK",
        "<|im_start|>user\nWhat is 1+1? Give only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n2",
        "<|im_start|>user\nReply with nothing.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>user\nSay only Done.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nDone",
    ] {
        let oracle = Oracle::compute(source, tokenizer, prefix)?;
        if oracle.stopped {
            return Ok(oracle);
        }
    }
    Err("P12_HTTP_TRAINED_EOS_NOT_QUALIFIED: no original CPU-selected EOS in four prefixes with eight greedy selections each".into())
}

fn validate_usage(usage: &Value, prompt: usize, completion: usize) -> Proof<()> {
    if usage["prompt_tokens"] != prompt
        || usage["completion_tokens"] != completion
        || usage["total_tokens"] != prompt + completion
    {
        return Err(format!(
            "source token usage differs: {usage}, expected {prompt}+{completion}"
        ));
    }
    Ok(())
}

fn json_completion(
    mut response: Response,
    alias: &str,
    expected: &(String, usize, &'static str),
    prompt: usize,
) -> Proof<()> {
    let body = response.body()?;
    if response.status != 200 {
        return Err(format!(
            "completion HTTP{} {}",
            response.status,
            String::from_utf8_lossy(&body)
        ));
    }
    let value: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
    if value["model"] != alias
        || value["choices"].as_array().is_none_or(|v| v.len() != 1)
        || value["choices"][0]["text"] != expected.0
        || value["choices"][0]["finish_reason"] != expected.2
    {
        return Err(format!(
            "JSON completion differs from independent source: {value}"
        ));
    }
    validate_usage(&value["usage"], prompt, expected.1)
}

#[derive(Default)]
struct Frames {
    bytes: Vec<u8>,
}
impl Frames {
    fn feed(&mut self, data: &[u8]) -> Proof<Vec<String>> {
        self.bytes.extend_from_slice(data);
        if self.bytes.len() > BODY_LIMIT {
            return Err("SSE pending frame exceeds bound".into());
        }
        let mut frames = Vec::new();
        while let Some(end) = self.bytes.windows(2).position(|w| w == b"\n\n") {
            let block = self.bytes.drain(..end + 2).collect::<Vec<_>>();
            let block = std::str::from_utf8(&block).map_err(|e| e.to_string())?;
            let data = block
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
                .collect::<Vec<_>>()
                .join("\n");
            if !data.is_empty() {
                frames.push(data);
            }
        }
        Ok(frames)
    }
}

fn sse_completion(
    mut response: Response,
    alias: &str,
    expected: &(String, usize, &'static str),
    prompt: usize,
) -> Proof<()> {
    if response.status != 200 || !response.chunked {
        return Err("SSE did not use HTTP200 chunked transport".into());
    }
    let mut frames = Frames::default();
    let mut text = String::new();
    let mut terminal = 0;
    let mut usage = 0;
    let mut done = 0;
    let mut id = None;
    let mut total = 0;
    while let Some(chunk) = response.chunk()? {
        total += chunk.len();
        if total > BODY_LIMIT {
            return Err("SSE response exceeds bound".into());
        }
        for frame in frames.feed(&chunk)? {
            if frame == "[DONE]" {
                if terminal != 1 || usage != 1 {
                    return Err("SSE DONE preceded terminal choice/usage".into());
                }
                done += 1;
                continue;
            }
            if done != 0 {
                return Err("SSE data followed terminal DONE".into());
            }
            let event: Value = serde_json::from_str(&frame).map_err(|e| e.to_string())?;
            if event.get("error").is_some() || event["model"] != alias {
                return Err(format!("SSE error/foreign alias: {event}"));
            }
            let current = event["id"].as_str().ok_or("SSE id missing")?;
            if id.as_deref().is_some_and(|id| id != current) {
                return Err("SSE response changed identity".into());
            }
            id.get_or_insert_with(|| current.to_owned());
            if !event["usage"].is_null() {
                if terminal != 1 {
                    return Err("SSE usage preceded terminal choice".into());
                }
                usage += 1;
                validate_usage(&event["usage"], prompt, expected.1)?;
            }
            for choice in event["choices"].as_array().ok_or("SSE choices missing")? {
                if let Some(reason) = choice["finish_reason"].as_str() {
                    terminal += 1;
                    if choice["text"] != "" {
                        return Err("SSE terminal choice repeated text".into());
                    }
                    if reason != expected.2 {
                        return Err(format!("wrong SSE finish reason: {event}"));
                    }
                } else {
                    if terminal != 0 {
                        return Err("SSE text followed terminal choice".into());
                    }
                    text.push_str(choice["text"].as_str().ok_or("SSE text missing")?);
                }
            }
        }
    }
    if !frames.bytes.is_empty() || text != expected.0 || terminal != 1 || usage != 1 || done != 1 {
        return Err(format!(
            "SSE source text/terminal/usage differs: text={text:?}, expected={:?}, terminal={terminal}, usage={usage}, done={done}",
            expected.0
        ));
    }
    Ok(())
}

fn disconnect_after_first_text(
    response: Response,
    alias: &str,
    expected_prefix: &str,
) -> Proof<()> {
    disconnect_text_when(response, alias, expected_prefix, || Ok(()))
}

fn disconnect_text_when(
    mut response: Response,
    alias: &str,
    expected_prefix: &str,
    before_disconnect: impl FnOnce() -> Proof<()>,
) -> Proof<()> {
    if response.status != 200 || !response.chunked {
        return Err("canceled request lacked real SSE response".into());
    }
    let mut frames = Frames::default();
    while let Some(chunk) = response.chunk()? {
        for frame in frames.feed(&chunk)? {
            if frame == "[DONE]" {
                return Err("cancel target finished before authentic text disconnect".into());
            }
            let event: Value = serde_json::from_str(&frame).map_err(|e| e.to_string())?;
            if event.get("error").is_some() || event["model"] != alias {
                return Err(format!("cancel target SSE error: {event}"));
            }
            if let Some(text) = event["choices"][0]["text"].as_str()
                && !text.is_empty()
            {
                if !expected_prefix.starts_with(text) {
                    return Err("cancel target text differs from CPU prefix".into());
                }
                before_disconnect()?;
                response.disconnect();
                return Ok(());
            }
        }
    }
    Err("cancel target ended without source text".into())
}

fn counter(value: &Value, key: &str) -> Proof<u64> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("health counter {key} missing: {value}"))
}

/// Queue the pair during observable real GPU prefill. Active=4 admits both
/// together after the gate completes, without adding a server batching delay.
fn queued_pair(server: &OwnedServer, first: &Value, second: &Value) -> Proof<(Response, Response)> {
    let before = server.health()?["admission"].clone();
    let attempts = counter(&before, "gpu_step_attempts")?;
    let tokens = counter(&before, "completed_model_token_steps")?;
    let gate_body = payload(
        &server.aliases[0],
        &"A real GPU prefill gate. ".repeat(12),
        8,
        true,
    );
    let gate = open_request(server.address, "POST", "/v1/completions", Some(&gate_body))?;
    server.wait_health(|health| {
        health["active_sessions"].as_u64().is_some_and(|n| n >= 1)
            && health["gpu_step_attempts"]
                .as_u64()
                .is_some_and(|n| n > attempts)
            && health["completed_model_token_steps"].as_u64() == Some(tokens)
    })?;
    // Open/write both before reading either response; JSON waits for Done.
    let first = open_request(server.address, "POST", "/v1/completions", Some(first))?;
    let second = open_request(server.address, "POST", "/v1/completions", Some(second))?;
    let _ = gate.shutdown(Shutdown::Both);
    drop(gate);
    Ok((Response::read(first)?, Response::read(second)?))
}

/// Real queued requests, all written before reading any response. Cancel the
/// observable GPU quantum's owner so active8 can admit all eight queued rows.
fn queued_group(server: &OwnedServer, bodies: &[Value]) -> Proof<Vec<Response>> {
    if !matches!(bodies.len(), 4 | 8) {
        return Err("wide HTTP group requires four or eight real requests".into());
    }
    let before = server.health()?["admission"].clone();
    let attempts = counter(&before, "gpu_step_attempts")?;
    let tokens = counter(&before, "completed_model_token_steps")?;
    let gate_body = payload(
        &server.aliases[0],
        &"A real GPU prefill gate. ".repeat(12),
        8,
        true,
    );
    let gate = open_request(server.address, "POST", "/v1/completions", Some(&gate_body))?;
    server.wait_health(|health| {
        health["active_sessions"] == 1
            && health["gpu_step_attempts"]
                .as_u64()
                .is_some_and(|n| n > attempts)
            && health["completed_model_token_steps"].as_u64() == Some(tokens)
    })?;
    let streams = bodies
        .iter()
        .map(|body| open_request(server.address, "POST", "/v1/completions", Some(body)))
        .collect::<Proof<Vec<_>>>()?;
    server.wait_health_for(Duration::from_secs(5), |health| {
        health["queued_requests"] == bodies.len()
            && health["completed_model_token_steps"].as_u64() == Some(tokens)
    })?;
    let _ = gate.shutdown(Shutdown::Both);
    drop(gate);
    streams.into_iter().map(Response::read).collect()
}

fn band_counter(health: &Value, width: usize) -> Proof<u64> {
    counter(&health["gpu_fused_batches_by_rows"], &width.to_string())
}

fn physical_accounting(health: &Value) -> Proof<()> {
    let b2 = band_counter(health, 2)?;
    let b4 = band_counter(health, 4)?;
    let b8 = band_counter(health, 8)?;
    let batches = b2 + b4 + b8;
    if counter(health, "failed_gpu_quanta")? != 0
        || health["active_sessions"] != 0
        || health["queued_requests"] != 0
        || counter(health, "completed_forward_kernel_dispatches")? == 0
        || counter(health, "native_projection_dispatches_completed")? == 0
        || counter(health, "gpu_fused_batches")? != batches
        || counter(health, "gpu_fused_rows_completed")? != 2 * b2 + 4 * b4 + 8 * b8
        || counter(health, "gpu_fused_state_only_batches")?
            + counter(health, "gpu_fused_greedy_batches")?
            != batches
        || counter(health, "gpu_fused_state_only_rows")?
            > counter(health, "gpu_fused_rows_completed")?
        || counter(health, "native_sampler_dispatches_completed")?
            != 4 * counter(health, "gpu_fused_greedy_batches")?
        || counter(health, "native_admission_refusals")? != 0
    {
        return Err(format!("wide HTTP physical accounting differs: {health}"));
    }
    Ok(())
}

fn proof_cases(
    server: &OwnedServer,
    oracle: &Oracle,
    eos: &Oracle,
    tokenizer: &QwenTokenizer,
) -> Proof<()> {
    let expected4 = oracle.expected(tokenizer, 4)?;
    let expected8 = oracle.expected(tokenizer, 8)?;
    let initial = server.health()?["admission"].clone();
    if initial["batch_mode"] != "fused" || initial["native_batch_rows_max"] != 2 {
        return Err("HTTP did not advertise only native CUDA B2".into());
    }
    let (json, sse) = queued_pair(
        server,
        &payload(&server.aliases[0], &oracle.prompt, 4, false),
        &payload(&server.aliases[1], &oracle.prompt, 8, true),
    )?;
    json_completion(json, &server.aliases[0], &expected4, oracle.prompt_tokens)?;
    sse_completion(sse, &server.aliases[1], &expected8, oracle.prompt_tokens)?;
    let first = server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    // Only the two differently named aliases could form a native pair here.
    // The native engine rejects foreign ResidentModel/source allocations.
    if counter(&first, "gpu_fused_batches")? <= counter(&initial, "gpu_fused_batches")? {
        return Err("actual source-alias HTTP pair never executed a native batch".into());
    }
    let eos_expected = eos.expected(tokenizer, 8)?;
    if !eos.stopped || eos_expected.2 != "stop" || eos_expected.1 >= 8 {
        return Err("actual bounded original EOS oracle was not established".into());
    }
    let (json_eos, sse_eos) = queued_pair(
        server,
        &payload(&server.aliases[0], &eos.prompt, 8, false),
        &payload(&server.aliases[1], &eos.prompt, 8, true),
    )?;
    json_completion(
        json_eos,
        &server.aliases[0],
        &eos_expected,
        eos.prompt_tokens,
    )?;
    sse_completion(
        sse_eos,
        &server.aliases[1],
        &eos_expected,
        eos.prompt_tokens,
    )?;
    let after_eos =
        server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    // This external check proves actual native execution and trained stop on
    // the wire. Prompt-tail phases/all42 domains are checked internally.
    if counter(&after_eos, "gpu_fused_batches")? <= counter(&first, "gpu_fused_batches")? {
        return Err("actual trained-EOS HTTP pair never executed a native batch".into());
    }
    let cancelled_before = counter(&after_eos, "cancelled_requests")?;
    let (cancel, survivor) = queued_pair(
        server,
        &payload(&server.aliases[0], &oracle.prompt, 8, true),
        &payload(&server.aliases[1], &oracle.prompt, 8, true),
    )?;
    let survivor_alias = server.aliases[1].clone();
    let survivor_expected = expected8.clone();
    let prompt_tokens = oracle.prompt_tokens;
    let survived = thread::spawn(move || {
        sse_completion(survivor, &survivor_alias, &survivor_expected, prompt_tokens)
    });
    let disconnected = disconnect_after_first_text(cancel, &server.aliases[0], &expected8.0);
    let survivor = survived
        .join()
        .map_err(|_| "HTTP survivor reader panicked")?;
    disconnected?;
    survivor?;
    server.wait_health(|h| {
        h["active_sessions"] == 0
            && h["queued_requests"] == 0
            && h["cancelled_requests"]
                .as_u64()
                .is_some_and(|n| n >= cancelled_before + 2)
    })?;

    let replay_body = payload(&server.aliases[0], &oracle.prompt, 8, true);
    let replay = Response::read(open_request(
        server.address,
        "POST",
        "/v1/completions",
        Some(&replay_body),
    )?)?;
    sse_completion(replay, &server.aliases[0], &expected8, oracle.prompt_tokens)?;
    let final_health =
        server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    if counter(&final_health, "failed_gpu_quanta")? != 0
        || counter(&final_health, "completed_forward_kernel_dispatches")? == 0
        || counter(&final_health, "native_projection_dispatches_completed")? == 0
        || counter(&final_health, "native_sampler_dispatches_completed")? == 0
        || counter(&final_health, "gpu_fused_rows_completed")?
            != 2 * counter(&final_health, "gpu_fused_batches")?
        || counter(&final_health, "gpu_fused_state_only_batches")?
            + counter(&final_health, "gpu_fused_greedy_batches")?
            != counter(&final_health, "gpu_fused_batches")?
        || counter(&final_health, "gpu_fused_state_only_rows")?
            != 2 * counter(&final_health, "gpu_fused_state_only_batches")?
        || counter(&final_health, "native_sampler_dispatches_completed")?
            != 4 * counter(&final_health, "gpu_fused_greedy_batches")?
    {
        return Err(format!(
            "HTTP physical dispatch/error accounting differs: {final_health}"
        ));
    }
    println!(
        "P12_HTTP_CUDA_NATIVE_B2_PASS source_aliases=shared JSON_SSE=CPU_exact trained_EOS_JSON_SSE=stop_exact EOS_excluded_usage=true terminal_usage_DONE=once disconnect_survivor_replay=exact health={final_health}"
    );
    Ok(())
}

fn validate_group(
    responses: Vec<Response>,
    bodies: &[Value],
    oracle: &Oracle,
    tokenizer: &QwenTokenizer,
) -> Proof<()> {
    if responses.len() != bodies.len() {
        return Err("HTTP group lost a request/response".into());
    }
    for (response, body) in responses.into_iter().zip(bodies) {
        let alias = body["model"].as_str().ok_or("group alias missing")?;
        let maximum = body["max_tokens"].as_u64().ok_or("group maximum missing")? as usize;
        let expected = oracle.expected(tokenizer, maximum)?;
        if body["stream"] == true {
            sse_completion(response, alias, &expected, oracle.prompt_tokens)?;
        } else {
            json_completion(response, alias, &expected, oracle.prompt_tokens)?;
        }
    }
    Ok(())
}

fn wide_proof_cases(
    server: &OwnedServer,
    width: usize,
    oracle: &Oracle,
    eos: &Oracle,
    tokenizer: &QwenTokenizer,
) -> Proof<()> {
    let initial = server.health()?["admission"].clone();
    if initial["batch_mode"] != "fused" || initial["native_batch_rows_max"] != width {
        return Err(format!("HTTP did not advertise native CUDA B{width}"));
    }
    if oracle.ids.len() != 8 || oracle.stopped {
        return Err("wide retirement proof requires eight original nonterminal tokens".into());
    }
    let bodies = (0..width)
        .map(|index| {
            payload(
                &server.aliases[index % 2],
                &oracle.prompt,
                (index + 2).min(8),
                index % 2 == 1,
            )
        })
        .collect::<Vec<_>>();
    let responses = queued_group(server, &bodies)?;
    validate_group(responses, &bodies, oracle, tokenizer)?;
    let lengths = server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    if band_counter(&lengths, width)? < band_counter(&initial, width)? + 2
        || band_counter(&lengths, 2)? <= band_counter(&initial, 2)?
        || (width == 8 && band_counter(&lengths, 4)? <= band_counter(&initial, 4)?)
    {
        return Err(format!(
            "actual wide HTTP group lacks full-width and odd-survivor receipts: {lengths}"
        ));
    }
    physical_accounting(&lengths)?;

    let eos_expected = eos.expected(tokenizer, 8)?;
    if !eos.stopped || eos_expected.2 != "stop" || eos_expected.1 >= 8 {
        return Err("wide HTTP proof lacks original trained-EOS evidence".into());
    }
    let eos_bodies = (0..width)
        .map(|index| payload(&server.aliases[index % 2], &eos.prompt, 8, index % 2 == 0))
        .collect::<Vec<_>>();
    let responses = queued_group(server, &eos_bodies)?;
    validate_group(responses, &eos_bodies, eos, tokenizer)?;
    let after_eos =
        server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    if band_counter(&after_eos, width)? <= band_counter(&lengths, width)? {
        return Err("trained-EOS group did not physically execute its native width".into());
    }
    physical_accounting(&after_eos)?;

    let expected8 = oracle.expected(tokenizer, 8)?;
    let cancel_bodies = (0..width)
        .map(|index| payload(&server.aliases[index % 2], &oracle.prompt, 8, true))
        .collect::<Vec<_>>();
    let mut responses = queued_group(server, &cancel_bodies)?;
    let cancel = responses.remove(0);
    let mut survivors = Vec::new();
    for (index, response) in responses.into_iter().enumerate() {
        let alias = server.aliases[(index + 1) % 2].clone();
        let expected = expected8.clone();
        let prompt_tokens = oracle.prompt_tokens;
        survivors.push(thread::spawn(move || {
            sse_completion(response, &alias, &expected, prompt_tokens)
        }));
    }
    let disconnected = disconnect_text_when(cancel, &server.aliases[0], &expected8.0, || {
        let live = server.health()?["admission"].clone();
        let accepted = counter(&live, "accepted_output_tokens")?
            - counter(&after_eos, "accepted_output_tokens")?;
        if live["active_sessions"] != width
            || live["queued_requests"] != 0
            || band_counter(&live, width)? <= band_counter(&after_eos, width)?
            || accepted >= (8 * width) as u64
        {
            return Err(format!(
                "cancel target was not demonstrably live in the real wide cohort: {live}"
            ));
        }
        Ok(())
    });
    let mut survivor_error = None;
    for survivor in survivors {
        let result = survivor
            .join()
            .map_err(|_| "wide HTTP survivor reader panicked".to_owned())
            .and_then(|result| result);
        if let Err(error) = result {
            survivor_error.get_or_insert(error);
        }
    }
    disconnected?;
    if let Some(error) = survivor_error {
        return Err(error);
    }
    let cancelled = counter(&after_eos, "cancelled_requests")?;
    let after_cancel = server.wait_health(|h| {
        h["active_sessions"] == 0
            && h["queued_requests"] == 0
            && h["cancelled_requests"]
                .as_u64()
                .is_some_and(|count| count >= cancelled + 2)
    })?;
    if band_counter(&after_cancel, width)? <= band_counter(&after_eos, width)? {
        return Err("live canceled HTTP group never executed its native width".into());
    }
    let replay_body = payload(&server.aliases[0], &oracle.prompt, 8, true);
    let replay = Response::read(open_request(
        server.address,
        "POST",
        "/v1/completions",
        Some(&replay_body),
    )?)?;
    sse_completion(replay, &server.aliases[0], &expected8, oracle.prompt_tokens)?;
    let final_health =
        server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    physical_accounting(&final_health)?;
    println!(
        "P12_HTTP_CUDA_NATIVE_WIDE_PASS rows={width} actual_queued_aliases=shared \
        JSON_SSE_text_usage=CPU_exact trained_EOS_JSON_SSE=stop_exact terminal_usage_DONE=once \
        live_disconnect_survivors_replay=exact weighted_native_rows_and_sampler=exact \
        all42_state=internal_gate vLLM_performance=not_qualified health={final_health}"
    );
    Ok(())
}

fn qualify(name: &str) -> Proof<()> {
    let _serial = GPU_HTTP_LOCK
        .lock()
        .map_err(|_| "physical HTTP test lock poisoned")?;
    if cfg!(debug_assertions) {
        return Err("physical HTTP proof requires cargo test --release".into());
    }
    let directory: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let model_path = directory.join("model.gguf");
    for file in [
        &model_path,
        &directory.join("config.json"),
        &directory.join("tokenizer.json"),
    ] {
        if !file.is_file() {
            return Err(format!(
                "approved original source missing: {}",
                file.display()
            ));
        }
    }
    let source = LoadedQwenModel::open(&model_path, &directory.join("config.json"))
        .map_err(|e| e.to_string())?;
    let tokenizer =
        QwenTokenizer::from_file(&directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut chosen = None;
    for prompt in [
        "The sequence is 1, 2, 3,",
        "Continue counting: one, two, three,",
        "A simple example is",
    ] {
        let oracle = Oracle::compute(&source, &tokenizer, prompt)?;
        if oracle.ids.len() >= 4 && !oracle.expected(&tokenizer, 4)?.0.is_empty() {
            chosen = Some(oracle);
            break;
        }
    }
    let oracle =
        chosen.ok_or("no bounded original prompt supplied four nonterminal source tokens")?;
    let eos = find_source_eos(&source, &tokenizer)?;
    let server = OwnedServer::start(&model_path)?;
    proof_cases(&server, &oracle, &eos, &tokenizer).map_err(|error| {
        format!(
            "{name}: {error}\nowned HTTP server stderr:\n{}",
            server.diagnostics()
        )
    })
}

fn qualify_wide(name: &str) -> Proof<()> {
    let _serial = GPU_HTTP_LOCK
        .lock()
        .map_err(|_| "physical HTTP test lock poisoned")?;
    if cfg!(debug_assertions) {
        return Err("physical HTTP proof requires cargo test --release".into());
    }
    let directory: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let model_path = directory.join("model.gguf");
    for file in [
        &model_path,
        &directory.join("config.json"),
        &directory.join("tokenizer.json"),
    ] {
        if !file.is_file() {
            return Err(format!(
                "approved original source missing: {}",
                file.display()
            ));
        }
    }
    let source = LoadedQwenModel::open(&model_path, &directory.join("config.json"))
        .map_err(|e| e.to_string())?;
    let tokenizer =
        QwenTokenizer::from_file(&directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut chosen = None;
    for prompt in [
        "The sequence is 1, 2, 3,",
        "Continue counting: one, two, three,",
        "A simple example is",
    ] {
        let oracle = Oracle::compute(&source, &tokenizer, prompt)?;
        if oracle.ids.len() == 8 && !oracle.stopped && !oracle.expected(&tokenizer, 8)?.0.is_empty()
        {
            chosen = Some(oracle);
            break;
        }
    }
    let oracle =
        chosen.ok_or("no bounded original prompt supplied eight nonterminal source tokens")?;
    let eos = find_source_eos(&source, &tokenizer)?;
    for width in [4, 8] {
        let server = OwnedServer::start_rows(&model_path, width)?;
        wide_proof_cases(&server, width, &oracle, &eos, &tokenizer).map_err(|error| {
            format!(
                "{name} B{width}: {error}\nowned HTTP server stderr:\n{}",
                server.diagnostics()
            )
        })?;
    }
    Ok(())
}

#[test]
#[ignore = "release physical CUDA HTTP; approved original .8B source; run serially"]
fn original_qwen35_08b_native_b2_http_json_sse_disconnect_and_replay() -> Proof<()> {
    qualify("qwen35-0.8b")
}

#[test]
#[ignore = "release physical CUDA HTTP; approved original 2B source; run serially"]
fn original_qwen35_2b_native_b2_http_json_sse_disconnect_and_replay() -> Proof<()> {
    qualify("qwen35-2b")
}

#[test]
#[ignore = "release physical CUDA B4/B8 HTTP; approved original .8B source; run serially"]
fn original_qwen35_08b_native_wide_http_json_sse_disconnect_and_replay() -> Proof<()> {
    qualify_wide("qwen35-0.8b")
}

#[test]
#[ignore = "release physical CUDA B4/B8 HTTP; approved original 2B source; run serially"]
fn original_qwen35_2b_native_wide_http_json_sse_disconnect_and_replay() -> Proof<()> {
    qualify_wide("qwen35-2b")
}

fn observed_sse_completion(
    mut response: Response,
    alias: &str,
    expected: &(String, usize, &'static str),
    expected_ids: &[u32],
    prompt: usize,
    wire_id: &str,
    first: Option<std::sync::mpsc::SyncSender<()>>,
) -> Proof<Vec<Value>> {
    if response.status != 200 || !response.chunked {
        return Err("observed SSE did not use HTTP200 chunked transport".into());
    }
    let mut frames = Frames::default();
    let (mut text, mut ids, mut provider_id) = (String::new(), Vec::<u32>::new(), None::<String>);
    let (mut terminal, mut usage, mut done, mut total) = (0, 0, 0, 0);
    let mut first = first;
    let mut metadata = None;
    while let Some(chunk) = response.chunk()? {
        total += chunk.len();
        if total > BODY_LIMIT {
            return Err("observed SSE response exceeds bound".into());
        }
        for frame in frames.feed(&chunk)? {
            if frame == "[DONE]" {
                if terminal != 1 || usage != 1 || done != 0 {
                    return Err("observed SSE DONE order/count".into());
                }
                done += 1;
                continue;
            }
            if done != 0 {
                return Err("observed SSE data followed DONE".into());
            }
            let event: Value = serde_json::from_str(&frame).map_err(|e| e.to_string())?;
            if event.get("error").is_some() || event["model"] != alias {
                return Err(format!("observed SSE foreign/error event: {event}"));
            }
            let current = event["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .ok_or("observed SSE id missing")?;
            if provider_id.as_deref().is_some_and(|id| id != current) {
                return Err("observed SSE response changed identity".into());
            }
            provider_id.get_or_insert_with(|| current.into());
            if !event["usage"].is_null() {
                if terminal != 1 || usage != 0 {
                    return Err("observed SSE duplicate/early usage".into());
                }
                usage += 1;
                validate_usage(&event["usage"], prompt, expected.1)?;
                metadata = Some(event["zerogravity_token_metrics"].clone());
            }
            for choice in event["choices"]
                .as_array()
                .ok_or("observed SSE choices missing")?
            {
                if choice["index"].as_u64() != Some(0) {
                    return Err("observed SSE foreign choice index".into());
                }
                if let Some(reason) = choice["finish_reason"].as_str() {
                    terminal += 1;
                    if choice["text"] != ""
                        || reason != expected.2
                        || !choice["token_ids"].is_null()
                    {
                        return Err("observed SSE terminal differs/repeats IDs".into());
                    }
                } else {
                    if terminal != 0 {
                        return Err("observed SSE text after terminal".into());
                    }
                    text.push_str(choice["text"].as_str().ok_or("observed SSE text missing")?);
                    let token_ids = choice["token_ids"]
                        .as_array()
                        .filter(|ids| ids.len() == 1)
                        .ok_or("individual actual token ID required")?;
                    let token = token_ids[0]
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or("observed SSE invalid token ID")?;
                    ids.push(token);
                    if ids.len() > 512 {
                        return Err("observed SSE token bound".into());
                    }
                    if let Some(notify) = first.take() {
                        notify.send(()).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
    }
    if !frames.bytes.is_empty()
        || text != expected.0
        || ids != expected_ids
        || terminal != 1
        || usage != 1
        || done != 1
    {
        return Err(format!(
            "observed SSE text/IDs/terminal mismatch: ids={ids:?}, expected={expected_ids:?}, terminal={terminal} usage={usage} done={done}"
        ));
    }
    let metadata = metadata.ok_or("observed terminal metadata missing")?;
    if metadata["schema"] != "zerogravity.serving.request-token-metrics.v1"
        || metadata["wire_request_id"] != wire_id
        || metadata["provider_response_id"].as_str() != provider_id.as_deref()
        || metadata["complete"] != true
        || metadata["completion_tokens"] != ids.len()
        || metadata["clock"] != "request_monotonic_seconds"
    {
        return Err(format!("observed metadata binding differs: {metadata}"));
    }
    let records = metadata["tokens"]
        .as_array()
        .ok_or("observed token records missing")?;
    if records.len() != ids.len() {
        return Err("observed token/usage count differs".into());
    }
    let (mut previous_commit, mut previous_write) = (0.0, 0.0);
    for (index, (record, id)) in records.iter().zip(&ids).enumerate() {
        let commit = record["commit_observed_at_seconds"]
            .as_f64()
            .ok_or("commit clock missing")?;
        let received = record["output_received_at_seconds"]
            .as_f64()
            .ok_or("receive clock missing")?;
        let written = record["http_write_completed_at_seconds"]
            .as_f64()
            .ok_or("write clock missing")?;
        if record["index"] != index
            || record["token_id"] != *id
            || !commit.is_finite()
            || !received.is_finite()
            || !written.is_finite()
            || commit < previous_commit
            || received < commit
            || received < previous_write
            || written < received
            || record["gpu"]["producer_frontier"] != prompt + index
        {
            return Err(format!(
                "observed record order/frontier/clock differs: {record}"
            ));
        }
        previous_commit = commit;
        previous_write = written;
    }
    Ok(records.clone())
}

fn qualify_unequal_arrival_metrics(name: &str) -> Proof<()> {
    let _serial = GPU_HTTP_LOCK
        .lock()
        .map_err(|_| "physical HTTP lock poisoned")?;
    if cfg!(debug_assertions) {
        return Err("physical HTTP proof requires --release".into());
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let model_path = directory.join("model.gguf");
    let source = LoadedQwenModel::open(&model_path, &directory.join("config.json"))
        .map_err(|e| e.to_string())?;
    let tokenizer =
        QwenTokenizer::from_file(&directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let short = Oracle::compute(&source, &tokenizer, "a")?;
    if short.ids.len() != 8 || short.stopped {
        return Err("short unequal-arrival source prompt needs eight non-EOS outputs".into());
    }
    let mut chosen = None;
    for prompt in [
        "The sequence is 1, 2, 3, 4, 5, 6, 7, 8,",
        "Continue counting the following sequence of whole numbers: one, two, three, four, five,",
    ] {
        let oracle = Oracle::compute(&source, &tokenizer, prompt)?;
        if oracle.prompt_tokens > short.prompt_tokens + 8
            && oracle.ids.len() == 8
            && !oracle.stopped
        {
            chosen = Some(oracle);
            break;
        }
    }
    let long=chosen.ok_or("long unequal-arrival source prompt needs distinct token frontier and eight non-EOS outputs")?;
    let server = OwnedServer::start_configured(&model_path, 2, true)?;
    if server.health()?["per_request_metrics_enabled"] != true {
        return Err("actual metrics server flag not enabled".into());
    }
    let mut first_body = payload(&server.aliases[0], &long.prompt, 8, true);
    first_body["return_token_ids"] = json!(true);
    let first_response = Response::read(open_bound_request(
        server.address,
        "POST",
        "/v1/completions",
        Some(&first_body),
        Some("unequal:first"),
    )?)?;
    let first_expected = long.expected(&tokenizer, 8)?;
    let first_alias = server.aliases[0].clone();
    let first_ids = long.ids.clone();
    let first_prompt = long.prompt_tokens;
    let (notify, arrival) = std::sync::mpsc::sync_channel(1);
    let first = thread::spawn(move || {
        observed_sse_completion(
            first_response,
            &first_alias,
            &first_expected,
            &first_ids,
            first_prompt,
            "unequal:first",
            Some(notify),
        )
    });
    // B is offered only after an actual ID from A is received; both sockets
    // are continuously drained while B prefills and the cohort progresses.
    arrival
        .recv_timeout(REQUEST_LIMIT)
        .map_err(|e| format!("first actual token arrival: {e}"))?;
    let mut second_body = payload(&server.aliases[1], &short.prompt, 8, true);
    second_body["return_token_ids"] = json!(true);
    let second_response = Response::read(open_bound_request(
        server.address,
        "POST",
        "/v1/completions",
        Some(&second_body),
        Some("unequal:second"),
    )?)?;
    let second_records = observed_sse_completion(
        second_response,
        &server.aliases[1],
        &short.expected(&tokenizer, 8)?,
        &short.ids,
        short.prompt_tokens,
        "unequal:second",
        None,
    )?;
    let first_records = first
        .join()
        .map_err(|_| "first observed SSE reader panicked")??;
    let paired = first_records.iter().any(|a| {
        second_records.iter().any(|b| {
            let ca = &a["gpu"]["native_cohort"];
            let cb = &b["gpu"]["native_cohort"];
            ca["publication_sequence"]
                .as_u64()
                .is_some_and(|sequence| sequence > 0)
                && ca == cb
                && ca["width"] == 2
                && a["gpu"]["producer_frontier"] != b["gpu"]["producer_frontier"]
        })
    });
    if !paired {
        return Err(format!(
            "actual HTTP unequal arrival never shared an observed native cohort: first={first_records:?},second={second_records:?}"
        ));
    }
    let health = server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
    physical_accounting(&health)?;
    println!(
        "P12_UNEQUAL_ARRIVAL_HTTP_PASS source={name} different_source_prompt_lengths=true B_arrival_after_A_actual_ID=true both_sockets_drained=true native_membership_frontiers=observed CPU_ID_text_usage=exact terminal_usage_DONE=once health={health}"
    );
    Ok(())
}

#[test]
#[ignore = "release physical CUDA unequal-arrival metrics HTTP; approved .8B source; serial"]
fn original_qwen35_08b_native_unequal_arrival_token_observations_http() -> Proof<()> {
    qualify_unequal_arrival_metrics("qwen35-0.8b")
}
#[test]
#[ignore = "release physical CUDA unequal-arrival metrics HTTP; approved 2B source; serial"]
fn original_qwen35_2b_native_unequal_arrival_token_observations_http() -> Proof<()> {
    qualify_unequal_arrival_metrics("qwen35-2b")
}

#[path = "serve_gpu/prefix.rs"]
mod prefix;
