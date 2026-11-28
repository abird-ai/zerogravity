//! Process-owned operational sink. Only cold/host observations enter here.
#[cfg(any(feature = "tui", feature = "web", feature = "serve"))]
pub(crate) mod ring;
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use zerogravity::runtime::text::FailureDetails;

const SEGMENT_BYTES: u64 = 4 * 1024 * 1024;
const SEGMENTS: u64 = 4;
const RETAIN_CLOSED: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code, reason = "entrypoints are independently feature-gated")]
pub(crate) enum Mode {
    Command,
    Interactive,
    Web,
    Service,
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

pub(crate) fn clock(ms: u64) -> String {
    let seconds = ms / 1000 % 86400;
    format!(
        "{:02}:{:02}:{:02}.{:03}Z",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60,
        ms % 1000
    )
}

/// Local bounded diagnostic, separate from the safe wire failure projection.
/// Arbitrary input and provider URLs are not log correlation identities.
pub(crate) fn bounded_message(message: &str) -> String {
    let lower = message.to_ascii_lowercase();
    // Do not try to reconstruct arbitrary quoted credentials or header
    // values. Suppress the entire diagnostic if it looks credential-bearing.
    if [
        "bearer ",
        "authorization:",
        "authorization=",
        "api_key",
        "api-key",
        "access_token",
        "password",
        "hf_",
        "ghp_",
        "github_pat_",
        "sk-",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return "[credential-bearing diagnostic omitted]".into();
    }
    let mut out = String::new();
    for word in message.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        // The original serving readiness URL is public numeric loopback
        // without credentials, query strings or arbitrary path content.
        let loopback = word
            .strip_prefix("http://")
            .and_then(|url| url.strip_suffix("/v1"))
            .and_then(|addr| addr.parse::<std::net::SocketAddr>().ok())
            .is_some_and(|addr| addr.ip().is_loopback());
        let value = if word.contains("://") && !loopback {
            "[redacted URL]"
        } else {
            word
        };
        for c in value.chars() {
            if out.len() + c.len_utf8() > 2048 {
                return out;
            }
            out.push(if c.is_control() { ' ' } else { c });
        }
    }
    out
}

pub(crate) fn causes(error: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut current = Some(error);
    let mut result = Vec::new();
    for _ in 0..8 {
        let Some(error) = current else { break };
        let text = bounded_message(&error.to_string());
        if result.last() != Some(&text) {
            result.push(text);
        }
        current = error.source();
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub(crate) enum Level {
    Info,
    #[serde(rename = "WARN")]
    Warning,
    Error,
}
impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
    fn parse(value: &str) -> Option<Self> {
        match value {
            "info" => Some(Self::Info),
            "warn" => Some(Self::Warning),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Record<'a> {
    pub timestamp_unix_ms: u64,
    pub level: Level,
    pub component: &'a str,
    pub correlation: &'a str,
    pub message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<FailureDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    pub causes: &'a [String],
}

struct Session {
    directory: PathBuf,
    // A held OS file lock identifies a live session, including during rotation.
    _lock: File,
    file: File,
    segment: u64,
    written: u64,
}
enum Destination {
    Stderr,
    Explicit { path: PathBuf, file: File },
    Session(Session),
}
struct Log {
    destination: Destination,
    mode: Mode,
    jsonl: bool,
    minimum: Level,
    started: Instant,
    sequence: u64,
    failure: Option<String>,
}
static LOG: OnceLock<Mutex<Log>> = OnceLock::new();

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
fn private_file(path: &Path, new: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.append(true);
    if new {
        options.create_new(true);
    } else {
        options.create(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    if !new
        && let Ok(metadata) = fs::symlink_metadata(path)
        && !metadata.is_file()
    {
        return Err(invalid("log destination must be a regular file"));
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(invalid("log destination must be a regular file"));
    }
    private_metadata(&metadata)?;
    Ok(file)
}
fn private_metadata(metadata: &fs::Metadata) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(invalid(
                "log destination is not private (require owner-only permissions)",
            ));
        }
    }
    let _ = metadata;
    Ok(())
}
fn state_root() -> io::Result<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|path| path.join(".local/state"))
        })
        .ok_or_else(|| invalid("absolute XDG_STATE_HOME or HOME required for session logs"))?;
    Ok(base.join("abird/zerogravity/logs"))
}
fn private_directory(path: &Path, parents: bool) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(parents);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Err(invalid("log directory must not be a symlink"));
    }
    private_metadata(&metadata)
}
fn segment_path(directory: &Path, segment: u64) -> PathBuf {
    directory.join(format!("events-{segment:016x}.jsonl"))
}
fn segment_name(name: &str) -> bool {
    name.strip_prefix("events-")
        .and_then(|value| value.strip_suffix(".jsonl"))
        .is_some_and(|value| value.len() == 16 && value.bytes().all(|b| b.is_ascii_hexdigit()))
}

impl Session {
    fn open(root: &Path) -> io::Result<Self> {
        private_directory(root, true)?;
        // Serialize directory publication and pruning, never session writes.
        let retention = private_file(&root.join(".retention"), false)?;
        retention.lock()?;
        let stamp = now_ms();
        let mut ordinal = 0u64;
        let directory = loop {
            let path = root.join(format!(
                "session-{stamp:020}-{}-{ordinal}",
                std::process::id()
            ));
            match private_directory(&path, false) {
                Ok(()) => break path,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => ordinal += 1,
                Err(error) => return Err(error),
            }
        };
        let lock = private_file(&directory.join(".active"), true)?;
        lock.lock()?;
        let file = private_file(&segment_path(&directory, 0), true)?;
        let session = Self {
            directory,
            _lock: lock,
            file,
            segment: 0,
            written: 0,
        };
        prune(root, &session.directory)?;
        drop(retention);
        Ok(session)
    }
    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.written > 0 && self.written.saturating_add(bytes.len() as u64) > SEGMENT_BYTES {
            self.file.sync_all()?;
            self.segment = self
                .segment
                .checked_add(1)
                .ok_or_else(|| invalid("log segment overflow"))?;
            self.file = private_file(&segment_path(&self.directory, self.segment), true)?;
            self.written = 0;
            if self.segment >= SEGMENTS {
                fs::remove_file(segment_path(&self.directory, self.segment - SEGMENTS))?;
            }
        }
        self.file.write_all(bytes)?;
        self.written += bytes.len() as u64;
        self.file.flush()
    }
}

/// Only our exact session directories and segment files are candidates.
/// A crashed session releases its lock; a live concurrent host never does.
fn prune(root: &Path, current: &Path) -> io::Result<()> {
    let mut closed = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix("session-") else {
            continue;
        };
        let parts: Vec<_> = rest.split('-').collect();
        if parts.len() != 3
            || parts[0].len() != 20
            || !parts
                .iter()
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
            || entry.path() == current
            || !entry.file_type()?.is_dir()
            || private_metadata(&entry.metadata()?).is_err()
        {
            continue;
        }
        let path = entry.path();
        let lock_path = path.join(".active");
        if !fs::symlink_metadata(&lock_path).is_ok_and(|m| m.is_file()) {
            continue;
        }
        let Ok(lock) = OpenOptions::new().read(true).write(true).open(&lock_path) else {
            continue;
        };
        if lock.try_lock().is_ok() {
            closed.push((path, lock));
        }
    }
    closed.sort_by(|a, b| b.0.cmp(&a.0));
    for (path, _lock) in closed.into_iter().skip(RETAIN_CLOSED) {
        let files = fs::read_dir(&path)?.collect::<io::Result<Vec<_>>>()?;
        if !files.iter().all(|file| {
            file.file_type().is_ok_and(|kind| kind.is_file())
                && file
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name == ".active" || segment_name(name))
        }) {
            continue;
        }
        for file in files {
            if file.file_name() != ".active" {
                fs::remove_file(file.path())?;
            }
        }
        fs::remove_file(path.join(".active"))?;
        fs::remove_dir(path)?;
    }
    Ok(())
}

impl Log {
    fn write(&mut self, record: &Record<'_>) -> io::Result<()> {
        if (self.failure.is_some() && self.mode == Mode::Interactive) || record.level < self.minimum
        {
            return Ok(());
        }
        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        let mut line = if self.jsonl {
            serde_json::to_vec(&serde_json::json!({
                "schema": "zerogravity.operational.v1", "pid": std::process::id(),
                "sequence": sequence, "since_start_ms": self.started.elapsed().as_millis(),
                "event": record,
            }))
            .map_err(io::Error::other)?
        } else {
            let mut text = format!(
                "{} {} {:7} [{}] {}",
                clock(record.timestamp_unix_ms),
                record.level.label(),
                record.component,
                record.correlation,
                record.message
            );
            text.push_str(&format!(
                " unix_ms={} pid={} seq={sequence}",
                record.timestamp_unix_ms,
                std::process::id()
            ));
            if let Some(stage) = record.stage {
                text.push_str(&format!(" stage={stage}"));
            }
            if let Some(elapsed) = record.elapsed_ms {
                text.push_str(&format!(" elapsed_ms={elapsed}"));
            }
            if let Some(details) = record.details {
                text.push_str(&format!(" code={}", details.code));
                if let Some(status) = details.status {
                    text.push_str(&format!(" status={status}"));
                }
                if let Some(bytes) = details.requested_bytes {
                    text.push_str(&format!(" requested_bytes={bytes}"));
                }
                if let Some(bytes) = details.available_bytes {
                    text.push_str(&format!(" available_bytes={bytes}"));
                }
                if let Some(code) = details.os_error {
                    text.push_str(&format!(" os_error={code}"));
                }
            }
            for cause in record.causes {
                text.push_str(&format!("; cause: {cause}"));
            }
            text.into_bytes()
        };
        line.push(b'\n');
        match &mut self.destination {
            Destination::Stderr => io::stderr().lock().write_all(&line),
            Destination::Explicit { file, .. } => {
                file.write_all(&line)?;
                file.flush()
            }
            Destination::Session(session) => session.append(&line),
        }
    }
}

/// No implicit file for one-off commands. Initialize after actual UI/serve
/// mode is selected, before any model or action work starts.
pub(crate) fn start(mode: Mode) {
    if LOG.get().is_some() {
        return;
    }
    let explicit = std::env::var_os("ZEROGRAVITY_LOG_FILE");
    if mode == Mode::Command && explicit.is_none() {
        return;
    }
    let opened = match explicit.as_deref() {
        Some(value) if value == "stderr" => {
            if mode == Mode::Interactive && io::stderr().is_terminal() {
                Err(invalid(
                    "TUI stderr logging requires redirecting stderr outside the terminal",
                ))
            } else {
                Ok(Destination::Stderr)
            }
        }
        Some(value) => {
            let path = Path::new(value);
            if !path.is_absolute() {
                Err(invalid(
                    "ZEROGRAVITY_LOG_FILE requires an absolute path or stderr",
                ))
            } else {
                private_file(path, false).and_then(|file| {
                    file.try_lock().map_err(io::Error::other)?;
                    Ok(Destination::Explicit {
                        path: path.to_owned(),
                        file,
                    })
                })
            }
        }
        None if matches!(mode, Mode::Interactive | Mode::Web) => state_root()
            .and_then(|root| Session::open(&root))
            .map(Destination::Session),
        None => Ok(Destination::Stderr),
    };
    let (destination, failure) = match opened {
        Ok(value) => (value, None),
        Err(error) => (
            Destination::Stderr,
            Some(bounded_message(&error.to_string())),
        ),
    };
    let jsonl = if matches!(&destination, Destination::Session(_)) {
        true
    } else {
        match std::env::var("ZEROGRAVITY_LOG_FORMAT").as_deref() {
            Ok("jsonl") => true,
            Ok("text") => false,
            _ => !matches!(&destination, Destination::Stderr),
        }
    };
    let requested_level = std::env::var("ZEROGRAVITY_LOG_LEVEL").ok();
    let minimum = requested_level
        .as_deref()
        .and_then(Level::parse)
        .unwrap_or(Level::Info);
    let _ = LOG.set(Mutex::new(Log {
        destination,
        mode,
        jsonl,
        minimum,
        started: Instant::now(),
        sequence: 0,
        failure,
    }));
    if requested_level
        .as_deref()
        .is_some_and(|value| Level::parse(value).is_none())
    {
        status(
            "logging",
            Level::Warning,
            "Invalid ZEROGRAVITY_LOG_LEVEL; using info",
        );
    }
    if mode != Mode::Interactive {
        if let Some(message) = failure_message() {
            eprintln!("WARNING: operational log unavailable; using stderr: {message}");
        } else if matches!(mode, Mode::Web | Mode::Command)
            && let Some(location) = location()
        {
            eprintln!("{location}");
        }
    }
}

pub(crate) fn active() -> bool {
    LOG.get().is_some()
}
pub(crate) fn interactive() -> bool {
    LOG.get().is_some_and(|log| {
        log.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .mode
            == Mode::Interactive
    })
}
pub(crate) fn replaces_terminal_failure() -> bool {
    LOG.get().is_some_and(|log| {
        let log = log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        log.mode == Mode::Service && log.failure.is_none()
    })
}
pub(crate) fn failure_message() -> Option<String> {
    LOG.get()?
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .failure
        .clone()
}
pub(crate) fn location() -> Option<String> {
    let log = LOG
        .get()?
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(failure) = &log.failure {
        return Some(format!("Log unavailable: {failure}"));
    }
    match &log.destination {
        Destination::Session(session) => Some(format!(
            "Log: {}",
            segment_path(&session.directory, session.segment).display()
        )),
        Destination::Explicit { path, .. } => Some(format!("Log: {}", path.display())),
        Destination::Stderr => Some("Log: stderr".into()),
    }
}

pub(crate) fn record(record: Record<'_>) {
    let Some(log) = LOG.get() else { return };
    let mut log = log
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Err(error) = log.write(&record) {
        let message = bounded_message(&error.to_string());
        let first_failure = log.failure.is_none();
        log.failure = Some(message.clone());
        if log.mode != Mode::Interactive && first_failure {
            eprintln!("WARNING: operational log failed; using stderr: {message}");
            log.destination = Destination::Stderr;
            log.jsonl = false;
            let _ = log.write(&record);
        }
    }
}
pub(crate) fn finish() {
    let Some(log) = LOG.get() else { return };
    let mut log = log
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = match &mut log.destination {
        Destination::Stderr => io::stderr().lock().flush(),
        Destination::Explicit { file, .. } => file.sync_all(),
        Destination::Session(session) => session.file.sync_all(),
    };
    if let Err(error) = result {
        let message = bounded_message(&error.to_string());
        log.failure = Some(message.clone());
        eprintln!("WARNING: operational log final flush failed: {message}");
    }
}

pub(crate) fn status(component: &str, level: Level, message: &str) {
    record(Record {
        timestamp_unix_ms: now_ms(),
        level,
        component,
        correlation: "process",
        message: &bounded_message(message),
        details: None,
        stage: None,
        elapsed_ms: None,
        causes: &[],
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_live_sessions_rotate_and_retain_closed_history() -> io::Result<()> {
        struct Temp(PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let root = std::env::temp_dir().join(format!(
            "zg-operational-{}-{}",
            std::process::id(),
            now_ms()
        ));
        private_directory(&root, false)?;
        let _cleanup = Temp(root.clone());
        let mut active = Session::open(&root)?;
        let active_path = active.directory.clone();
        let concurrent = Session::open(&root)?;
        let other_path = concurrent.directory.clone();
        // Real complete records remain bounded within four segments.
        let mut line = vec![b'x'; SEGMENT_BYTES as usize - 1];
        line.push(b'\n');
        for _ in 0..SEGMENTS + 2 {
            active.append(&line)?;
        }
        assert!(!segment_path(&active_path, 0).exists());
        assert_eq!(fs::read_dir(&active_path)?.count(), SEGMENTS as usize + 1);
        assert_eq!(
            fs::metadata(segment_path(&active_path, active.segment))?.len(),
            SEGMENT_BYTES
        );
        let unrelated = root.join("user-not-a-session.log");
        fs::write(&unrelated, b"untouched")?;
        for _ in 0..RETAIN_CLOSED + 3 {
            let session = Session::open(&root)?;
            session.file.sync_all()?;
            drop(session);
        }
        // One more open performs retention on the already closed sessions.
        let newest = Session::open(&root)?;
        assert!(active_path.is_dir() && other_path.is_dir());
        assert_eq!(fs::read(&unrelated)?, b"untouched");
        assert_eq!(
            fs::read_dir(&root)?
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .count(),
            RETAIN_CLOSED + 3
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&active_path)?.permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(active.file.metadata()?.permissions().mode() & 0o777, 0o600);
        }
        drop((active, concurrent, newest));
        Ok(())
    }

    #[test]
    fn operational_record_preserves_typed_cause_without_credential_text() -> io::Result<()> {
        use zerogravity::runtime::text::ServiceError;
        let cause = zerogravity::Error::resource("device memory", "model admission", 12, 8);
        let error = ServiceError::from_core(503, "source_capacity", cause);
        let details = error.details();
        assert_eq!(details.code, "source_capacity");
        assert_eq!(details.requested_bytes, Some(12));
        assert_eq!(details.available_bytes, Some(8));
        let chain = causes(&error);
        assert!(chain.iter().any(|cause| cause.contains("model admission")));
        assert!(
            chain
                .iter()
                .any(|cause| cause.contains("requested 12 bytes"))
        );
        assert_eq!(
            bounded_message("Bearer secret fixture value"),
            "[credential-bearing diagnostic omitted]"
        );
        assert_eq!(
            bounded_message("failed https://example.invalid?key=private"),
            "failed [redacted URL]"
        );
        assert_eq!(
            bounded_message("API http://127.0.0.1:8000/v1"),
            "API http://127.0.0.1:8000/v1"
        );
        let record = Record {
            timestamp_unix_ms: 1795829000123,
            level: Level::Error,
            component: "models",
            correlation: "action #7",
            message: "Load failed",
            details: Some(details),
            stage: Some("loading"),
            elapsed_ms: Some(123),
            causes: &chain,
        };
        let json = serde_json::to_value(record).map_err(io::Error::other)?;
        assert_eq!(json["timestamp_unix_ms"], 1795829000123u64);
        assert_eq!(json["details"]["requested_bytes"], 12);
        assert_eq!(json["details"]["status"], 503);
        assert_eq!(json["stage"], "loading");
        Ok(())
    }
}
