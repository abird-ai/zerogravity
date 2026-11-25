//! Presentation/keyboard adapter over the same bounded control observations.
//! No second transfer loop, cache writer, request parser or progress authority.
use crate::output::{Event, Switch, reporter};
use rustix::termios::{self, LocalModes, OptionalActions, SpecialCodeIndex, Termios};
use std::{
    io::{self, IsTerminal, Write},
    time::Duration,
};
use zerogravity::{
    Error, Result,
    control::{
        ControlPlane, ControlPoll, DownloadCommand, DownloadOperation, DownloadStatus,
        DownloadSummary,
    },
};

/// Normal exit/unwind restores the original terminal modes. Keep ISIG so the
/// existing signal guard remains the sole SIGINT/SIGTERM cancellation owner.
pub(super) struct TerminalInput(Termios);
impl TerminalInput {
    /// Validate the capability before credentials, cache or provider IO.
    pub(super) fn prepare() -> Result<Option<Termios>> {
        let policy = reporter();
        let requested = policy.download_interaction();
        let enabled = match requested {
            Switch::Never => false,
            Switch::Auto => policy.download_auto_input(),
            Switch::Always => true,
        };
        if !enabled {
            return Ok(None);
        }
        let stdin = io::stdin();
        let capable = stdin.is_terminal()
            && io::stderr().is_terminal()
            && std::env::var_os("TERM").is_none_or(|term| term != "dumb")
            && termios::tcgetpgrp(&stdin).is_ok_and(|group| group == rustix::process::getpgrp());
        if capable {
            match termios::tcgetattr(&stdin) {
                Ok(saved) => return Ok(Some(saved)),
                Err(error) if requested == Switch::Always => return Err(Error::Io(error.into())),
                Err(_) => {}
            }
        }
        if requested == Switch::Always {
            return Err(Error::Unsupported(
                "interactive pull requires a foreground capable terminal".into(),
            ));
        }
        Ok(None)
    }

    /// Enter only after the process cancellation handlers are installed.
    pub(super) fn enter(saved: Option<Termios>) -> Result<Option<Self>> {
        let Some(saved) = saved else {
            return Ok(None);
        };
        let mut mode = saved.clone();
        mode.local_modes
            .remove(LocalModes::ICANON | LocalModes::ECHO | LocalModes::ECHONL);
        mode.special_codes[SpecialCodeIndex::VMIN] = 0;
        mode.special_codes[SpecialCodeIndex::VTIME] = 0;
        termios::tcsetattr(io::stdin(), OptionalActions::Now, &mode)
            .map_err(|error| Error::Io(error.into()))?;
        Ok(Some(Self(saved)))
    }

    fn keys(&self, control: ControlPlane<'_>, id: u64) -> Result<()> {
        let stdin = io::stdin();
        if !termios::tcgetpgrp(&stdin).is_ok_and(|group| group == rustix::process::getpgrp()) {
            return Err(Error::State(
                "download terminal no longer foreground".into(),
            ));
        }
        let mut bytes = [0u8; 32];
        let count = match rustix::io::read(&stdin, &mut bytes[..]) {
            Ok(count) => count,
            Err(error) if error == rustix::io::Errno::INTR || error == rustix::io::Errno::AGAIN => {
                return Ok(());
            }
            Err(error) => return Err(Error::Io(error.into())),
        };
        for byte in &bytes[..count] {
            let command = match byte {
                b'c' | b'C' => DownloadCommand::Cancel,
                b' ' => {
                    let Some(row) = control.download().filter(|row| row.id == id) else {
                        break;
                    };
                    if matches!(
                        row.status,
                        DownloadStatus::Paused | DownloadStatus::PauseRequested
                    ) {
                        DownloadCommand::Resume
                    } else {
                        DownloadCommand::Pause
                    }
                }
                _ => continue,
            };
            // A key racing a completed operation or publication never retargets.
            let _ = control.control_download(id, command);
        }
        Ok(())
    }
}
impl Drop for TerminalInput {
    fn drop(&mut self) {
        if termios::tcsetattr(io::stdin(), OptionalActions::Now, &self.0).is_err() {
            crate::output::emit!(
                Warning,
                "Could not restore terminal input modes; run stty sane in your terminal."
            );
        }
    }
}

#[derive(Default)]
struct Display {
    last_state: Option<(zerogravity::control::DownloadPhase, DownloadStatus, bool)>,
    last_progress: Option<(Option<String>, u64, u64, u64, u64)>,
    last_jsonl: Option<DownloadSummary>,
}
impl Display {
    fn render(&mut self, row: &DownloadSummary) {
        let policy = reporter();
        if policy.download_jsonl() {
            if (policy.progress() || row.status.terminal()) && self.last_jsonl.as_ref() != Some(row)
            {
                // The same owner snapshot as the human view. Notifications
                // may coalesce; terminal receipts are rendered separately.
                let line = serde_json::json!({"event":"download.state", "download":row});
                let mut out = io::stdout().lock();
                let _ = serde_json::to_writer(&mut out, &line);
                let _ = writeln!(out);
                self.last_jsonl = Some(row.clone());
            }
            return;
        }
        if !policy.progress() {
            return;
        }
        let state = (row.phase, row.status, row.cancel_requested);
        if self.last_state != Some(state) {
            crate::output::emit!(
                Progress,
                "Download: {:?} · {:?}{}",
                row.phase,
                row.status,
                if row.cancel_requested {
                    " (cancel requested)"
                } else {
                    ""
                }
            );
            self.last_state = Some(state);
        }
        if let Some(path) = &row.path {
            let progress = (
                row.path.clone(),
                row.total_bytes,
                row.cached_bytes,
                row.resumed_bytes,
                row.checkpoint_bytes,
            );
            if self.last_progress.as_ref() != Some(&progress) {
                let total =
                    super::selected_bytes(row.total_expected_bytes, row.total_maximum_bytes);
                let file = super::selected_bytes(row.file_expected_bytes, row.file_maximum_bytes);
                let rate = row.logical_bytes_per_second.filter(|rate| *rate > 0);
                let eta = rate.and_then(|rate| {
                    row.total_expected_bytes
                        .map(|total| total.saturating_sub(row.total_bytes) / rate)
                });
                let suffix = rate.map_or_else(String::new, |rate| {
                    format!(
                        " · {:.1} MiB/s logical{}",
                        rate as f64 / (1024.0 * 1024.0),
                        eta.map_or_else(String::new, |seconds| format!(" · ~{seconds}s"))
                    )
                });
                let _ = policy.write(
                    Event::Progress(format_args!(
                        "Pull bytes: {}/{} (file {}/{}, cached {}, resumed {}) · {path}{suffix}",
                        row.total_bytes,
                        total,
                        row.file_bytes,
                        file,
                        row.cached_bytes,
                        row.resumed_bytes
                    )),
                    &mut io::stderr().lock(),
                );
                self.last_progress = Some(progress);
            }
        }
    }
}

/// At most one scoped transfer worker, joined on every normal return. With
/// suppressed presentation there is no extra thread. Progress remains best
/// effort; only the original typed receipt determines command success.
pub(super) fn drive<'a, T: Send>(
    control: ControlPlane<'a>,
    operation: DownloadOperation<'a>,
    terminal: Option<TerminalInput>,
    run: impl FnOnce(DownloadOperation<'a>) -> Result<T> + Send,
) -> Result<T> {
    let mut display = Display::default();
    let policy = reporter();
    let id = operation.id();
    let observe = |display: &mut Display| {
        if let Some(row) = control.download().filter(|row| row.id == id) {
            display.render(&row);
        }
    };
    if terminal.is_none() && !policy.progress() {
        let result = run(operation);
        observe(&mut display);
        return result;
    }
    if terminal.is_some() {
        crate::output::emit!(Status, "space pause/resume · c cancel · Ctrl-C cancel");
    }
    let mut events = control.subscribe();
    observe(&mut display);
    let result = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("zg-download".into())
            .spawn_scoped(scope, move || run(operation))?;
        let mut input_ok = terminal.is_some();
        while !worker.is_finished() {
            if input_ok
                && let Some(terminal) = &terminal
                && let Err(error) = terminal.keys(control, id)
            {
                crate::output::emit!(
                    Warning,
                    "Interactive input unavailable; requesting cancellation: {error}"
                );
                let _ = control.control_download(id, DownloadCommand::Cancel);
                input_ok = false;
            }
            if !matches!(events.poll(Duration::from_millis(50)), ControlPoll::Pending) {
                observe(&mut display);
            }
        }
        worker
            .join()
            .map_err(|_| Error::State("download worker panicked".into()))?
    });
    observe(&mut display);
    result
}

pub(super) fn jsonl_receipt(
    receipt: &zerogravity::acquisition::PullReceipt,
    reference: &str,
) -> Result<()> {
    use zerogravity::cache::CommitOutcome;
    let (snapshot, durable) = match receipt.outcome() {
        CommitOutcome::Durable(snapshot) => (snapshot, true),
        CommitOutcome::DurabilityUnknown { snapshot, .. } => (snapshot, false),
    };
    let manifest: serde_json::Value =
        serde_json::from_slice(&receipt.manifest().canonical_bytes()?)
            .map_err(|_| Error::State("cache manifest encoding".into()))?;
    let mut out = io::stdout().lock();
    serde_json::to_writer(
        &mut out,
        &serde_json::json!({
            "event":"download.receipt",
            "snapshot":snapshot.digest(),
            "reference":reference,
            "durable":durable,
            "manifest":manifest,
            "execution_qualified":false,
        }),
    )
    .map_err(|error| Error::State(error.to_string()))?;
    writeln!(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Read,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };
    use zerogravity::{
        Runtime,
        acquisition::{
            PullLimits, ResolvedFile, ResolvedPullPlan,
            http::{HttpRequest, HttpResponse, HttpTransport},
        },
        cache::{CacheOptions, CacheStore, FileRole, ManifestSource},
    };

    // Explicit process fixture for the real terminal/JSONL adapter. No
    // production test URL override, credentials or remote transfer.
    #[test]
    #[ignore = "owned monitor subprocess; invoked with private fixture root"]
    fn monitor_child() {
        let root = std::path::PathBuf::from(std::env::var_os("ZG_MONITOR_FIXTURE").unwrap());
        let mode = std::env::var("ZG_MONITOR_MODE").unwrap();
        let interactive = mode == "interactive" || mode == "cancel";
        crate::output::install(crate::output::Options {
            jsonl: !interactive,
            plain: !interactive,
            interactive: if interactive {
                Switch::Always
            } else {
                Switch::Never
            },
            verbosity: if mode == "quiet" {
                crate::output::Verbosity::Quiet
            } else {
                crate::output::Verbosity::Normal
            },
            ..Default::default()
        });
        const SIZE: usize = 8192;
        struct Body(std::io::Cursor<Vec<u8>>);
        impl Read for Body {
            fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
                std::thread::sleep(Duration::from_millis(12));
                let n = out.len().min(128);
                self.0.read(&mut out[..n])
            }
        }
        #[derive(Default)]
        struct Transport(AtomicUsize);
        impl HttpTransport for Transport {
            fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
                self.0.fetch_add(1, Ordering::Relaxed);
                let start = request
                    .range
                    .as_ref()
                    .map_or(0, |range| range.start as usize);
                let mut headers = std::collections::BTreeMap::from([
                    ("etag".into(), "\"monitor-stable\"".into()),
                    ("content-length".into(), (SIZE - start).to_string()),
                ]);
                if let Some(range) = &request.range {
                    assert_eq!(range.validator, "\"monitor-stable\"");
                    assert_eq!(range.total, SIZE as u64);
                    headers.insert(
                        "content-range".into(),
                        format!("bytes {start}-{}/{SIZE}", SIZE - 1),
                    );
                }
                Ok(HttpResponse {
                    status: if start == 0 { 200 } else { 206 },
                    headers,
                    body: Box::new(Body(std::io::Cursor::new(vec![b'a'; SIZE - start]))),
                })
            }
        }
        let store =
            CacheStore::open(CacheOptions::new(Some(root.clone()), 16 * 1024 * 1024)).unwrap();
        let plan = ResolvedPullPlan::new(
            ManifestSource::new("github", "fixture/monitor", "1".repeat(40)).unwrap(),
            vec![
                ResolvedFile::new(
                    "LICENSE",
                    "https://fixture.invalid/file?secret=NOT_IN_OUTPUT",
                    FileRole::Auxiliary,
                    SIZE as u64,
                    None,
                    "fixture-file",
                )
                .unwrap(),
            ],
            &PullLimits::default(),
        )
        .unwrap();
        let admission = zerogravity::artifacts::transfer_admission(128 * 1024 * 1024).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let _signals = crate::signals::CancellationSignals::register(cancel.clone()).unwrap();
        let terminal = TerminalInput::enter(TerminalInput::prepare().unwrap()).unwrap();
        let runtime = Runtime::builder().build().unwrap();
        let control = runtime.control();
        let operation = control.begin_resolved_download(cancel).unwrap();
        let transport = Transport::default();
        let result = drive(control, operation, terminal, |operation| {
            operation.execute_plan(
                &store,
                &plan,
                &transport,
                &admission,
                Some("fixture"),
                &mut |_| {},
            )
        });
        if mode == "cancel" {
            assert!(result.is_err());
            assert_eq!(
                control.download().unwrap().status,
                DownloadStatus::Cancelled
            );
            assert!(store.resolve_ref("fixture").is_err());
            assert!(!control.cached_partials(&root).unwrap().is_empty());
        } else {
            let receipt = result.unwrap();
            if interactive {
                assert_eq!(transport.0.load(Ordering::Relaxed), 2);
            } else {
                jsonl_receipt(&receipt, "fixture").unwrap();
            }
            assert_eq!(
                store
                    .resolve_ref("fixture")
                    .unwrap()
                    .manifest()
                    .total_bytes(),
                SIZE as u64
            );
        }
    }
}
