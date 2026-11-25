//! Process signals only request shutdown; original request/device owners retire work.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) struct Shutdown {
    pub(super) requested: Arc<AtomicBool>,
    #[cfg(target_os = "linux")]
    signals: [Option<signal_hook::SigId>; 2],
    // Bootstrap flag is priced as soon as the configured host exists, then
    // stays admitted through HTTP shutdown and the final inference joins.
    _charge: Option<super::pending::ServingCharge>,
}

impl Shutdown {
    pub(super) fn install() -> std::io::Result<Self> {
        let mut shutdown = Self {
            requested: Arc::new(AtomicBool::new(false)),
            #[cfg(target_os = "linux")]
            signals: [None, None],
            _charge: None,
        };
        #[cfg(target_os = "linux")]
        for (slot, signal) in shutdown
            .signals
            .iter_mut()
            .zip([signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT])
        {
            *slot = Some(signal_hook::flag::register(
                signal,
                Arc::clone(&shutdown.requested),
            )?);
        }
        #[cfg(not(target_os = "linux"))]
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "serving signal handling requires Linux",
        ));
        #[cfg(target_os = "linux")]
        Ok(shutdown)
    }

    pub(super) fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    pub(super) fn admit(
        &mut self,
        router: &impl zerogravity::serving::TextService,
    ) -> Result<(), super::ServiceError> {
        self._charge = Some(router.charge_ingress(
            "shutdown-signal",
            std::mem::size_of::<(usize, usize, AtomicBool)>() as u64,
        )?);
        Ok(())
    }

    pub(super) fn wait_control(
        &self,
        control: zerogravity::control::ControlPlane<'_>,
        runtime: &zerogravity::Runtime,
        binding: zerogravity::control::ServeBinding,
    ) -> Result<(), super::ServiceError> {
        use std::time::{Duration, Instant};
        use zerogravity::control::ServeSummary;
        let live = crate::output::reporter().serve_status();
        let mut sampled = Instant::now();
        let mut previous = None;
        while !self.is_requested() {
            match control.serving().map_err(super::ServiceError::inference)? {
                ServeSummary::Listening {
                    binding: current,
                    accepted_requests,
                    active_connections,
                } if current == binding => {
                    if live && sampled.elapsed() >= Duration::from_secs(2) {
                        if let Some(scheduler) = runtime
                            .scheduler_snapshot()
                            .map_err(super::ServiceError::inference)?
                        {
                            let state = (
                                accepted_requests,
                                active_connections,
                                scheduler.counters.active,
                                scheduler.queue.pending,
                                scheduler.counters.output_tokens,
                            );
                            // No idle spam; only emit when observed counters changed.
                            if previous != Some(state) {
                                crate::output::emit!(
                                    Progress,
                                    "Serving · HTTP {} · connections {} · active {}/{} · queued {} · output tokens {}",
                                    state.0,
                                    state.1,
                                    state.2,
                                    scheduler.active_limit,
                                    state.3,
                                    state.4
                                );
                                previous = Some(state);
                            }
                        }
                        sampled = Instant::now();
                    }
                }
                ServeSummary::Finished { binding: current } if current == binding => break,
                _ => {
                    return Err(super::ServiceError::new(
                        500,
                        "listener_changed",
                        "owned serving listener is no longer present",
                    ));
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn wait(&self, server: zerogravity::serving::ServerHandle) -> std::io::Result<()> {
        while !self.is_requested() && !server.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        server
            .shutdown()
            .map_err(|error| std::io::Error::other(error.to_string()))
    }
}

impl Drop for Shutdown {
    fn drop(&mut self) {
        // Remove our callbacks; signal-hook retains its OS dispatcher until
        // this standalone command exits. No prior disposition is restored.
        #[cfg(target_os = "linux")]
        for id in self.signals.iter_mut().filter_map(Option::take) {
            signal_hook::low_level::unregister(id);
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::serve::{engine, join_router, pending::ServingHost};
    use std::io::Write;
    use zerogravity::{
        identity::SourceIdentity,
        runtime::{LogitModel, PreparedModel, TextSession, TextTokenizer},
        serving::ServerBuilder,
    };

    fn refuse_execution<T>() -> zerogravity::Result<T> {
        Err(zerogravity::Error::Unsupported(
            "signal fixture has no tokenizer, model math or executable session".into(),
        ))
    }

    // Test-local allocation identity and admission metadata only; no verified
    // source, tokenizer output, logits or committed model state exists here.
    struct RefusingModel {
        source: SourceIdentity,
        tokenizer: Arc<RefusingTokenizer>,
    }
    struct RefusingTokenizer;
    impl TextTokenizer for RefusingTokenizer {
        fn eos(&self) -> u32 {
            1
        }
        fn decode(&self, _: &[u32], _: bool) -> zerogravity::Result<String> {
            refuse_execution()
        }
        fn prompt_tokens(&self, _: &str, _: bool) -> zerogravity::Result<Vec<u32>> {
            refuse_execution()
        }
        fn prompt_token_bound(&self, _: usize, _: bool) -> usize {
            0
        }
        fn admitted_prompt_scratch_bound(&self, _: usize, _: bool) -> zerogravity::Result<u64> {
            refuse_execution()
        }
        fn decode_scratch_bound(&self, _: usize) -> usize {
            0
        }
    }
    impl PreparedModel for RefusingModel {
        fn source(&self) -> SourceIdentity {
            self.source
        }
        fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
            self.tokenizer.clone()
        }
        fn max_context(&self) -> usize {
            16
        }

        fn max_token_bytes(&self) -> usize {
            0
        }
        fn vocabulary_size(&self) -> u64 {
            2
        }
    }
    impl LogitModel for RefusingModel {
        fn max_session_bytes(&self) -> u64 {
            0
        }
        fn session_scratch_bytes(&self) -> u64 {
            0
        }
        fn new_session(self: Arc<Self>) -> zerogravity::Result<Box<dyn TextSession>> {
            refuse_execution()
        }
    }

    fn idle_engine(host: ServingHost) -> engine::Router {
        let model: Arc<dyn LogitModel> = Arc::new(RefusingModel {
            source: SourceIdentity::for_new_allocation().unwrap(),
            tokenizer: Arc::new(RefusingTokenizer),
        });
        engine::Router::new(
            [("signal-fixture-unexecutable".into(), model)]
                .into_iter()
                .collect(),
            1,
            1,
            16,
            8,
            host,
            Some(8),
        )
        .unwrap()
    }

    /// External TERM/INT controls, including PID1 where supported, exercise
    /// HTTP handler and idle engine-worker closure. No inference/device coverage.
    #[test]
    #[ignore = "owned subprocess for tests/fixtures/serve_shutdown.py"]
    fn process_fixture() {
        assert_eq!(
            std::env::var("ZG_SERVE_SHUTDOWN_FIXTURE").as_deref(),
            Ok("1")
        );
        let mut shutdown = Shutdown::install().unwrap();
        let host = ServingHost::new(256).unwrap();
        let baseline = host.snapshot().unwrap().3;
        let engine = idle_engine(host.clone());
        assert_eq!(engine.queue_snapshot().consumers, 1, "one real idle worker");
        assert_eq!(engine.active_sessions(), 0);
        let router = Arc::new(engine);
        shutdown.admit(router.as_ref()).unwrap();
        let address =
            std::env::var("ZG_SERVE_SHUTDOWN_LISTEN").unwrap_or_else(|_| "127.0.0.1:0".into());
        let server = ServerBuilder::from_service(router.clone())
            .bind(address.parse().unwrap())
            .build()
            .unwrap()
            .start()
            .unwrap();
        assert!(!shutdown.is_requested(), "signal before fixture readiness");
        println!(
            "{}",
            serde_json::json!({"event":"READY","pid":std::process::id(),
                "address":server.local_addr().to_string()})
        );
        std::io::stdout().flush().unwrap();
        let stopped = shutdown.wait(server);
        // The embedded server must release its router borrows first. Explicitly
        // close the original queue and join its inference worker before CLOSED.
        let joined = join_router(router);
        stopped.unwrap();
        joined.unwrap();
        assert!(shutdown.is_requested());
        // Unregister/drop the final process-signal owner before observing zero.
        drop(shutdown);
        assert_eq!(
            host.snapshot().unwrap().3,
            baseline,
            "all transport and worker leases retired"
        );
        println!(
            "{}",
            serde_json::json!({"event":"CLOSED","pid":std::process::id(),
                "serving_host_bytes":host.snapshot().unwrap().3 - baseline})
        );
        std::io::stdout().flush().unwrap();
    }

    #[test]
    fn process_signals_close_serving() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/serve_shutdown.py");
        let output = std::process::Command::new("python3")
            .arg("-B")
            .arg(script)
            .arg("--test-binary")
            .arg(std::env::current_exe().unwrap())
            .arg("--child-test")
            .arg("serve::shutdown::tests::process_fixture")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .expect("launch owned serving signal controls");
        assert!(
            output.status.success(),
            "signal controls failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("serving signal control report JSON");
        let rows = report["serving_shutdown_controls"]
            .as_array()
            .expect("serving signal control rows");
        assert_eq!(rows.len(), 4, "TERM/INT × process/PID1 rows");
        for row in rows.iter().filter(|row| row["namespace_pid1"] == false) {
            assert_eq!(row["passed"], true, "ordinary process signal control");
            assert_ne!(row["skipped"], true, "ordinary process control cannot skip");
        }
        if report["pid1_supported"] == true {
            for row in rows.iter().filter(|row| row["namespace_pid1"] == true) {
                assert_eq!(row["passed"], true, "PID1 signal control");
                assert_ne!(row["skipped"], true, "supported PID1 control cannot skip");
            }
        } else {
            assert!(
                report["pid1_skip_reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.is_empty()),
                "unsupported PID1 environment must report why"
            );
            for row in rows.iter().filter(|row| row["namespace_pid1"] == true) {
                assert_eq!(
                    row["skipped"], true,
                    "unsupported PID1 controls are explicit"
                );
            }
        }
    }
}
