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

    pub(super) fn admit(&mut self, router: &super::Router) -> Result<(), super::ServiceError> {
        self._charge = Some(router.charge_ingress(
            "shutdown-signal",
            std::mem::size_of::<(usize, usize, AtomicBool)>() as u64,
        )?);
        Ok(())
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
    use crate::serve::{Router, engine, http, join_router};
    use std::io::Write;
    use std::net::TcpListener;

    /// External driver sends actual TERM/INT, including in a PID namespace.
    /// No model route exists; this fixture proves transport/worker ownership.
    #[test]
    #[ignore = "owned subprocess for tests/fixtures/serve_shutdown.py"]
    fn process_fixture() {
        assert_eq!(
            std::env::var("ZG_SERVE_SHUTDOWN_FIXTURE").as_deref(),
            Ok("1")
        );
        let mut shutdown = Shutdown::install().unwrap();
        let (router, host) = engine::Router::transport_test();
        let router = Arc::new(Router::Cpu(router));
        shutdown.admit(&router).unwrap();
        let address =
            std::env::var("ZG_SERVE_SHUTDOWN_LISTEN").unwrap_or_else(|_| "127.0.0.1:0".into());
        let listener = TcpListener::bind(address).unwrap();
        listener.set_nonblocking(true).unwrap();
        assert!(!shutdown.is_requested(), "signal before fixture readiness");
        println!(
            "{}",
            serde_json::json!({"event":"READY","pid":std::process::id(),
                "address":listener.local_addr().unwrap().to_string()})
        );
        std::io::stdout().flush().unwrap();
        let result = http::serve(
            listener,
            Arc::clone(&router),
            None,
            false,
            Arc::clone(&shutdown.requested),
        );
        let joined = join_router(router);
        result.unwrap();
        joined.unwrap();
        assert!(shutdown.is_requested());
        // Unregister/drop the final process-signal owner before observing zero.
        drop(shutdown);
        assert_eq!(
            host.serving_bytes(),
            0,
            "all transport and worker leases retired"
        );
        println!(
            "{}",
            serde_json::json!({"event":"CLOSED","pid":std::process::id(),
                "serving_host_bytes":host.serving_bytes()})
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
    }
}
