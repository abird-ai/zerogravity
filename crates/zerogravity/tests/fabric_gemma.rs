//! Explicit existing-source HIP SDK lifecycle; no ordinary serving qualification.
#![cfg(all(feature = "gemma4-hip-qualification", target_os = "linux"))]

use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity::{
    Error,
    cache::{CacheOptions, CacheStore},
    diagnostics::gemma_text::{Backend, Report, generate},
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter},
};

#[test]
#[ignore = "approved existing Gemma snapshot and gfx1151; isolated serial release process"]
fn original_hip_sdk_presentation_and_cancel_retire() {
    const TEST: &str = "original_hip_sdk_presentation_and_cancel_retire";
    const CHILD: &str = "ZG_GEMMA_HIP_SDK_CHILD";
    const DONE: &str = "GEMMA_HIP_SDK_CHILD_OK";
    let mode = std::env::var(CHILD).ok();
    if mode.is_none() {
        for mode in ["complete", "presentation-refusal"] {
            // Direct ld.so launch makes current_exe the loader, not this test.
            // Caller-owned wrapper keeps child launches on the same HIP runtime.
            let mut command = if let Some(runner) = std::env::var_os("ZG_GEMMA_HIP_SDK_RUNNER") {
                let runner = std::path::PathBuf::from(runner);
                assert!(runner.is_absolute(), "absolute test runner required");
                let executable = std::path::PathBuf::from(std::env::args_os().next().unwrap());
                assert!(
                    executable.is_absolute(),
                    "absolute test executable required"
                );
                let mut command = std::process::Command::new(runner);
                command.arg(executable);
                command
            } else {
                std::process::Command::new(std::env::current_exe().unwrap())
            };
            let child = command
                .args(["--exact", TEST, "--ignored", "--nocapture"])
                .env(CHILD, mode)
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "{mode} child {:?}: {} {}",
                child.status,
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
            assert!(String::from_utf8_lossy(&child.stdout).contains(&format!("{DONE} {mode}")));
            println!("GEMMA_HIP_SDK_CASE_PASS mode={mode}");
        }
        return;
    }
    let mode = mode.unwrap();
    assert!(matches!(mode.as_str(), "complete" | "presentation-refusal"));
    if cfg!(debug_assertions) {
        panic!("release physical target required");
    }
    let pci = "/sys/bus/pci/devices/0000:c6:00.0";
    assert_eq!(
        std::fs::read_to_string(format!("{pci}/vendor"))
            .unwrap()
            .trim(),
        "0x1002"
    );
    assert_eq!(
        std::fs::read_to_string(format!("{pci}/device"))
            .unwrap()
            .trim(),
        "0x1586"
    );
    let selector = std::env::var("ZG_APPROVED_GEMMA_HIP_SNAPSHOT")
        .expect("explicit existing approved snapshot");
    let (finish, deadline) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if matches!(
            deadline.recv_timeout(std::time::Duration::from_secs(1200)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ) {
            eprintln!("Gemma HIP SDK case exceeded1200seconds");
            std::process::exit(124);
        }
    });
    let store =
        CacheStore::open_existing(CacheOptions::new(None, 96 * 1024 * 1024 * 1024)).unwrap();
    let flag = AtomicBool::new(false);
    let mut texts = 0;
    let mut summaries = 0;
    let refuse = mode == "presentation-refusal";
    let result = generate(
        &store,
        &selector,
        "One",
        1,
        Backend::HipResident,
        &flag,
        |event| {
            match event {
                Report::Text(_) => {
                    texts += 1;
                    if refuse {
                        return Err(Error::State("caller presentation refused".into()));
                    }
                }
                Report::Finished {
                    backend,
                    emitted,
                    calls,
                    frontier,
                    first_root,
                } => {
                    assert!(!refuse, "no completion after refused presentation");
                    assert!(matches!(backend, Backend::HipResident));
                    assert!(emitted <= 1);
                    assert_eq!(calls, 1);
                    assert!(frontier > 0);
                    assert!(first_root.is_none());
                    summaries += 1;
                }
            }
            Ok(())
        },
    );
    if refuse {
        let error = result.unwrap_err();
        assert_eq!(
            (texts, summaries),
            (1, 0),
            "presentation not reached: {error:?}"
        );
        assert!(
            matches!(error, Error::State(ref reason) if reason == "caller presentation refused")
        );
    } else {
        result.unwrap();
        assert_eq!((texts, summaries), (1, 1));
    }
    let arbiter = ResourceArbiter::process_shared();
    let host = PhysicalDeviceKey::new("host", "system-ram").unwrap();
    let physical = PhysicalDeviceKey::new("pci", "0000:c6:00.0").unwrap();
    let baseline = (
        arbiter.snapshot(&host).unwrap(),
        arbiter.snapshot(&physical).unwrap(),
    );
    // Original process-lifetime context remains; model close is not global zero.
    assert_eq!(baseline.0.reserved, 2 * 1024 * 1024 * 1024);
    assert_eq!(baseline.1.reserved, 4 * 1024 * 1024 * 1024);
    assert!(!baseline.0.fenced && !baseline.1.fenced);
    if !refuse {
        // The original finite candidate refuses a warm runtime; no reload claim.
        let error = generate(
            &store,
            &selector,
            "One",
            1,
            Backend::HipResident,
            &flag,
            |_| panic!("cold-only candidate emitted after warm load"),
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::State(ref reason) if reason == "Gemma qualifier requires cold runtime configuration")
        );
        for (key, expected) in [(&host, &baseline.0), (&physical, &baseline.1)] {
            let after = arbiter.snapshot(key).unwrap();
            assert_eq!(after.reserved, expected.reserved);
            assert!(!after.fenced);
        }
    }
    flag.store(true, Ordering::Release);
    let error = generate(
        &store,
        &selector,
        "One",
        1,
        Backend::HipResident,
        &flag,
        |_| panic!("pre-cancelled source emitted"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::State(ref reason) if reason == "cache operation canceled"));
    for (key, expected) in [(&host, &baseline.0), (&physical, &baseline.1)] {
        let after = arbiter.snapshot(key).unwrap();
        assert_eq!(after.reserved, expected.reserved);
        assert!(!after.fenced);
    }
    finish.send(()).unwrap();
    watchdog.join().unwrap();
    println!("{DONE} {mode}");
}
