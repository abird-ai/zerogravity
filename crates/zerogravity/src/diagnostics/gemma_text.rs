//! Explicit managed Gemma candidates, not a runtime/serving qualification.
//! Original snapshot, tokenizer, session, feedback and final-owner checks.
pub use crate::gemma4::execution::CPU_PROFILE;
use crate::gemma4::execution::{Context, MAX_CONTEXT, MAX_OUTPUT, Model};
use crate::{Error, Result, cache::CacheStore};
use std::sync::{Arc, atomic::AtomicBool};
pub const HIP_PROFILE: &str = "gemma4-e2b-bf16-hip-resident-candidate";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "gemma4-hip-qualification")]
    HipResident,
}
impl Backend {
    pub fn from_profile(profile: &str) -> Result<Self> {
        match profile {
            CPU_PROFILE => Ok(Self::Cpu),
            #[cfg(feature = "gemma4-hip-qualification")]
            HIP_PROFILE => Ok(Self::HipResident),
            _ => Err(Error::Unsupported(
                "Gemma source profile is not compiled or not registered".into(),
            )),
        }
    }
}
pub enum Report<'a> {
    Text(&'a str),
    Finished {
        backend: Backend,
        emitted: usize,
        calls: usize,
        frontier: u64,
        first_root: Option<u64>,
    },
}

/// Standalone candidate ownership, as before; not simultaneous model residency.
/// HIP retains its cold-only runtime: one successful candidate load per process.
/// Cache policy, cancellation, signals and all process IO belong to the caller.
pub fn generate(
    store: &CacheStore,
    selector: &str,
    prompt: &str,
    max_output: usize,
    backend: Backend,
    flag: &AtomicBool,
    mut report: impl FnMut(Report<'_>) -> Result<()>,
) -> Result<()> {
    if selector.is_empty()
        || selector.len() > 4096
        || selector.chars().any(char::is_control)
        || prompt.len() > 4096
        || !(1..=MAX_OUTPUT).contains(&max_output)
    {
        return Err(Error::Limit(
            "bounded Gemma snapshot/plain prompt/output required".into(),
        ));
    }
    let context = match backend {
        Backend::Cpu => Context::new()?,
        #[cfg(feature = "gemma4-hip-qualification")]
        Backend::HipResident => Context::new_hip_qualification()?,
    };
    let snapshot = Arc::new(
        if selector.len() == 64
            && selector
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            store.open_snapshot_cancellable(selector, flag)?
        } else {
            store.resolve_ref_cancellable(selector, flag)?
        },
    );
    let input = context.prepare_text(&snapshot, prompt, Some(flag))?;
    if input.ids().len() + max_output - 1 > MAX_CONTEXT {
        return Err(Error::Limit(
            "Gemma prompt+autoregressive outputs exceeds64consumed rows".into(),
        ));
    }
    #[cfg(feature = "gemma4-hip-qualification")]
    if backend == Backend::HipResident {
        // Retain an admitted decoder from the real caller input, not a fixture prompt.
        let decoder = context.prepare_text(&snapshot, prompt, Some(flag))?;
        let loaded = Model::open(snapshot.clone(), context.clone(), Some(flag))?;
        let result = gpu::generate(&loaded, input, &decoder, max_output, flag, &mut report);
        drop(decoder);
        let closed = loaded.close(None);
        drop(snapshot);
        drop(context);
        return completed(result, closed);
    }
    let loaded = Model::open(snapshot.clone(), context.clone(), Some(flag))?;
    let result = (|| -> Result<()> {
        let mut session = loaded.cpu_session(input)?;
        let first = session.infer_prompt(max_output, Some(flag))?;
        let mut calls = 1;
        while !session.stopped() {
            session.infer_next(Some(flag))?;
            calls += 1;
        }
        let text = session.decode_generated(true, Some(flag))?;
        report(Report::Text(text.as_str()))?;
        report(Report::Finished {
            backend,
            emitted: session.generated_ids().len(),
            calls,
            frontier: session.binding().frontier(),
            first_root: Some(first.root.root()),
        })
    })();
    // Returned setup/cancellation/presentation errors still validate final source.
    let closed = loaded.close(None);
    drop(snapshot);
    let reserved = context.reserved_bytes();
    drop(context);
    let retired = (|| -> Result<()> {
        let reserved = reserved?;
        let final_bytes = crate::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&crate::resource_arbiter::PhysicalDeviceKey::new(
                "host",
                "system-ram",
            )?)?
            .reserved;
        if reserved != 3 * 1024 * 1024 || final_bytes != 0 {
            return Err(Error::State(
                "Gemma ordinary source/session/decoder owners not retired".into(),
            ));
        }
        Ok(())
    })();
    completed(completed(result, closed), retired)
}
fn completed(result: Result<()>, closed: Result<()>) -> Result<()> {
    match (result, closed) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(close)) => Err(Error::State(format!(
            "Gemma operation failed: {error}; source retirement failed: {close}"
        ))),
    }
}
#[cfg(feature = "gemma4-hip-qualification")]
mod gpu;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_profiles_and_cleanup_errors_remain_explicit() {
        assert_eq!(Backend::from_profile(CPU_PROFILE).unwrap(), Backend::Cpu);
        assert!(Backend::from_profile("general-gemma").is_err());
        #[cfg(not(feature = "gemma4-hip-qualification"))]
        assert!(Backend::from_profile(HIP_PROFILE).is_err());
        let result = completed(
            Err(Error::State("presentation".into())),
            Err(Error::State("final source".into())),
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("presentation") && error.contains("final source"));
        let error = completed(
            Err(Error::State(error)),
            Err(Error::State("retirement accounting".into())),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("presentation")
                && error.contains("final source")
                && error.contains("retirement accounting")
        );
    }
    #[test]
    #[ignore = "requires explicit existing approved Gemma snapshot; standalone CPU only"]
    fn existing_source_cpu_sdk_and_presentation_failure_retire_original_owners() {
        const TEST: &str = "diagnostics::gemma_text::tests::existing_source_cpu_sdk_and_presentation_failure_retire_original_owners";
        const MARKER: &str = "ZG_GEMMA_SDK_CPU_CHILD";
        const DONE: &str = "GEMMA_CPU_SDK_CHILD_OK";
        if std::env::var_os(MARKER).is_none() {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--ignored", "--nocapture"])
                .env(MARKER, "1")
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "child {:?}: {} {}",
                child.status,
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
            assert!(String::from_utf8_lossy(&child.stdout).contains(DONE));
            return;
        }
        let selector = std::env::var("ZG_APPROVED_GEMMA_CPU_SNAPSHOT")
            .expect("explicit existing approved snapshot");
        let (finish, deadline) = std::sync::mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if matches!(
                deadline.recv_timeout(std::time::Duration::from_secs(600)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                eprintln!("Gemma CPU SDK case exceeded600seconds");
                std::process::exit(124);
            }
        });
        let store = CacheStore::open_existing(crate::cache::CacheOptions::new(
            None,
            96 * 1024 * 1024 * 1024,
        ))
        .unwrap();
        let flag = AtomicBool::new(false);
        let mut output = None;
        let mut summary = None;
        generate(&store, &selector, "One", 1, Backend::Cpu, &flag, |event| {
            match event {
                Report::Text(text) => {
                    assert!(output.replace(text.to_owned()).is_none());
                }
                Report::Finished {
                    backend,
                    emitted,
                    calls,
                    frontier,
                    first_root,
                } => {
                    assert_eq!(backend, Backend::Cpu);
                    assert!(emitted <= 1);
                    assert_eq!(calls, 1);
                    assert!(frontier > 0 && first_root.is_some_and(|root| root > 0));
                    assert!(summary.replace((emitted, calls, frontier)).is_none());
                }
            }
            Ok(())
        })
        .unwrap();
        assert!(
            output.is_some() && summary.is_some(),
            "actual source completion and presentation"
        );
        let arbiter = crate::resource_arbiter::ResourceArbiter::process_shared();
        let key = crate::resource_arbiter::PhysicalDeviceKey::new("host", "system-ram").unwrap();
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
        assert!(!arbiter.snapshot(&key).unwrap().fenced);
        let error = generate(&store, &selector, "One", 1, Backend::Cpu, &flag, |_| {
            Err(Error::State("caller presentation refused".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("caller presentation refused"));
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
        assert!(!arbiter.snapshot(&key).unwrap().fenced);
        flag.store(true, std::sync::atomic::Ordering::Release);
        assert!(
            generate(
                &store,
                &selector,
                "One",
                1,
                Backend::Cpu,
                &flag,
                |_| panic!("cancelled source emitted")
            )
            .is_err()
        );
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
        finish.send(()).unwrap();
        watchdog.join().unwrap();
        println!("{DONE}");
    }
}
