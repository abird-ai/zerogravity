//! Checked NVRTC retirement, including unwinding and uncertain destruction.
use std::panic::{AssertUnwindSafe, catch_unwind};

pub(super) trait ProgramResource {
    type Error;
    fn destroy(&mut self) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub(super) enum CleanupFailure<E> {
    Refused(E),
    Panicked,
}

/// No actor/context backedge. An uncertain resource keeps its source alive
/// until process exit; the original submitting owner must retain its charge.
pub(super) struct ProgramGuard<'a, P: ProgramResource> {
    resource: Option<P>,
    unknown: &'a mut bool,
}

impl<'a, P: ProgramResource> ProgramGuard<'a, P> {
    pub(super) fn new(resource: P, unknown: &'a mut bool) -> Self {
        Self {
            resource: Some(resource),
            unknown,
        }
    }

    pub(super) fn resource(&self) -> &P {
        self.resource.as_ref().expect("live compiler program")
    }

    pub(super) fn finish(mut self) -> Result<(), CleanupFailure<P::Error>> {
        self.retire()
    }

    fn retire(&mut self) -> Result<(), CleanupFailure<P::Error>> {
        let Some(mut resource) = self.resource.take() else {
            return Ok(());
        };
        let retired = catch_unwind(AssertUnwindSafe(|| resource.destroy()));
        match retired {
            Ok(Ok(())) => Ok(()),
            outcome => {
                *self.unknown = true;
                // A failed destroy may have consumed the raw handle. Never
                // retry or free source memory the compiler could still borrow.
                std::mem::forget(resource);
                Err(match outcome {
                    Ok(Err(error)) => CleanupFailure::Refused(error),
                    Err(_) => CleanupFailure::Panicked,
                    Ok(Ok(())) => unreachable!(),
                })
            }
        }
    }
}

impl<P: ProgramResource> Drop for ProgramGuard<'_, P> {
    fn drop(&mut self) {
        if self.resource.is_some() && std::thread::panicking() {
            // A lost actor result cannot establish successful preparation,
            // even if program retirement itself succeeds during unwinding.
            *self.unknown = true;
        }
        let _ = self.retire();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[derive(Clone, Copy)]
    enum Outcome {
        Success,
        Refused,
        Panic,
    }
    struct Program {
        outcome: Outcome,
        attempts: Rc<Cell<usize>>,
        source_drops: Rc<Cell<usize>>,
    }
    impl ProgramResource for Program {
        type Error = &'static str;
        fn destroy(&mut self) -> Result<(), Self::Error> {
            assert_eq!(self.source_drops.get(), 0);
            self.attempts.set(self.attempts.get() + 1);
            match self.outcome {
                Outcome::Success => Ok(()),
                Outcome::Refused => Err("injected nvrtcDestroyProgram failure"),
                Outcome::Panic => panic!("injected destroy panic"),
            }
        }
    }
    impl Drop for Program {
        fn drop(&mut self) {
            self.source_drops.set(self.source_drops.get() + 1);
        }
    }
    fn program(outcome: Outcome) -> (Program, Rc<Cell<usize>>, Rc<Cell<usize>>) {
        let attempts = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        (
            Program {
                outcome,
                attempts: attempts.clone(),
                source_drops: drops.clone(),
            },
            attempts,
            drops,
        )
    }

    #[test]
    fn checked_retirement_releases_source_exactly_once() {
        let (resource, attempts, drops) = program(Outcome::Success);
        let mut unknown = false;
        let guard = ProgramGuard::new(resource, &mut unknown);
        assert_eq!(guard.resource().attempts.get(), 0);
        guard.finish().unwrap();
        assert_eq!((attempts.get(), drops.get()), (1, 1));
        assert!(!unknown);
    }

    #[test]
    fn early_body_error_still_retires_program() {
        let (resource, attempts, drops) = program(Outcome::Success);
        let mut unknown = false;
        let body = || -> Result<(), &'static str> {
            let _guard = ProgramGuard::new(resource, &mut unknown);
            Err("compile/log/PTX refused")?;
            Ok(())
        };
        assert_eq!(body(), Err("compile/log/PTX refused"));
        assert_eq!((attempts.get(), drops.get()), (1, 1));
        assert!(!unknown);
    }

    #[test]
    fn checked_destroy_failure_retains_source_without_retry() {
        let (resource, attempts, drops) = program(Outcome::Refused);
        let mut unknown = false;
        assert!(matches!(
            ProgramGuard::new(resource, &mut unknown).finish(),
            Err(CleanupFailure::Refused(_))
        ));
        assert_eq!((attempts.get(), drops.get()), (1, 0));
        assert!(unknown);
    }

    #[test]
    fn checked_destroy_panic_retains_source_without_retry() {
        let (resource, attempts, drops) = program(Outcome::Panic);
        let mut unknown = false;
        assert!(matches!(
            ProgramGuard::new(resource, &mut unknown).finish(),
            Err(CleanupFailure::Panicked)
        ));
        assert_eq!((attempts.get(), drops.get()), (1, 0));
        assert!(unknown);
    }

    #[test]
    fn implicit_destroy_failure_is_sticky_and_retains_source() {
        let (resource, attempts, drops) = program(Outcome::Refused);
        let mut unknown = false;
        drop(ProgramGuard::new(resource, &mut unknown));
        assert_eq!((attempts.get(), drops.get()), (1, 0));
        assert!(unknown);
    }

    #[test]
    fn body_panic_retires_program_but_cannot_claim_ready() {
        let (resource, attempts, drops) = program(Outcome::Success);
        let mut unknown = false;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _guard = ProgramGuard::new(resource, &mut unknown);
            panic!("injected lost compiler result");
        }));
        assert!(outcome.is_err());
        assert_eq!((attempts.get(), drops.get()), (1, 1));
        assert!(unknown);
    }

    #[test]
    fn simultaneous_body_and_cleanup_panics_retain_without_double_panic() {
        let (resource, attempts, drops) = program(Outcome::Panic);
        let mut unknown = false;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _guard = ProgramGuard::new(resource, &mut unknown);
            panic!("injected lost compiler result");
        }));
        assert!(outcome.is_err());
        assert_eq!((attempts.get(), drops.get()), (1, 0));
        assert!(unknown);
    }
}
