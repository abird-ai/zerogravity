//! Execution placement only; original flight/source/dispatch guards stay authoritative.
use super::*;

mod diagnostic;
pub(crate) use diagnostic::{
    ADMISSION_BYTES as ENQUEUE_PROFILE_ADMISSION_BYTES, enabled as enqueue_profile_enabled,
    head as profile_head_enqueue,
};

/// Derived from prepared operation behavior, never a mathematical permission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnqueuePlacement {
    /// Source IO or blocking checkpoint compatibility has not been established.
    Caller,
    /// Only synchronous original commands; completion stays outside the scope.
    DeviceActor,
}

impl ResidentSubmission<'_> {
    /// Keep the actual armed flight alive while its synchronous command scope
    /// runs on the original actor. Nested client calls use that same runner.
    /// No capture, completion, source permission or fallback is introduced.
    pub(crate) fn ordered_enqueue<T: Send + 'static>(
        &mut self,
        enqueue: impl FnOnce(&mut Self) -> Result<T> + Send,
    ) -> Result<T> {
        self.require_accesses(std::iter::empty())?;
        let service = self.service.clone();
        service
            .client()
            .exclusive(|| {
                self.require_accesses(std::iter::empty())?;
                enqueue(self)
            })
            .map_err(|error| Error::State(format!("original armed actor enqueue: {error}")))?
    }
}

#[cfg(test)]
mod tests;
