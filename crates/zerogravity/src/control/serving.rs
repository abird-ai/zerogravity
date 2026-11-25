//! Optional control-owned transport. The original ServerHandle owns shutdown.
use super::{ControlEventKind, ControlPlane, ServeBinding, ServeSummary};
use crate::{
    Error, Result,
    runtime::PrefixSharingScope,
    serving::{ServerHandle, ingress::IngressLimits},
};
use std::{
    net::SocketAddr,
    sync::{Mutex, MutexGuard, TryLockError},
};

/// Trusted Rust caller supplies explicit inference auth, never control auth.
/// Not serialized or logged.
pub struct ServeOptions {
    pub address: SocketAddr,
    pub limits: IngressLimits,
    pub api_key: Option<String>,
    pub enable_per_request_metrics: bool,
    pub prefix_sharing_scope: Option<PrefixSharingScope>,
}

#[derive(Default)]
pub(super) struct ServeState {
    slot: Mutex<Slot>,
}
#[derive(Default)]
struct Slot {
    generation: u64,
    server: Option<(ServeBinding, ServerHandle)>,
}

impl ServeState {
    fn change(&self) -> Result<MutexGuard<'_, Slot>> {
        self.slot.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => Error::State("serving transition in progress".into()),
            TryLockError::Poisoned(_) => Error::State("serving control lock poisoned".into()),
        })
    }
    pub(super) fn observe(&self) -> Result<ServeSummary> {
        let slot = match self.slot.try_lock() {
            Ok(slot) => slot,
            Err(TryLockError::WouldBlock) => return Ok(ServeSummary::Transitioning),
            Err(TryLockError::Poisoned(_)) => {
                return Err(Error::State("serving control lock poisoned".into()));
            }
        };
        Ok(match &slot.server {
            None => ServeSummary::Idle,
            Some((binding, handle)) if handle.is_finished() => {
                ServeSummary::Finished { binding: *binding }
            }
            Some((binding, handle)) => {
                let metrics = handle.metrics();
                ServeSummary::Listening {
                    binding: *binding,
                    accepted_requests: metrics.accepted_requests,
                    active_connections: metrics.active_connections,
                }
            }
        })
    }
}

impl ControlPlane<'_> {
    /// Start exactly one control-owned listener over the existing runtime.
    /// The slot excludes concurrent start/stop; snapshots report a transition
    /// without blocking on the original transport admission or thread join.
    pub fn start_serving(&self, options: ServeOptions) -> Result<ServeBinding> {
        let result = (|| {
            let mut slot = self.state.serving.change()?;
            if slot.server.is_some() {
                return Err(Error::State(
                    "control-owned inference server already exists; stop it first".into(),
                ));
            }
            let generation = slot
                .generation
                .checked_add(1)
                .ok_or_else(|| Error::Limit("serving generation exhausted".into()))?;
            let handle = self
                .runtime
                .server()
                .bind(options.address)
                .limits(options.limits)
                .api_key(options.api_key)
                .enable_per_request_metrics(options.enable_per_request_metrics)
                .prefix_sharing_scope(options.prefix_sharing_scope)
                .build()?
                .start()?;
            let binding = ServeBinding {
                generation,
                address: handle.local_addr(),
            };
            slot.generation = generation;
            slot.server = Some((binding, handle));
            Ok(binding)
        })();
        self.emit(ControlEventKind::ServeStartFinished {
            success: result.is_ok(),
        });
        result
    }

    /// Stop and join only the observed listener, not the runtime or its models.
    /// A stale generation cannot stop a replacement. Also valid after runtime
    /// shutdown: transport and inference have independent owners.
    pub fn stop_serving(&self, generation: u64) -> Result<()> {
        let result = (|| {
            let mut slot = self.state.serving.change()?;
            if slot
                .server
                .as_ref()
                .is_none_or(|(binding, _)| binding.generation != generation)
            {
                return Err(Error::State(
                    "selected inference server generation is no longer owned".into(),
                ));
            }
            let (_, handle) = slot.server.take().expect("checked server generation");
            handle.shutdown()
        })();
        self.emit(ControlEventKind::ServeStopFinished {
            generation,
            success: result.is_ok(),
        });
        result
    }
}
