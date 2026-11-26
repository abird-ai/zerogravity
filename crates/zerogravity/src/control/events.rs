//! Bounded transient notifications, not a second state store or task stream.
use serde::Serialize;
use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex, MutexGuard},
    time::Duration,
};

const CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReclaimStatus {
    DryRun,
    Busy,
    Referenced,
    Durable,
    Incomplete,
    Failed,
}

/// Notifications report original action outcomes, never inferred readiness.
/// No prompts, response text, paths, credentials or model handles are retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlEventKind {
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    DownloadChanged {
        id: u64,
        // Status at emission; a later operation may replace this observation.
        status: super::DownloadStatus,
    },
    ModelLoadFinished {
        success: bool,
    },
    ModelUnloadFinished {
        registration_id: u64,
        success: bool,
    },
    RequestAccepted {
        registration_id: u64,
    },
    CacheAliasFinished {
        success: bool,
    },
    CacheVerificationFinished {
        success: bool,
    },
    CacheReclamationFinished {
        outcome: ReclaimStatus,
    },
    ServeStartFinished {
        success: bool,
    },
    ServeStopFinished {
        generation: u64,
        success: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ControlEvent {
    pub sequence: u64,
    #[serde(flatten)]
    pub kind: ControlEventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlPoll {
    Event(ControlEvent),
    /// Missed transient notifications: reload the relevant authoritative views.
    Resync {
        missed: u64,
    },
    Pending,
}

#[derive(Default)]
struct Log {
    next: u64,
    events: VecDeque<ControlEvent>,
}

#[derive(Default)]
pub(super) struct Events {
    log: Mutex<Log>,
    changed: Condvar,
}
impl Events {
    fn lock(&self) -> MutexGuard<'_, Log> {
        self.log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn publish(&self, kind: ControlEventKind) {
        let mut log = self.lock();
        let sequence = log.next;
        log.next = log.next.wrapping_add(1);
        if log.events.len() == CAPACITY {
            log.events.pop_front();
        }
        log.events.push_back(ControlEvent { sequence, kind });
        self.changed.notify_all();
    }

    pub(super) fn subscribe(&self) -> ControlEvents<'_> {
        ControlEvents {
            events: self,
            next: self.lock().next,
        }
    }
}

/// A borrowed independent cursor. Subscribe before reading initial snapshots;
/// events invalidate views, so an event may already be reflected in a snapshot.
/// Dropping a cursor does not cancel or retain any execution owner.
pub struct ControlEvents<'a> {
    events: &'a Events,
    next: u64,
}

impl ControlEvents<'_> {
    /// Wait at most 30s (zero is nonblocking). A lagging subscriber never
    /// blocks the action owner and receives an explicit resnapshot signal.
    pub fn poll(&mut self, timeout: Duration) -> ControlPoll {
        let (log, _) = self
            .events
            .changed
            .wait_timeout_while(
                self.events.lock(),
                timeout.min(Duration::from_secs(30)),
                |log| log.next == self.next,
            )
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pending = log.next.wrapping_sub(self.next);
        if pending == 0 {
            return ControlPoll::Pending;
        }
        if pending > log.events.len() as u64 {
            self.next = log.next;
            return ControlPoll::Resync { missed: pending };
        }
        let event = log.events[log.events.len() - pending as usize];
        self.next = self.next.wrapping_add(1);
        ControlPoll::Event(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_bounded_cursors_and_explicit_resync() {
        let events = Events::default();
        let mut fast = events.subscribe();
        let mut slow = events.subscribe();
        for i in 0..CAPACITY + 1 {
            let kind = ControlEventKind::RequestAccepted {
                registration_id: i as u64,
            };
            events.publish(kind);
            assert_eq!(
                fast.poll(Duration::ZERO),
                ControlPoll::Event(ControlEvent {
                    sequence: i as u64,
                    kind
                })
            );
        }
        assert_eq!(
            slow.poll(Duration::ZERO),
            ControlPoll::Resync {
                missed: (CAPACITY + 1) as u64
            }
        );
        assert_eq!(slow.poll(Duration::ZERO), ControlPoll::Pending);
        let kind = ControlEventKind::ModelLoadFinished { success: true };
        events.publish(kind);
        assert_eq!(slow.poll(Duration::ZERO), fast.poll(Duration::ZERO));
        assert_eq!(fast.poll(Duration::ZERO), ControlPoll::Pending);
    }
}
