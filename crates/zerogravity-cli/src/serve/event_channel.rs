//! Charged, bounded event transport.
//!
//! This keeps the original `std::mpsc::sync_channel` API shape used by the
//! serving code, but the fixed-layout slots are allocated explicitly so the
//! router can admit their real storage before publishing a request.

use super::engine::ServiceError;
use super::pending::{ServingCharge, ServingHost, allocate_slots, checked_sum, slot_bytes};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use zerogravity::resource_arbiter::AllocationClass;

pub(super) mod mpsc {
    use super::*;

    pub struct SyncSender<T> {
        inner: Arc<Inner<T>>,
    }
    pub struct Receiver<T> {
        inner: Arc<Inner<T>>,
    }
    pub(super) struct Inner<T> {
        state: Mutex<State<T>>,
        ready: Condvar,
        // The exact slot/control envelope stays charged for the whole channel
        // lifetime, including clones retained by fallback error readers.
        _charge: Option<ServingCharge>,
    }
    struct State<T> {
        slots: Box<[Option<T>]>,
        head: usize,
        len: usize,
        senders: usize,
        receiver_closed: bool,
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum TrySendError<T> {
        Full(T),
        Disconnected(T),
    }
    #[cfg(test)]
    #[derive(Debug, PartialEq, Eq)]
    pub enum TryRecvError {
        Empty,
        Disconnected,
    }
    #[derive(Debug, PartialEq, Eq)]
    pub enum RecvTimeoutError {
        Timeout,
        Disconnected,
    }

    pub(super) fn channel_bytes<T>(capacity: usize) -> Result<u64, ServiceError> {
        let slots = slot_bytes::<T>(capacity)?;
        let control = u64::try_from(
            std::mem::size_of::<Inner<T>>()
                .checked_add(2 * std::mem::size_of::<usize>())
                .and_then(|n| n.checked_add(std::mem::align_of::<Inner<T>>()))
                .ok_or_else(|| ServiceError::new(400, "size_overflow", "event size overflow"))?,
        )
        .map_err(|_| ServiceError::new(400, "size_overflow", "event size overflow"))?;
        checked_sum(&[slots, control])
    }

    #[cfg(test)]
    pub fn sync_channel<T>(capacity: usize) -> (SyncSender<T>, Receiver<T>) {
        build_channel(capacity, None).expect("event transport allocation")
    }
    /// Charged production transport. `owner`/`session` identify the request
    /// whose payload slots are admitted; the charge stays on `Inner`.
    pub(crate) fn try_sync_channel<T>(
        host: &ServingHost,
        owner: &str,
        session: &str,
        capacity: usize,
    ) -> Result<(SyncSender<T>, Receiver<T>), ServiceError> {
        let charge = host.charge(
            owner,
            session,
            AllocationClass::Scratch,
            channel_bytes::<T>(capacity)?,
        )?;
        build_channel(capacity, Some(charge))
    }
    fn build_channel<T>(
        capacity: usize,
        charge: Option<ServingCharge>,
    ) -> Result<(SyncSender<T>, Receiver<T>), ServiceError> {
        if capacity == 0 {
            return Err(ServiceError::new(
                400,
                "invalid_config",
                "event transport capacity must be positive",
            ));
        }
        let slots = allocate_slots::<T>(capacity)?;
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                slots,
                head: 0,
                len: 0,
                senders: 1,
                receiver_closed: false,
            }),
            ready: Condvar::new(),
            _charge: charge,
        });
        Ok((
            SyncSender {
                inner: Arc::clone(&inner),
            },
            Receiver { inner },
        ))
    }
    impl<T> Clone for SyncSender<T> {
        fn clone(&self) -> Self {
            let mut state = self.inner.state.lock().expect("event channel");
            state.senders += 1;
            Self {
                inner: Arc::clone(&self.inner),
            }
        }
    }
    impl<T> SyncSender<T> {
        pub fn try_send(&self, item: T) -> Result<(), TrySendError<T>> {
            let mut state = self.inner.state.lock().expect("event channel");
            if state.receiver_closed {
                return Err(TrySendError::Disconnected(item));
            }
            let capacity = state.slots.len();
            if state.len == capacity {
                return Err(TrySendError::Full(item));
            }
            let index = (state.head + state.len) % capacity;
            state.slots[index] = Some(item);
            state.len += 1;
            self.inner.ready.notify_one();
            Ok(())
        }
    }
    impl<T> Drop for SyncSender<T> {
        fn drop(&mut self) {
            let mut state = self
                .inner
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.senders = state.senders.saturating_sub(1);
            if state.senders == 0 {
                self.inner.ready.notify_all();
            }
        }
    }
    impl<T> Receiver<T> {
        fn pop(state: &mut State<T>) -> Option<T> {
            if state.len == 0 {
                return None;
            }
            let capacity = state.slots.len();
            let index = state.head;
            let value = state.slots[index].take();
            state.head = (state.head + 1) % capacity;
            state.len -= 1;
            value
        }
        #[cfg(test)]
        pub fn try_recv(&self) -> Result<T, TryRecvError> {
            let mut state = self.inner.state.lock().expect("event channel");
            if let Some(value) = Self::pop(&mut state) {
                self.inner.ready.notify_one();
                return Ok(value);
            }
            if state.senders == 0 {
                Err(TryRecvError::Disconnected)
            } else {
                Err(TryRecvError::Empty)
            }
        }
        pub fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvTimeoutError> {
            let deadline = Instant::now() + timeout;
            let mut state = self.inner.state.lock().expect("event channel");
            loop {
                if let Some(value) = Self::pop(&mut state) {
                    self.inner.ready.notify_one();
                    return Ok(value);
                }
                if state.senders == 0 {
                    return Err(RecvTimeoutError::Disconnected);
                }
                let now = Instant::now();
                if now >= deadline {
                    return Err(RecvTimeoutError::Timeout);
                }
                let (next, _) = self
                    .inner
                    .ready
                    .wait_timeout(state, deadline - now)
                    .expect("event wait");
                state = next;
            }
        }
        #[cfg(test)]
        pub fn recv(&self) -> Result<T, ()> {
            let mut state = self.inner.state.lock().expect("event channel");
            loop {
                if let Some(value) = Self::pop(&mut state) {
                    self.inner.ready.notify_one();
                    return Ok(value);
                }
                if state.senders == 0 {
                    return Err(());
                }
                state = self.inner.ready.wait(state).expect("event wait");
            }
        }
    }
    impl<T> Drop for Receiver<T> {
        fn drop(&mut self) {
            let mut state = self
                .inner
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.receiver_closed = true;
            self.inner.ready.notify_all();
            // Retire each payload outside the channel lock: a payload may own
            // another sender or a host charge whose Drop re-enters this
            // channel or the serving ledger. `receiver_closed` already refuses
            // new sends, so the pop/reset cycle cannot lose a live item.
            loop {
                let value = Self::pop(&mut state);
                drop(state);
                match value {
                    Some(value) => drop(value),
                    None => return,
                }
                state = self
                    .inner
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }
}

/// Control storage for the cancel handle and its condition variable. These
/// are actual fixed-size synchronization objects, not a request heuristic.
pub(super) fn cancel_bytes() -> Result<u64, ServiceError> {
    let bytes = std::mem::size_of::<super::pending::CancelToken>()
        .checked_add(std::mem::size_of::<std::sync::atomic::AtomicBool>())
        .and_then(|n| n.checked_add(4 * std::mem::size_of::<usize>()))
        .and_then(|n| n.checked_add(std::mem::size_of::<Condvar>()))
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "cancel size overflow"))?;
    u64::try_from(bytes)
        .map_err(|_| ServiceError::new(400, "size_overflow", "cancel size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_retains_channel_charge_after_sender_drop() {
        let host = ServingHost::isolated(512).unwrap();
        let (sender, receiver) = mpsc::try_sync_channel::<u8>(&host, "test", "event", 2).unwrap();
        let charged = host.serving_bytes();
        drop(sender);
        assert_eq!(host.serving_bytes(), charged);
        assert_eq!(
            receiver.recv_timeout(Duration::from_millis(1)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        );
        drop(receiver);
        assert_eq!(host.serving_bytes(), 0);
    }

    #[test]
    fn sender_retains_channel_charge_until_last_owner_drops() {
        let host = ServingHost::isolated(512).unwrap();
        let (sender, receiver) = mpsc::try_sync_channel::<u8>(&host, "test", "event", 2).unwrap();
        let charged = host.serving_bytes();
        assert!(charged > 0);
        drop(receiver);
        assert_eq!(host.serving_bytes(), charged);
        assert!(sender.try_send(1).is_err());
        drop(sender);
        assert_eq!(host.serving_bytes(), 0);
    }

    #[test]
    fn receiver_retires_payloads_outside_the_channel_lock() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Probe {
            sender: Option<mpsc::SyncSender<Probe>>,
            reentered: Arc<AtomicBool>,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                let Some(sender) = self.sender.take() else {
                    return;
                };
                let placeholder = Probe {
                    sender: None,
                    reentered: Arc::clone(&self.reentered),
                };
                match sender.try_send(placeholder) {
                    Err(mpsc::TrySendError::Disconnected(placeholder)) => {
                        drop(placeholder);
                        self.reentered.store(true, Ordering::Release);
                    }
                    _ => panic!("receiver must close before retiring payloads"),
                }
            }
        }
        let host = ServingHost::isolated(512).unwrap();
        let (sender, receiver) =
            mpsc::try_sync_channel::<Probe>(&host, "test", "reentrant", 2).unwrap();
        let reentered = Arc::new(AtomicBool::new(false));
        assert!(
            sender
                .try_send(Probe {
                    sender: Some(sender.clone()),
                    reentered: Arc::clone(&reentered),
                })
                .is_ok()
        );
        // Under the old drop the receiver held the state lock while dropping
        // this payload, so the re-entrant send would deadlock.
        drop(receiver);
        assert!(reentered.load(Ordering::Acquire));
        assert!(host.serving_bytes() > 0);
        drop(sender);
        assert_eq!(host.serving_bytes(), 0);
    }
}
