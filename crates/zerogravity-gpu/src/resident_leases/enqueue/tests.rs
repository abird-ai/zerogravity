//! CPU-only contract used by Client::exclusive; no device/client/Ready fixture.
use cubecl_common::{
    device::{DeviceId, DeviceService, ServerUtilitiesHandle},
    device_handle::DeviceHandle,
};
use std::sync::{
    Arc,
    atomic::{AtomicU16, Ordering},
};

struct Probe {
    order: Vec<usize>,
}
impl DeviceService for Probe {
    fn init(_: DeviceId) -> Self {
        Self { order: Vec::new() }
    }
    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }
}
struct Shutdown(DeviceId);
impl Drop for Shutdown {
    fn drop(&mut self) {
        DeviceHandle::<Probe>::shutdown(self.0);
    }
}
struct Fixture {
    // Drop the last own handle before shutting down this unique mock runner.
    handle: DeviceHandle<Probe>,
    _shutdown: Shutdown,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU16 = AtomicU16::new(0);
        let id = DeviceId {
            type_id: u16::MAX,
            index_id: NEXT.fetch_add(1, Ordering::Relaxed),
        };
        Self {
            handle: DeviceHandle::new(id),
            _shutdown: Shutdown(id),
        }
    }
}

#[test]
fn borrowed_guard_and_nested_commands_share_runner_stream_and_order() {
    let fixture = Fixture::new();
    let client = fixture.handle.clone();
    let lock = async_lock::Mutex::new([0; 2]);
    let mut guard = cubecl::future::block_on(lock.lock());
    let caller = std::thread::current().id();
    let stream = cubecl_core::environment::stream::StreamId::current();
    let actor = fixture
        .handle
        .exclusive(|| {
            let actor = std::thread::current().id();
            assert_ne!(actor, caller);
            assert_eq!(
                cubecl_core::environment::stream::StreamId::current(),
                stream
            );
            client.submit(|probe| probe.order.push(1));
            let count = client
                .submit_blocking(|probe| {
                    assert_eq!(std::thread::current().id(), actor);
                    probe.order.push(2);
                    probe.order.len()
                })
                .unwrap();
            guard[0] = count;
            guard[1] = 7;
            actor
        })
        .unwrap();
    assert_ne!(actor, caller);
    assert_eq!(*guard, [2, 7]);
    assert_eq!(client.submit_blocking(|p| p.order.clone()).unwrap(), [1, 2]);
}

#[test]
fn nested_error_is_returned_once_without_retry_or_runner_loss() {
    let fixture = Fixture::new();
    let client = fixture.handle.clone();
    let mut calls = 0;
    let result = fixture
        .handle
        .exclusive(|| {
            calls += 1;
            client
                .submit_blocking(|probe| {
                    probe.order.push(9);
                    Err::<(), _>("actual original cut")
                })
                .unwrap()
        })
        .unwrap();
    assert_eq!(result, Err("actual original cut"));
    assert_eq!(calls, 1);
    assert_eq!(client.submit_blocking(|p| p.order.clone()).unwrap(), [9]);
}

#[test]
fn panic_finishes_borrowed_frame_before_returning_and_runner_survives() {
    let fixture = Fixture::new();
    let mut calls = 0;
    let result: Result<(), _> = fixture.handle.exclusive(|| {
        calls += 1;
        panic!("original command scope panic");
    });
    assert!(result.is_err());
    assert_eq!(calls, 1);
    assert_eq!(fixture.handle.exclusive(|| 17).unwrap(), 17);
}
