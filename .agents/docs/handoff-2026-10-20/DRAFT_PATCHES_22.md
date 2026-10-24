# Draft Patches — volume 22

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/COMMAND_ADMISSION.patch

Original UTF-8 bytes: 56107; SHA256 `cabcf7252ddb985dc7e4dc4d113ef13ec57fe133d24a858ef9c77d3d4f522245`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/cubecl-common-0.11.0/src/device/base.rs b/cubecl-common-0.11.0/src/device/base.rs
--- a/cubecl-common-0.11.0/src/device/base.rs
+++ b/cubecl-common-0.11.0/src/device/base.rs
@@ -3,6 +3,12 @@
     cmp::Ordering,
 };
 use cubecl_environment::sync::Arc;
+
+// Public contract exports through device::base, preserving device_handle ABI.
+pub use super::handle::{
+    CommandAdmission, CommandAdmissionError, CommandKind, CommandParticipation, CommandRequest,
+    CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RunningCommandPermit,
+};
 
 /// The device id.
 #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
@@ -56,6 +62,21 @@
         Self: Sized;
     /// Get the service utilities.
     fn utilities(&self) -> ServerUtilitiesHandle;
+    /// Cached optional command bridge. Pure metadata; no driver/registry work.
+    /// Returning Some does not establish complete CUDA participation.
+    fn command_admission(&self) -> Option<Arc<dyn crate::device::handle::CommandAdmission>> {
+        None
+    }
+    /// Same cached hook, obtainable BEFORE transport/registry locks or init.
+    /// Must not initialize a device, acquire an actor, or discover a provider.
+    fn bootstrap_command_admission(
+        _device_id: DeviceId,
+    ) -> Option<Arc<dyn crate::device::handle::CommandAdmission>>
+    where
+        Self: Sized,
+    {
+        None
+    }
     /// Which pipeline stage this service runs on.
     ///
     /// Services on [`DeviceServiceStage::Upstream`] produce work ahead of time (e.g. autodiff graph
diff --git a/cubecl-common-0.11.0/src/device/handle/admission/tests.rs b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
new file mode 100644
--- /dev/null
+++ b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
@@ -0,0 +1,581 @@
+//! Host fixtures only. No CUDA bridge, physical identity, driver or device work.
+use super::*;
+use crate::device::ServerUtilitiesHandle;
+use crate::device::handle::{CallError, DeviceHandle, DeviceHandleSpec, ServiceCreationError};
+use core::any::{Any, TypeId};
+use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
+use std::cell::Cell;
+use std::sync::Mutex;
+use std::vec::Vec;
+
+#[derive(Clone, Copy)]
+struct Entered {
+    gate: usize,
+    service: ServiceId,
+    stream: Option<StreamId>,
+    capture: bool,
+}
+
+std::thread_local! {
+    static ENTERED: Cell<Option<Entered>> = const { Cell::new(None) };
+}
+
+#[derive(Clone, Copy)]
+struct Seen {
+    service: ServiceId,
+    stream: Option<StreamId>,
+    kind: CommandKind,
+    explicit_scope: bool,
+}
+
+#[derive(Default)]
+struct GateState {
+    seen: Mutex<Vec<Seen>>,
+    live: AtomicUsize,
+    entered: AtomicUsize,
+    retired: AtomicUsize,
+    discarded: AtomicUsize,
+    refuse: AtomicBool,
+    epoch_live: AtomicBool,
+}
+
+struct Gate(Arc<GateState>);
+
+// SAFETY: synthetic host-only ledger; no physical-completion claims. Calls
+// never wait, and entered scopes are exact test identities, not ambient IDs.
+unsafe impl CommandAdmission for Gate {
+    fn admit(
+        &self,
+        request: CommandRequest<'_>,
+    ) -> Result<OwnedCommandPermit, CommandAdmissionError> {
+        if self.0.refuse.load(Ordering::SeqCst) {
+            return Err(CommandAdmissionError::Fenced);
+        }
+        let gate = Arc::as_ptr(&self.0) as usize;
+        let mut capture = request.scope.is_some();
+        ENTERED.with(|slot| {
+            if let Some(entered) = slot.get() {
+                if entered.gate == gate && entered.capture {
+                    if entered.service != request.service || entered.stream != request.stream {
+                        return Err(CommandAdmissionError::ForeignScope);
+                    }
+                    capture = true;
+                }
+            }
+            Ok(())
+        })?;
+        self.0.seen.lock().unwrap().push(Seen {
+            service: request.service,
+            stream: request.stream,
+            kind: request.kind,
+            explicit_scope: request.scope.is_some(),
+        });
+        self.0.live.fetch_add(1, Ordering::SeqCst);
+        Ok(OwnedCommandPermit::new(Posted {
+            state: self.0.clone(),
+            identity: Entered {
+                gate,
+                service: request.service,
+                stream: request.stream,
+                capture,
+            },
+            entered: false,
+        }))
+    }
+}
+
+struct Posted {
+    state: Arc<GateState>,
+    identity: Entered,
+    entered: bool,
+}
+
+// SAFETY: transfers the same owned synthetic host count into a same-thread
+// non-Send guard; Drop only changes atomics and never blocks or touches a device.
+unsafe impl PostedCommandPermit for Posted {
+    fn enter(mut self: Box<Self>) -> Box<dyn RunningCommandPermit> {
+        let previous = ENTERED.with(|slot| slot.replace(Some(self.identity)));
+        self.entered = true;
+        self.state.entered.fetch_add(1, Ordering::SeqCst);
+        Box::new(Running {
+            state: self.state.clone(),
+            previous,
+            _not_send: core::marker::PhantomData,
+        })
+    }
+}
+
+impl Drop for Posted {
+    fn drop(&mut self) {
+        if !self.entered {
+            self.state.live.fetch_sub(1, Ordering::SeqCst);
+            self.state.discarded.fetch_add(1, Ordering::SeqCst);
+        }
+    }
+}
+
+struct Running {
+    state: Arc<GateState>,
+    previous: Option<Entered>,
+    _not_send: core::marker::PhantomData<std::rc::Rc<()>>,
+}
+
+// SAFETY: host-only; restores exactly this thread's prior scope and does not
+// declare any physical work complete.
+unsafe impl RunningCommandPermit for Running {}
+
+impl Drop for Running {
+    fn drop(&mut self) {
+        ENTERED.with(|slot| slot.set(self.previous));
+        self.state.live.fetch_sub(1, Ordering::SeqCst);
+        self.state.retired.fetch_add(1, Ordering::SeqCst);
+    }
+}
+
+struct Epoch(Arc<GateState>);
+
+// SAFETY: synthetic host epoch only; no complete-participation issuer.
+unsafe impl CommandScopeOwner for Epoch {
+    fn validate(&self, _: ServiceId, _: StreamId) -> Result<(), CommandAdmissionError> {
+        if self.0.epoch_live.load(Ordering::SeqCst) {
+            Ok(())
+        } else {
+            Err(CommandAdmissionError::Stale)
+        }
+    }
+}
+
+struct Service {
+    hook: Option<Arc<dyn CommandAdmission>>,
+    value: usize,
+}
+
+impl DeviceService for Service {
+    fn init(_: DeviceId) -> Self {
+        Self {
+            hook: None,
+            value: 0,
+        }
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        Arc::new(())
+    }
+    fn command_admission(&self) -> Option<Arc<dyn CommandAdmission>> {
+        self.hook.clone()
+    }
+}
+
+type Task = Box<dyn FnOnce(&mut dyn Any) + Send>;
+
+struct State {
+    device: DeviceId,
+    value: Mutex<Box<dyn Any + Send>>,
+    queue: Mutex<Vec<Task>>,
+    hook: Option<Arc<dyn CommandAdmission>>,
+}
+
+#[derive(Clone)]
+struct Deferred(Arc<State>);
+
+impl Deferred {
+    fn run_one(&self) {
+        let task = self.0.queue.lock().unwrap().remove(0);
+        let mut value = self.0.value.lock().unwrap();
+        task(&mut **value);
+    }
+}
+
+// Deterministic fake transport deliberately leaves Submit queued. It tests
+// ownership across posting, not the original actor's implementation/runtime.
+impl DeviceHandleSpec for Deferred {
+    const BLOCKING: bool = false;
+    fn insert<S: DeviceService>(
+        device: DeviceId,
+        service: S,
+    ) -> Result<Self, ServiceCreationError> {
+        let hook = service.command_admission();
+        Ok(Self(Arc::new(State {
+            device,
+            value: Mutex::new(Box::new(service)),
+            queue: Mutex::new(Vec::new()),
+            hook,
+        })))
+    }
+    fn new<S: DeviceService>(device: DeviceId) -> Self {
+        Self::insert(device, S::init(device)).unwrap()
+    }
+    fn device_id(&self) -> DeviceId {
+        self.0.device
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        Arc::new(())
+    }
+    fn command_admission(&self) -> Option<Arc<dyn CommandAdmission>> {
+        self.0.hook.clone()
+    }
+    fn flush_queue(&self) {}
+    fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
+        &self,
+        task: T,
+    ) -> Result<R, CallError> {
+        let mut value = self.0.value.lock().unwrap();
+        Ok(task(&mut **value))
+    }
+    fn submit<T: FnOnce(&mut dyn Any) + Send + 'static>(&self, task: T) {
+        self.0.queue.lock().unwrap().push(Box::new(task));
+    }
+    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
+        Ok(task())
+    }
+}
+
+type Handle = DeviceHandle<Service, Deferred>;
+
+fn make_handle(hook: Option<Arc<dyn CommandAdmission>>) -> Handle {
+    Handle::insert(
+        super::super::next_test_device_id(),
+        Service { hook, value: 0 },
+    )
+    .unwrap()
+}
+
+fn make_gate() -> (Arc<dyn CommandAdmission>, Arc<GateState>) {
+    let state = Arc::new(GateState::default());
+    state.epoch_live.store(true, Ordering::SeqCst);
+    (Arc::new(Gate(state.clone())), state)
+}
+
+fn scope(
+    hook: Arc<dyn CommandAdmission>,
+    state: Arc<GateState>,
+    handle: &Handle,
+    stream: StreamId,
+) -> CommandScope {
+    // SAFETY: exact test bridge + synthetic epoch, no hardware/capture claim.
+    unsafe { CommandScope::from_original_epoch(hook, handle.service_id(), stream, Epoch(state)) }
+}
+
+#[test]
+fn default_none_uses_original_calls_and_stays_unproven() {
+    let handle = make_handle(None);
+    handle.submit(|value| value.value = 7);
+    assert_eq!(handle.handle.0.queue.lock().unwrap().len(), 1);
+    handle.handle.run_one();
+    assert_eq!(handle.submit_blocking(|value| value.value).unwrap(), 7);
+    assert_eq!(handle.exclusive(|| 19).unwrap(), 19);
+    assert_eq!(
+        handle.command_participation(),
+        CommandParticipation::Unproven
+    );
+    assert_eq!(handle.command_stream, None);
+}
+
+#[test]
+fn queued_closure_owns_permit_until_actual_execution() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    handle.submit(|value| value.value += 1);
+    assert_eq!(state.live.load(Ordering::SeqCst), 1);
+    assert_eq!(state.entered.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 0);
+    handle.handle.run_one();
+    assert_eq!(state.entered.load(Ordering::SeqCst), 1);
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 1);
+    assert_eq!(handle.handle.0.queue.lock().unwrap().len(), 0);
+}
+
+#[test]
+fn discarded_queued_closure_retires_only_unentered_host_owner() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    handle.submit(|_| panic!("discarded task must never execute"));
+    handle.handle.0.queue.lock().unwrap().clear();
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.discarded.load(Ordering::SeqCst), 1);
+    assert_eq!(state.entered.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 0);
+}
+
+#[test]
+fn refusal_precedes_enqueue_and_callback() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    state.refuse.store(true, Ordering::SeqCst);
+    let called = AtomicBool::new(false);
+    let error = handle
+        .submit_blocking(|_| called.store(true, Ordering::SeqCst))
+        .unwrap_err();
+    assert_eq!(
+        error.message(),
+        Some(CommandAdmissionError::Fenced.message())
+    );
+    assert!(!called.load(Ordering::SeqCst));
+    assert!(handle.handle.0.queue.lock().unwrap().is_empty());
+    assert!(state.seen.lock().unwrap().is_empty());
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+}
+
+#[test]
+fn unwind_restores_entered_scope_and_retires_host_owner() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
+        let _ = handle.exclusive(|| panic!("fixture body"));
+    }));
+    assert!(result.is_err());
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 1);
+    ENTERED.with(|entered| assert!(entered.get().is_none()));
+}
+
+#[test]
+fn ambient_stream_never_manufactures_explicit_binding() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    StreamId { value: 1701 }.executes(|| handle.submit_blocking(|_| ()).unwrap());
+    let seen = state.seen.lock().unwrap();
+    assert_eq!(seen.len(), 1);
+    assert_eq!(seen[0].stream, None);
+    assert!(!seen[0].explicit_scope);
+    assert_eq!(seen[0].service, handle.service_id());
+    assert_eq!(seen[0].kind, CommandKind::SubmitBlocking);
+}
+
+#[test]
+fn exact_scope_checks_unbound_foreign_hook_service_stream_and_stale_epoch() {
+    let (hook, state) = make_gate();
+    let mut handle = make_handle(Some(hook.clone()));
+    let stream = StreamId { value: 1702 };
+    let mut owner = scope(hook.clone(), state.clone(), &handle, stream);
+    assert!(
+        handle
+            .exclusive_with_command_scope(&mut owner, || ())
+            .is_err()
+    );
+    // SAFETY: this synthetic handle issues no device/stream operations.
+    unsafe { handle.set_command_stream(stream) };
+    let (foreign, _) = make_gate();
+    assert!(
+        owner
+            .validate(&foreign, handle.service_id(), Some(stream))
+            .is_err()
+    );
+    assert!(
+        owner
+            .validate(
+                &hook,
+                ServiceId {
+                    device: handle.device_id(),
+                    service: TypeId::of::<()>()
+                },
+                Some(stream)
+            )
+            .is_err()
+    );
+    assert!(
+        owner
+            .validate(&hook, handle.service_id(), Some(StreamId { value: 1703 }))
+            .is_err()
+    );
+    state.epoch_live.store(false, Ordering::SeqCst);
+    let error = handle
+        .exclusive_with_command_scope(&mut owner, || ())
+        .unwrap_err();
+    assert_eq!(
+        error.message(),
+        Some(CommandAdmissionError::Stale.message())
+    );
+    assert!(state.seen.lock().unwrap().is_empty());
+}
+
+#[test]
+fn original_exclusive_entry_transports_explicit_scope_and_allows_exact_nested_calls() {
+    let (hook, state) = make_gate();
+    let mut handle = make_handle(Some(hook.clone()));
+    let stream = StreamId { value: 1704 };
+    // SAFETY: synthetic host commands only; no stream driver operations.
+    unsafe { handle.set_command_stream(stream) };
+    let mut owner = scope(hook, state.clone(), &handle, stream);
+    handle
+        .exclusive_with_command_scope(&mut owner, || {
+            ENTERED.with(|entered| assert!(entered.get().unwrap().capture));
+            handle.submit_blocking(|value| value.value = 9).unwrap();
+            assert_eq!(state.live.load(Ordering::SeqCst), 1);
+        })
+        .unwrap();
+    let seen = state.seen.lock().unwrap();
+    assert_eq!(seen.len(), 2);
+    assert!(seen[0].explicit_scope);
+    assert!(!seen[1].explicit_scope);
+    assert!(
+        seen.iter()
+            .all(|r| r.service == handle.service_id() && r.stream == Some(stream))
+    );
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 2);
+}
+
+#[test]
+fn clone_and_seen_as_preserve_exact_service_stream_and_cached_hook() {
+    let (hook, state) = make_gate();
+    let mut handle = make_handle(Some(hook));
+    let stream = StreamId { value: 1705 };
+    // SAFETY: host-only synthetic commands.
+    unsafe { handle.set_command_stream(stream) };
+    let erased = handle.clone().seen_as::<dyn Any>(|value| value);
+    erased
+        .submit_blocking(|value| value.downcast_mut::<Service>().unwrap().value = 11)
+        .unwrap();
+    assert_eq!(erased.service_id(), handle.service_id());
+    let seen = state.seen.lock().unwrap();
+    assert_eq!(seen[0].service, handle.service_id());
+    assert_eq!(seen[0].stream, Some(stream));
+    assert_eq!(
+        erased.command_participation(),
+        CommandParticipation::Unproven
+    );
+}
+
+#[test]
+fn unsafe_registered_hook_does_not_issue_complete_participation() {
+    let (hook, _) = make_gate();
+    let handle = make_handle(Some(hook));
+    assert_eq!(
+        handle.command_participation(),
+        CommandParticipation::Unproven
+    );
+}
+
+// The registry below is ONLY fixture input for the three original transports.
+// It is not a provider discovery mechanism or a second physical queue.
+struct Registered {
+    hook: Arc<dyn CommandAdmission>,
+    state: Arc<GateState>,
+}
+
+static REGISTERED: std::sync::OnceLock<Mutex<std::collections::HashMap<DeviceId, Registered>>> =
+    std::sync::OnceLock::new();
+
+fn registered() -> &'static Mutex<std::collections::HashMap<DeviceId, Registered>> {
+    REGISTERED.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
+}
+
+struct OriginalService {
+    hook: Arc<dyn CommandAdmission>,
+    state: Arc<GateState>,
+    value: usize,
+}
+
+impl DeviceService for OriginalService {
+    fn init(device: DeviceId) -> Self {
+        ENTERED.with(|entered| {
+            let entered = entered.get().expect("actual constructor must be prepaid");
+            assert_eq!(entered.service, ServiceId::of::<Self>(device));
+            assert!(!entered.capture);
+        });
+        let map = registered().lock().unwrap();
+        let entry = map.get(&device).unwrap();
+        Self {
+            hook: entry.hook.clone(),
+            state: entry.state.clone(),
+            value: 0,
+        }
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        self.state.clone()
+    }
+    fn command_admission(&self) -> Option<Arc<dyn CommandAdmission>> {
+        Some(self.hook.clone())
+    }
+    fn bootstrap_command_admission(device: DeviceId) -> Option<Arc<dyn CommandAdmission>> {
+        Some(
+            registered()
+                .lock()
+                .unwrap()
+                .get(&device)
+                .unwrap()
+                .hook
+                .clone(),
+        )
+    }
+}
+
+fn original_handle<I: DeviceHandleSpec>(device: DeviceId) -> DeviceHandle<OriginalService, I> {
+    let (hook, state) = make_gate();
+    assert!(
+        registered()
+            .lock()
+            .unwrap()
+            .insert(device, Registered { hook, state })
+            .is_none()
+    );
+    DeviceHandle::new(device)
+}
+
+fn exercise_original_transport<I: DeviceHandleSpec + Sync>() {
+    let fixture = super::super::DeviceFixture::new(
+        original_handle::<I>,
+        DeviceHandle::<OriginalService, I>::shutdown,
+    );
+    // The reentrant transport initializes lazily at this exact original call.
+    let state = fixture
+        .submit_blocking(|service| service.state.clone())
+        .unwrap();
+    let hook = registered()
+        .lock()
+        .unwrap()
+        .get(&fixture.device_id())
+        .unwrap()
+        .hook
+        .clone();
+    let mut pinned: DeviceHandle<OriginalService, I> = (*fixture).clone();
+    let stream = StreamId { value: 1706 };
+    // SAFETY: host-only service; no CUDA or logical driver stream is created.
+    unsafe { pinned.set_command_stream(stream) };
+    // SAFETY: synthetic epoch with same fixture hook/service/explicit stream.
+    let mut owner = unsafe {
+        CommandScope::from_original_epoch(hook, pinned.service_id(), stream, Epoch(state.clone()))
+    };
+    pinned
+        .exclusive_with_command_scope(&mut owner, || {
+            pinned
+                .submit_blocking(|service| service.value = 23)
+                .unwrap();
+            assert_eq!(pinned.submit_blocking(|service| service.value).unwrap(), 23);
+        })
+        .unwrap();
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert!(
+        state
+            .seen
+            .lock()
+            .unwrap()
+            .iter()
+            .any(|seen| seen.kind == CommandKind::LocateOrInitialize)
+    );
+    assert_eq!(
+        pinned.command_participation(),
+        CommandParticipation::Unproven
+    );
+    registered()
+        .lock()
+        .unwrap()
+        .remove(&fixture.device_id())
+        .unwrap();
+}
+
+#[test]
+fn original_channel_transports_owned_bootstrap_and_explicit_reentry() {
+    exercise_original_transport::<super::super::channel::ChannelDeviceHandle>();
+}
+
+#[test]
+fn original_mutex_transports_owned_bootstrap_and_explicit_reentry() {
+    exercise_original_transport::<super::super::mutex::MutexDeviceHandle>();
+}
+
+#[test]
+fn original_reentrant_transports_lazy_bootstrap_and_explicit_reentry() {
+    exercise_original_transport::<super::super::reentrant::ReentrantMutexDeviceHandle>();
+}
diff --git a/cubecl-common-0.11.0/src/device/handle/admission.rs b/cubecl-common-0.11.0/src/device/handle/admission.rs
new file mode 100644
--- /dev/null
+++ b/cubecl-common-0.11.0/src/device/handle/admission.rs
@@ -0,0 +1,265 @@
+//! Optional host-command ownership; never a device-completion certificate.
+use crate::device::{DeviceId, DeviceService, ServiceId};
+use alloc::boxed::Box;
+use core::fmt;
+use cubecl_environment::{stream::StreamId, sync::Arc};
+
+/// No complete CUDA participation issuer exists in this transport slice.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum CommandParticipation {
+    /// Constructors, external events/storage and final retirement are unproven.
+    Unproven,
+}
+
+/// The original actor operation, before entering its transport or registry lock.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum CommandKind {
+    /// May create the service; an inserted service was constructed earlier.
+    LocateOrInitialize,
+    /// Install an already constructed service; does not cover its construction.
+    Install,
+    /// A posted closure, including its captures, until host execution finishes.
+    Submit,
+    /// A host-blocking actor call, not a device synchronization guarantee.
+    SubmitBlocking,
+    /// The original actor's reentrant exclusive closure.
+    Exclusive,
+}
+
+/// An admission refusal occurs before enqueue, never after a command is posted.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum CommandAdmissionError {
+    /// No audited bridge for the requested operation.
+    Unproven,
+    /// Hook, service or explicit logical stream differs.
+    ForeignScope,
+    /// Original physical queue epoch was revoked.
+    Stale,
+    /// Original physical card is fenced.
+    Fenced,
+    /// The original request was cancelled before admission.
+    Cancelled,
+}
+
+impl CommandAdmissionError {
+    pub(crate) const fn message(self) -> &'static str {
+        match self {
+            Self::Unproven => "device command admission: participation unproven",
+            Self::ForeignScope => "device command admission: foreign capture scope",
+            Self::Stale => "device command admission: stale original queue epoch",
+            Self::Fenced => "device command admission: original physical card fenced",
+            Self::Cancelled => "device command admission: cancelled before enqueue",
+        }
+    }
+}
+
+impl fmt::Display for CommandAdmissionError {
+    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+        f.write_str(self.message())
+    }
+}
+
+/// Exact original service and optional explicitly pinned Client stream.
+pub struct CommandRequest<'a> {
+    /// Actual transport registry service; not a PCI-card identity.
+    pub service: ServiceId,
+    /// `None` stays unbound; no inference from the caller's ambient StreamId.
+    pub stream: Option<StreamId>,
+    /// Host operation being admitted.
+    pub kind: CommandKind,
+    /// Supplied only by an explicit original pinned-client scope.
+    pub scope: Option<&'a CommandScope>,
+}
+
+/// A cached bridge into the SAME original physical QueueState/ResourceArbiter.
+///
+/// # Safety
+///
+/// Admission may wait only before enqueue/registry lock. Temporary capture close
+/// parks an ordinary caller rather than rejecting previously accepted work. A
+/// refusal must imply that neither enqueue nor driver effects occurred. Entered
+/// original commands may make nested actor calls without waiting for their own
+/// permit to drain. Capture reentry requires the explicit validated scope, never
+/// merely a thread/stream ID. No new physical registry, scheduler or byte ledger.
+/// All callbacks, hook destruction and permit Drops must avoid waiting on
+/// actor/drop work needed by drain. This first slice MUST NOT register a bridge
+/// that closes admission: external events/storage/final destruction still lack
+/// pre-admitted retirement. A later audited bridge may implement the waiting
+/// contract above only AFTER all participation is integrated and qualified.
+/// This interface alone does NOT cover device-terminal ownership.
+pub unsafe trait CommandAdmission: Send + Sync + 'static {
+    /// Returns an owned host-command permit, before any transport enqueue.
+    fn admit(
+        &self,
+        request: CommandRequest<'_>,
+    ) -> Result<OwnedCommandPermit, CommandAdmissionError>;
+}
+
+/// Queued ownership moves into the actual original actor closure.
+pub struct OwnedCommandPermit {
+    owner: Box<dyn PostedCommandPermit>,
+}
+
+impl OwnedCommandPermit {
+    /// Seal an original bridge's owned permit; no complete capability is issued.
+    pub fn new(owner: impl PostedCommandPermit + 'static) -> Self {
+        Self {
+            owner: Box::new(owner),
+        }
+    }
+
+    pub(super) fn owner_enter_for_transport(self) -> Box<dyn RunningCommandPermit> {
+        self.owner.enter()
+    }
+
+    pub(crate) fn run<R>(self, task: impl FnOnce() -> R) -> R {
+        let _entered = self.owner.enter();
+        task()
+    }
+}
+
+/// Owner retained while queued, including transport discard/panic paths.
+///
+/// # Safety
+///
+/// Drop is nonblocking and performs no driver calls. `enter` is nonblocking,
+/// infallible for admitted work, and establishes only the original command's
+/// actor-local reentry, including an explicitly admitted capture scope if any.
+/// It transfers all ownership into the returned guard. Unexpected enter/panic
+/// must retain/fence through the original queue rather than declare completion.
+pub unsafe trait PostedCommandPermit: Send {
+    /// Called on the original execution thread, not on the posting thread.
+    fn enter(self: Box<Self>) -> Box<dyn RunningCommandPermit>;
+}
+
+/// Host execution owner; intentionally not Send or a hardware Ready event.
+///
+/// # Safety
+///
+/// Drop is nonblocking, releases actor-local scope on the same thread, and
+/// retires only HOST-command ownership. Any posted CUDA work retains a separate
+/// original flight/event owner until known terminal completion or quarantine.
+pub unsafe trait RunningCommandPermit {}
+
+/// Original capture-epoch owner; no implementation is supplied here.
+///
+/// # Safety
+///
+/// Own the actual noncloneable capture epoch/ticket and original physical key.
+/// Validation and Drop are nonblocking; revoked/unknown work stays under the
+/// original queue fence/quarantine. Caller-supplied IDs alone cannot implement
+/// this contract. An admitted permit must retain the actual epoch through its
+/// entire queued/executing lifetime, not borrow this object past `admit`.
+pub unsafe trait CommandScopeOwner: Send + Sync {
+    /// Validate actual epoch and exact service/stream, before enqueue.
+    fn validate(&self, service: ServiceId, stream: StreamId) -> Result<(), CommandAdmissionError>;
+}
+
+/// Move-only authority supplied by an original pinned Client/capture epoch.
+/// No safe/public complete-participation factory exists in this slice.
+pub struct CommandScope {
+    admission: Arc<dyn CommandAdmission>,
+    service: ServiceId,
+    stream: StreamId,
+    owner: Box<dyn CommandScopeOwner>,
+}
+
+impl CommandScope {
+    /// Bridge an already-owned original epoch; does not grant CUDA eligibility.
+    ///
+    /// # Safety
+    ///
+    /// `owner` must own the same original QueueState/key/epoch used by
+    /// `admission`, and service/stream must come from the actual pinned Client.
+    /// All same-card lifecycle participation remains a separate CLOSED gate.
+    pub unsafe fn from_original_epoch(
+        admission: Arc<dyn CommandAdmission>,
+        service: ServiceId,
+        stream: StreamId,
+        owner: impl CommandScopeOwner + 'static,
+    ) -> Self {
+        Self {
+            admission,
+            service,
+            stream,
+            owner: Box::new(owner),
+        }
+    }
+
+    /// Exact original service; metadata alone is not physical ownership proof.
+    pub fn service_id(&self) -> ServiceId {
+        self.service
+    }
+
+    /// Explicit pinned Client logical stream, never ambient StreamId::current.
+    pub fn stream_id(&self) -> StreamId {
+        self.stream
+    }
+
+    pub(crate) fn validate(
+        &self,
+        admission: &Arc<dyn CommandAdmission>,
+        service: ServiceId,
+        stream: Option<StreamId>,
+    ) -> Result<(), CommandAdmissionError> {
+        if !Arc::ptr_eq(&self.admission, admission)
+            || self.service != service
+            || stream != Some(self.stream)
+        {
+            return Err(CommandAdmissionError::ForeignScope);
+        }
+        self.owner.validate(service, self.stream)
+    }
+}
+
+pub(super) fn admit(
+    hook: Option<&Arc<dyn CommandAdmission>>,
+    request: CommandRequest<'_>,
+) -> Result<Option<OwnedCommandPermit>, CommandAdmissionError> {
+    if let Some(scope) = request.scope {
+        scope.validate(
+            hook.ok_or(CommandAdmissionError::Unproven)?,
+            request.service,
+            request.stream,
+        )?;
+    }
+    hook.map(|hook| hook.admit(request)).transpose()
+}
+
+pub(super) fn bootstrap<S: DeviceService>(
+    device: DeviceId,
+    kind: CommandKind,
+) -> Result<
+    (
+        Option<Arc<dyn CommandAdmission>>,
+        Option<OwnedCommandPermit>,
+    ),
+    CommandAdmissionError,
+> {
+    // MUST be called before CHANNELS/GLOBAL/DEVICE_REGISTRY or actor locks.
+    let hook = S::bootstrap_command_admission(device);
+    let permit = admit(
+        hook.as_ref(),
+        CommandRequest {
+            service: ServiceId::of::<S>(device),
+            stream: None,
+            kind,
+            scope: None,
+        },
+    )?;
+    Ok((hook, permit))
+}
+
+pub(super) fn same_hook(
+    expected: &Option<Arc<dyn CommandAdmission>>,
+    actual: &Option<Arc<dyn CommandAdmission>>,
+) -> bool {
+    match (expected, actual) {
+        (None, None) => true,
+        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
+        _ => false,
+    }
+}
+
+#[cfg(test)]
+mod tests;
diff --git a/cubecl-common-0.11.0/src/device/handle/base.rs b/cubecl-common-0.11.0/src/device/handle/base.rs
--- a/cubecl-common-0.11.0/src/device/handle/base.rs
+++ b/cubecl-common-0.11.0/src/device/handle/base.rs
@@ -7,12 +7,13 @@
 ///
 /// When a task panics on a device's runner thread, the panic payload is
 /// captured and carried here instead of being logged and dropped.
-/// A `CallError` without a payload means the runner channel disconnected
-/// before the task could produce a result.
+/// A `CallError` without a payload or admission refusal means the runner
+/// channel disconnected before the task could produce a result.
 pub struct CallError {
     /// The captured panic payload, or `None` if the call failed because the
     /// runner channel disconnected
     payload: Option<Box<dyn Any + Send>>,
+    admission: Option<super::CommandAdmissionError>,
 }
 
 impl CallError {
@@ -23,6 +24,7 @@
     pub(crate) fn from_panic(payload: Box<dyn Any + Send>) -> Self {
         Self {
             payload: Some(payload),
+            admission: None,
         }
     }
 
@@ -30,12 +32,31 @@
     /// a result (no panic payload available).
     #[cfg_attr(not(multi_threading), allow(dead_code))]
     pub(crate) fn disconnected() -> Self {
-        Self { payload: None }
+        Self {
+            payload: None,
+            admission: None,
+        }
+    }
+
+    /// Refusal before enqueue, distinct from actor panic or disconnection.
+    pub(crate) fn from_admission(error: super::CommandAdmissionError) -> Self {
+        Self {
+            payload: None,
+            admission: Some(error),
+        }
+    }
+
+    /// Typed pre-enqueue refusal, if this call never reached the actor.
+    pub fn admission_error(&self) -> Option<super::CommandAdmissionError> {
+        self.admission
     }
 
     /// Returns the panic message if the failure was a panic whose payload is a
     /// string (the common case for `panic!`, indexing, `unwrap`, ...).
     pub fn message(&self) -> Option<&str> {
+        if let Some(error) = self.admission {
+            return Some(error.message());
+        }
         let payload = self.payload.as_ref()?;
         if let Some(s) = payload.downcast_ref::<&'static str>() {
             Some(*s)
@@ -51,7 +72,10 @@
     /// The payload can be handed to [`std::panic::resume_unwind`] to re-raise
     /// the original panic on the caller's thread.
     pub fn into_panic(self) -> Option<Box<dyn Any + Send>> {
-        self.payload
+        match self.admission {
+            Some(error) => Some(Box::new(error.message())),
+            None => self.payload,
+        }
     }
 
     /// Re-raises the failure on the current thread, diverging.
@@ -91,6 +115,9 @@
 
 impl core::fmt::Debug for CallError {
     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
+        if let Some(error) = self.admission {
+            return write!(f, "CallError(command refused before enqueue: {error})");
+        }
         match self.message() {
             Some(message) => write!(
                 f,
@@ -140,6 +167,13 @@
     /// Retrieves the server utilities for this thread.
     fn utilities(&self) -> ServerUtilitiesHandle;
 
+    /// Cached metadata only: do not acquire the service actor or initialize it.
+    fn command_admission(
+        &self,
+    ) -> Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>> {
+        None
+    }
+
     /// Doesn't flush the service state, but flushes any task enqueued in the communication
     /// channel.
     ///
diff --git a/cubecl-common-0.11.0/src/device/handle/channel.rs b/cubecl-common-0.11.0/src/device/handle/channel.rs
--- a/cubecl-common-0.11.0/src/device/handle/channel.rs
+++ b/cubecl-common-0.11.0/src/device/handle/channel.rs
@@ -62,6 +62,12 @@
 
     fn utilities(&self) -> ServerUtilitiesHandle {
         self.state.utilities()
+    }
+
+    fn command_admission(
+        &self,
+    ) -> Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>> {
+        self.state.service.command_admission.clone()
     }
 
     /// Runs `task` on the device thread, blocking until it returns.
@@ -240,6 +246,7 @@
 struct ChannelService {
     type_id: TypeId,
     utilities: ServerUtilitiesHandle,
+    command_admission: Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>>,
 }
 
 #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
@@ -298,6 +305,13 @@
         device_id: DeviceId,
         service: Option<S>,
     ) -> Result<Self, ServiceCreationError> {
+        let kind = if service.is_some() {
+            super::CommandKind::Install
+        } else {
+            super::CommandKind::LocateOrInitialize
+        };
+        let (bootstrap_hook, bootstrap_permit) = super::admission::bootstrap::<S>(device_id, kind)
+            .map_err(|e| ServiceCreationError::new(e.message().into()))?;
         let type_id = TypeId::of::<S>();
         let runner_id = RunnerId {
             device: device_id,
@@ -351,6 +365,11 @@
                     "Service already initialized.".into(),
                 ));
             }
+            if !super::admission::same_hook(&bootstrap_hook, &existing.service.command_admission) {
+                return Err(ServiceCreationError::new(
+                    "Cached command bridge differs from bootstrap".into(),
+                ));
+            }
             return Ok(existing.clone());
         }
 
@@ -369,28 +388,43 @@
 
         // The service initialization function.
         let initialize_service = move || {
-            STATES.with(|state| {
-                let mut map = match state.try_borrow_mut() {
-                    Ok(map) => map,
-                    Err(err) => panic!(
-                        "The device service {:?} is already borrowed: {err}",
-                        core::any::type_name::<S>()
-                    ),
-                };
-
-                if service.is_some() && map.contains_key(&type_id) {
-                    callback.send(Err(())).unwrap();
-                } else {
-                    let service = service.unwrap_or_else(|| S::init(device_id));
-                    let utilities = service.utilities();
-
-                    map.entry(type_id)
-                        .or_insert_with(|| RefCell::new(Box::new(service)));
-                    callback
-                        .send(Ok(ChannelService { type_id, utilities }))
-                        .unwrap();
-                }
-            });
+            let initialize = || {
+                STATES.with(|state| {
+                    let mut map = match state.try_borrow_mut() {
+                        Ok(map) => map,
+                        Err(err) => panic!(
+                            "The device service {:?} is already borrowed: {err}",
+                            core::any::type_name::<S>()
+                        ),
+                    };
+
+                    if service.is_some() && map.contains_key(&type_id) {
+                        callback.send(Err(())).unwrap();
+                    } else {
+                        let service = service.unwrap_or_else(|| S::init(device_id));
+                        let utilities = service.utilities();
+                        let command_admission = service.command_admission();
+                        assert!(
+                            super::admission::same_hook(&bootstrap_hook, &command_admission),
+                            "Service command bridge differs from bootstrap"
+                        );
+
+                        map.entry(type_id)
+                            .or_insert_with(|| RefCell::new(Box::new(service)));
+                        callback
+                            .send(Ok(ChannelService {
+                                type_id,
+                                utilities,
+                                command_admission,
+                            }))
+                            .unwrap();
+                    }
+                })
+            };
+            match bootstrap_permit {
+                Some(permit) => permit.run(initialize),
+                None => initialize(),
+            }
         };
 
         // Same reason in [`send]` we need to call the function directly if we are on the runner
diff --git a/cubecl-common-0.11.0/src/device/handle/mod.rs b/cubecl-common-0.11.0/src/device/handle/mod.rs
--- a/cubecl-common-0.11.0/src/device/handle/mod.rs
+++ b/cubecl-common-0.11.0/src/device/handle/mod.rs
@@ -1,5 +1,10 @@
+mod admission;
 mod base;
 
+pub use admission::{
+    CommandAdmission, CommandAdmissionError, CommandKind, CommandParticipation, CommandRequest,
+    CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RunningCommandPermit,
+};
 pub use base::*;
 
 use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle, ServiceId};
@@ -34,6 +39,8 @@
     handle: I,
     service: ServiceId,
     cast: fn(&mut dyn Any) -> &mut S,
+    // Only Client::set_stream supplies this; None is never inferred as pinned.
+    command_stream: Option<cubecl_environment::stream::StreamId>,
 }
 
 impl<S: ?Sized, I: DeviceHandleSpec> Clone for DeviceHandle<S, I> {
@@ -42,6 +49,7 @@
             handle: self.handle.clone(),
             service: self.service,
             cast: self.cast,
+            command_stream: self.command_stream,
         }
     }
 }
@@ -60,6 +68,7 @@
             handle: I::insert::<S>(device_id, service)?,
             service: ServiceId::of::<S>(device_id),
             cast: downcast::<S>,
+            command_stream: None,
         })
     }
 
@@ -68,6 +77,7 @@
             handle: I::new::<S>(device_id),
             service: ServiceId::of::<S>(device_id),
             cast: downcast::<S>,
+            command_stream: None,
         }
     }
 }
@@ -85,6 +95,7 @@
             handle: self.handle,
             service: self.service,
             cast,
+            command_stream: self.command_stream,
         }
     }
 
@@ -107,12 +118,33 @@
         task: T,
     ) -> Result<R, CallError> {
         let cast = self.cast;
-        self.handle.submit_blocking(move |state| task(cast(state)))
+        let hook = self.handle.command_admission();
+        if hook.is_none() {
+            return self.handle.submit_blocking(move |state| task(cast(state)));
+        }
+        let permit = self.command_permit(hook.as_ref(), CommandKind::SubmitBlocking, None)?;
+        self.handle.submit_blocking(move |state| {
+            permit
+                .expect("registered hook returns owned permit")
+                .run(|| task(cast(state)))
+        })
     }
 
     pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
         let cast = self.cast;
-        self.handle.submit(move |state| task(cast(state)))
+        let hook = self.handle.command_admission();
+        if hook.is_none() {
+            self.handle.submit(move |state| task(cast(state)));
+            return;
+        }
+        let permit = self
+            .command_permit(hook.as_ref(), CommandKind::Submit, None)
+            .unwrap_or_resume();
+        self.handle.submit(move |state| {
+            permit
+                .expect("registered hook returns owned permit")
+                .run(|| task(cast(state)))
+        })
     }
 
     pub fn flush_queue(&self) {
@@ -120,7 +152,63 @@
     }
 
     pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
-        self.handle.exclusive(task)
+        let hook = self.handle.command_admission();
+        if hook.is_none() {
+            return self.handle.exclusive(task);
+        }
+        let permit = self.command_permit(hook.as_ref(), CommandKind::Exclusive, None)?;
+        self.handle.exclusive(move || {
+            permit
+                .expect("registered hook returns owned permit")
+                .run(task)
+        })
+    }
+
+    /// This first transport hook never issues complete CUDA participation.
+    pub fn command_participation(&self) -> CommandParticipation {
+        CommandParticipation::Unproven
+    }
+
+    /// Mirror the existing Client's explicit logical stream; not an authority.
+    ///
+    /// # Safety
+    /// Must match the actual stream argument on every command from this handle.
+    pub unsafe fn set_command_stream(&mut self, stream: cubecl_environment::stream::StreamId) {
+        self.command_stream = Some(stream);
+    }
+
+    /// Run on the original actor with explicit original capture-epoch ownership.
+    /// Does not begin capture or grant complete same-card participation.
+    pub fn exclusive_with_command_scope<R: Send, T: FnOnce() -> R + Send>(
+        &self,
+        scope: &mut CommandScope,
+        task: T,
+    ) -> Result<R, CallError> {
+        let hook = self.handle.command_admission();
+        let permit = self.command_permit(hook.as_ref(), CommandKind::Exclusive, Some(scope))?;
+        self.handle.exclusive(move || {
+            permit
+                .expect("validated scope requires registered hook")
+                .run(task)
+        })
+    }
+
+    fn command_permit(
+        &self,
+        hook: Option<&cubecl_environment::sync::Arc<dyn CommandAdmission>>,
+        kind: CommandKind,
+        scope: Option<&CommandScope>,
+    ) -> Result<Option<OwnedCommandPermit>, CallError> {
+        admission::admit(
+            hook,
+            CommandRequest {
+                service: self.service,
+                stream: self.command_stream,
+                kind,
+                scope,
+            },
+        )
+        .map_err(CallError::from_admission)
     }
 
     /// Stops the background runner threads for `device_id`, blocking until they
diff --git a/cubecl-common-0.11.0/src/device/handle/mutex.rs b/cubecl-common-0.11.0/src/device/handle/mutex.rs
--- a/cubecl-common-0.11.0/src/device/handle/mutex.rs
+++ b/cubecl-common-0.11.0/src/device/handle/mutex.rs
@@ -19,6 +19,7 @@
     service: Arc<Mutex<Box<dyn Any + Send>>>,
     device_id: DeviceId,
     utilities: ServerUtilitiesHandle,
+    command_admission: Option<Arc<dyn super::CommandAdmission>>,
 }
 
 /// The global storage for all device services.
@@ -33,6 +34,10 @@
     const BLOCKING: bool = true;
 
     fn new<S: DeviceService>(device_id: DeviceId) -> Self {
+        let (bootstrap_hook, bootstrap_permit) =
+            super::admission::bootstrap::<S>(device_id, super::CommandKind::LocateOrInitialize)
+                .unwrap_or_else(|e| panic!("{e}"));
+        let _entered = bootstrap_permit.map(|permit| permit.owner_enter_for_transport());
         let mut guard = DEVICE_REGISTRY.lock();
         if guard.is_none() {
             *guard = Some(HashMap::new());
@@ -49,13 +54,23 @@
             .or_insert_with(|| {
                 let state = S::init(device_id);
                 let utilities = state.utilities();
+                let command_admission = state.command_admission();
+                assert!(
+                    super::admission::same_hook(&bootstrap_hook, &command_admission),
+                    "Service command bridge differs from bootstrap"
+                );
                 MutexDeviceState {
                     service: Arc::new(Mutex::new(Box::new(state))),
                     device_id,
                     utilities,
+                    command_admission,
                 }
             })
             .clone();
+        assert!(
+            super::admission::same_hook(&bootstrap_hook, &state.command_admission),
+            "Cached command bridge differs from bootstrap"
+        );
 
         Self { state, device_id }
     }
@@ -66,6 +81,10 @@
 
     fn utilities(&self) -> ServerUtilitiesHandle {
         self.state.utilities.clone()
+    }
+
+    fn command_admission(&self) -> Option<Arc<dyn super::CommandAdmission>> {
+        self.state.command_admission.clone()
     }
 
     fn flush_queue(&self) {}
@@ -89,6 +108,10 @@
         device_id: DeviceId,
         service: S,
     ) -> Result<Self, ServiceCreationError> {
+        let (bootstrap_hook, bootstrap_permit) =
+            super::admission::bootstrap::<S>(device_id, super::CommandKind::Install)
+                .map_err(|e| ServiceCreationError::new(e.message().into()))?;
+        let _entered = bootstrap_permit.map(|permit| permit.owner_enter_for_transport());
         let mut guard = DEVICE_REGISTRY.lock();
         if guard.is_none() {
             *guard = Some(HashMap::new());
@@ -108,10 +131,16 @@
             .entry(type_id)
             .or_insert_with(|| {
                 let utilities = service.utilities();
+                let command_admission = service.command_admission();
+                assert!(
+                    super::admission::same_hook(&bootstrap_hook, &command_admission),
+                    "Service command bridge differs from bootstrap"
+                );
                 MutexDeviceState {
                     service: Arc::new(Mutex::new(Box::new(service))),
                     device_id,
                     utilities,
+                    command_admission,
                 }
             })
             .clone();
diff --git a/cubecl-common-0.11.0/src/device/handle/reentrant.rs b/cubecl-common-0.11.0/src/device/handle/reentrant.rs
--- a/cubecl-common-0.11.0/src/device/handle/reentrant.rs
+++ b/cubecl-common-0.11.0/src/device/handle/reentrant.rs
@@ -18,15 +18,18 @@
     type_name: &'static str,
     /// Builds the service on first use, since the map is filled lazily.
     init: fn(DeviceId) -> ReentrantMutexDeviceState,
+    command_admission: Option<Arc<dyn super::CommandAdmission>>,
 }
 
 /// The state for a service of type `S` on `device_id`, built on first use.
 fn init_state<S: DeviceService>(device_id: DeviceId) -> ReentrantMutexDeviceState {
     let service = S::init(device_id);
     let utilities = service.utilities();
+    let command_admission = service.command_admission();
     ReentrantMutexDeviceState {
         service: Cell::new(Some(Box::new(service))),
         utilities,
+        command_admission,
     }
 }
 
@@ -59,6 +62,10 @@
             .clone()
     }
 
+    fn command_admission(&self) -> Option<Arc<dyn super::CommandAdmission>> {
+        self.command_admission.clone()
+    }
+
     fn flush_queue(&self) {}
 
     fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
@@ -88,6 +95,7 @@
             type_id: self.type_id,
             type_name: self.type_name,
             init: self.init,
+            command_admission: self.command_admission.clone(),
         }
     }
 }
@@ -121,6 +129,21 @@
         state_new: S,
     ) -> Result<Self, alloc::string::String> {
         let lock = Self::locate::<S>(device);
+        let install = super::admission::admit(
+            lock.command_admission.as_ref(),
+            super::CommandRequest {
+                service: crate::device::ServiceId::of::<S>(device),
+                stream: None,
+                kind: super::CommandKind::Install,
+                scope: None,
+            },
+        )
+        .map_err(|e| alloc::string::String::from(e.message()))?;
+        let _entered = install.map(|permit| permit.owner_enter_for_transport());
+        let actual_hook = state_new.command_admission();
+        if !super::admission::same_hook(&lock.command_admission, &actual_hook) {
+            return Err("Service command bridge differs from bootstrap".into());
+        }
         let id = TypeId::of::<S>();
 
         let state = lock.lock.lock.lock();
@@ -139,6 +162,7 @@
             ReentrantMutexDeviceState {
                 service: Cell::new(Some(any)),
                 utilities,
+                command_admission: actual_hook,
             },
         );
 
@@ -165,6 +189,32 @@
     ///
     /// If the same state type is locked multiple times on the same thread.
     fn with_lock<R>(&self, f: impl FnOnce(&mut dyn Any) -> R) -> R {
+        // Only a cold guarded service needs a bootstrap permit. Inspect cached
+        // metadata, release the lock, THEN admit before the actual init lock.
+        // Warm scoped calls retain their explicit Client stream; no ambient
+        // stream is inferred for a second nested admission.
+        let initialize = if self.command_admission.is_some() && {
+            let state = self.lock.lock.lock();
+            let map = state.map.borrow();
+            !map.contains_key(&self.type_id)
+        } {
+            super::admission::admit(
+                self.command_admission.as_ref(),
+                super::CommandRequest {
+                    service: crate::device::ServiceId {
+                        device: self.device_id,
+                        service: self.type_id,
+                    },
+                    stream: None,
+                    kind: super::CommandKind::LocateOrInitialize,
+                    scope: None,
+                },
+            )
+            .unwrap_or_else(|e| panic!("{e}"))
+        } else {
+            None
+        };
+        let _entered = initialize.map(|permit| permit.owner_enter_for_transport());
         let key = self.type_id;
         let state = self.lock.lock.lock();
 
@@ -174,7 +224,17 @@
         let entry = {
             let mut map = state.map.borrow_mut();
             map.entry(key)
-                .or_insert_with(|| (self.init)(self.device_id))
+                .or_insert_with(|| {
+                    let initialized = (self.init)(self.device_id);
+                    assert!(
+                        super::admission::same_hook(
+                            &self.command_admission,
+                            &initialized.command_admission
+                        ),
+                        "Service command bridge differs from bootstrap"
+                    );
+                    initialized
+                })
                 .service
                 .take()
         };
@@ -244,10 +304,13 @@
     /// `None` means the state is currently borrowed by a `with_lock` call.
     service: Cell<Option<Box<dyn Any + Send + 'static>>>,
     utilities: ServerUtilitiesHandle,
+    command_admission: Option<Arc<dyn super::CommandAdmission>>,
 }
 
 impl DeviceStateLock {
     fn locate<S: DeviceService>(device: DeviceId) -> ReentrantMutexDeviceHandle {
+        // Cached metadata lookup MUST stay outside GLOBAL/device locks.
+        let command_admission = S::bootstrap_command_admission(device);
         let mut global = GLOBAL.lock();
 
         let locator_state = match &mut global.state {
@@ -282,6 +345,7 @@
             type_id: TypeId::of::<S>(),
             type_name: core::any::type_name::<S>(),
             init: init_state::<S>,
+            command_admission,
         }
     }
 }
diff --git a/cubecl-runtime-0.11.0/src/client.rs b/cubecl-runtime-0.11.0/src/client.rs
--- a/cubecl-runtime-0.11.0/src/client.rs
+++ b/cubecl-runtime-0.11.0/src/client.rs
@@ -22,7 +22,7 @@
 mod lazy;
 use cubecl_common::{
     bytes::{AllocationProperty, Bytes},
-    device::{DeviceId, ServiceId},
+    device::{CommandParticipation, CommandScope, DeviceId, ServiceId},
     device_handle::{CallResultExt, DeviceHandle},
     profile::ProfileDuration,
 };
@@ -252,6 +252,32 @@
         self.device.service_id()
     }
 
+    /// Transport hooks alone cannot authorize production CUDA capture.
+    pub fn command_participation(&self) -> CommandParticipation {
+        self.device.command_participation()
+    }
+
+    /// Existing actor serialization with an explicitly owned original epoch.
+    /// An unpinned Client is refused before enqueue; does not begin CUDA capture.
+    pub fn exclusive_with_command_scope<R: Send, T: FnOnce() -> R + Send>(
+        &self,
+        scope: &mut CommandScope,
+        task: T,
+    ) -> Result<R, ServerError> {
+        if self.stream_id != Some(scope.stream_id()) || self.service_id() != scope.service_id() {
+            return Err(ServerError::Generic {
+                reason: "Command scope requires the same explicitly pinned original Client".into(),
+                backtrace: BackTrace::capture(),
+            });
+        }
+        self.device
+            .exclusive_with_command_scope(scope, task)
+            .map_err(|err| ServerError::Generic {
+                reason: format!("{err:?}"),
+                backtrace: BackTrace::capture(),
+            })
+    }
+
     /// Whether the server behind this client is an `S`. The client is erased
     /// over its server type, so a caller naming one has to be checked here,
     /// before a downcast on the device thread turns the mismatch into a panic.
@@ -291,6 +317,8 @@
     /// This is highly unsafe and should probably only be used by the CubeCL/Burn projects for now.
     pub unsafe fn set_stream(&mut self, stream_id: StreamId) {
         self.stream_id = Some(stream_id);
+        // SAFETY: this is the same explicit stream used by every Client command.
+        unsafe { self.device.set_command_stream(stream_id) };
     }
 
     fn do_read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/followup-v2/CLASSIFICATION_FOLLOWUP.patch

Original UTF-8 bytes: 2469; SHA256 `8e13ec561aa0e4005cfc7f056da410ad471fa0e7d9b630193cf496a8c00353ae`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/cubecl-common-0.11.0/src/device/handle/admission/tests.rs b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
--- a/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
+++ b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
@@ -579,3 +579,34 @@
 fn original_reentrant_transports_lazy_bootstrap_and_explicit_reentry() {
     exercise_original_transport::<super::super::reentrant::ReentrantMutexDeviceHandle>();
 }
+
+#[test]
+fn admission_refusal_never_fabricates_an_actor_panic_payload() {
+    let refused = CallError::from_admission(CommandAdmissionError::Fenced);
+    assert_eq!(
+        refused.admission_error(),
+        Some(CommandAdmissionError::Fenced)
+    );
+    assert_eq!(
+        refused.message(),
+        Some(CommandAdmissionError::Fenced.message())
+    );
+    assert!(refused.into_panic().is_none());
+
+    let actual = CallError::from_panic(Box::new(1707u32));
+    assert_eq!(actual.admission_error(), None);
+    assert_eq!(
+        *actual.into_panic().unwrap().downcast::<u32>().unwrap(),
+        1707
+    );
+
+    let result = std::panic::catch_unwind(|| {
+        CallError::from_admission(CommandAdmissionError::Cancelled).resume()
+    });
+    let payload = result.unwrap_err();
+    let message = payload
+        .downcast_ref::<std::string::String>()
+        .map(std::string::String::as_str)
+        .or_else(|| payload.downcast_ref::<&'static str>().copied());
+    assert_eq!(message, Some(CommandAdmissionError::Cancelled.message()));
+}
diff --git a/cubecl-common-0.11.0/src/device/handle/base.rs b/cubecl-common-0.11.0/src/device/handle/base.rs
--- a/cubecl-common-0.11.0/src/device/handle/base.rs
+++ b/cubecl-common-0.11.0/src/device/handle/base.rs
@@ -72,15 +72,15 @@
     /// The payload can be handed to [`std::panic::resume_unwind`] to re-raise
     /// the original panic on the caller's thread.
     pub fn into_panic(self) -> Option<Box<dyn Any + Send>> {
-        match self.admission {
-            Some(error) => Some(Box::new(error.message())),
-            None => self.payload,
-        }
+        self.payload
     }
 
     /// Re-raises the failure on the current thread, diverging.
     #[track_caller]
     pub fn resume(self) -> ! {
+        if let Some(error) = self.admission {
+            panic!("{error}");
+        }
         match self.into_panic() {
             #[cfg(feature = "std")]
             Some(payload) => std::panic::resume_unwind(payload),

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/followup-v2/frozen-v2/CLASSIFICATION_FOLLOWUP.patch

Original UTF-8 bytes: 2469; SHA256 `8e13ec561aa0e4005cfc7f056da410ad471fa0e7d9b630193cf496a8c00353ae`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/cubecl-common-0.11.0/src/device/handle/admission/tests.rs b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
--- a/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
+++ b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
@@ -579,3 +579,34 @@
 fn original_reentrant_transports_lazy_bootstrap_and_explicit_reentry() {
     exercise_original_transport::<super::super::reentrant::ReentrantMutexDeviceHandle>();
 }
+
+#[test]
+fn admission_refusal_never_fabricates_an_actor_panic_payload() {
+    let refused = CallError::from_admission(CommandAdmissionError::Fenced);
+    assert_eq!(
+        refused.admission_error(),
+        Some(CommandAdmissionError::Fenced)
+    );
+    assert_eq!(
+        refused.message(),
+        Some(CommandAdmissionError::Fenced.message())
+    );
+    assert!(refused.into_panic().is_none());
+
+    let actual = CallError::from_panic(Box::new(1707u32));
+    assert_eq!(actual.admission_error(), None);
+    assert_eq!(
+        *actual.into_panic().unwrap().downcast::<u32>().unwrap(),
+        1707
+    );
+
+    let result = std::panic::catch_unwind(|| {
+        CallError::from_admission(CommandAdmissionError::Cancelled).resume()
+    });
+    let payload = result.unwrap_err();
+    let message = payload
+        .downcast_ref::<std::string::String>()
+        .map(std::string::String::as_str)
+        .or_else(|| payload.downcast_ref::<&'static str>().copied());
+    assert_eq!(message, Some(CommandAdmissionError::Cancelled.message()));
+}
diff --git a/cubecl-common-0.11.0/src/device/handle/base.rs b/cubecl-common-0.11.0/src/device/handle/base.rs
--- a/cubecl-common-0.11.0/src/device/handle/base.rs
+++ b/cubecl-common-0.11.0/src/device/handle/base.rs
@@ -72,15 +72,15 @@
     /// The payload can be handed to [`std::panic::resume_unwind`] to re-raise
     /// the original panic on the caller's thread.
     pub fn into_panic(self) -> Option<Box<dyn Any + Send>> {
-        match self.admission {
-            Some(error) => Some(Box::new(error.message())),
-            None => self.payload,
-        }
+        self.payload
     }
 
     /// Re-raises the failure on the current thread, diverging.
     #[track_caller]
     pub fn resume(self) -> ! {
+        if let Some(error) = self.admission {
+            panic!("{error}");
+        }
         match self.into_panic() {
             #[cfg(feature = "std")]
             Some(payload) => std::panic::resume_unwind(payload),

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/frozen/COMMAND_ADMISSION.patch

Original UTF-8 bytes: 56107; SHA256 `cabcf7252ddb985dc7e4dc4d113ef13ec57fe133d24a858ef9c77d3d4f522245`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/cubecl-common-0.11.0/src/device/base.rs b/cubecl-common-0.11.0/src/device/base.rs
--- a/cubecl-common-0.11.0/src/device/base.rs
+++ b/cubecl-common-0.11.0/src/device/base.rs
@@ -3,6 +3,12 @@
     cmp::Ordering,
 };
 use cubecl_environment::sync::Arc;
+
+// Public contract exports through device::base, preserving device_handle ABI.
+pub use super::handle::{
+    CommandAdmission, CommandAdmissionError, CommandKind, CommandParticipation, CommandRequest,
+    CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RunningCommandPermit,
+};
 
 /// The device id.
 #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
@@ -56,6 +62,21 @@
         Self: Sized;
     /// Get the service utilities.
     fn utilities(&self) -> ServerUtilitiesHandle;
+    /// Cached optional command bridge. Pure metadata; no driver/registry work.
+    /// Returning Some does not establish complete CUDA participation.
+    fn command_admission(&self) -> Option<Arc<dyn crate::device::handle::CommandAdmission>> {
+        None
+    }
+    /// Same cached hook, obtainable BEFORE transport/registry locks or init.
+    /// Must not initialize a device, acquire an actor, or discover a provider.
+    fn bootstrap_command_admission(
+        _device_id: DeviceId,
+    ) -> Option<Arc<dyn crate::device::handle::CommandAdmission>>
+    where
+        Self: Sized,
+    {
+        None
+    }
     /// Which pipeline stage this service runs on.
     ///
     /// Services on [`DeviceServiceStage::Upstream`] produce work ahead of time (e.g. autodiff graph
diff --git a/cubecl-common-0.11.0/src/device/handle/admission/tests.rs b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
new file mode 100644
--- /dev/null
+++ b/cubecl-common-0.11.0/src/device/handle/admission/tests.rs
@@ -0,0 +1,581 @@
+//! Host fixtures only. No CUDA bridge, physical identity, driver or device work.
+use super::*;
+use crate::device::ServerUtilitiesHandle;
+use crate::device::handle::{CallError, DeviceHandle, DeviceHandleSpec, ServiceCreationError};
+use core::any::{Any, TypeId};
+use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
+use std::cell::Cell;
+use std::sync::Mutex;
+use std::vec::Vec;
+
+#[derive(Clone, Copy)]
+struct Entered {
+    gate: usize,
+    service: ServiceId,
+    stream: Option<StreamId>,
+    capture: bool,
+}
+
+std::thread_local! {
+    static ENTERED: Cell<Option<Entered>> = const { Cell::new(None) };
+}
+
+#[derive(Clone, Copy)]
+struct Seen {
+    service: ServiceId,
+    stream: Option<StreamId>,
+    kind: CommandKind,
+    explicit_scope: bool,
+}
+
+#[derive(Default)]
+struct GateState {
+    seen: Mutex<Vec<Seen>>,
+    live: AtomicUsize,
+    entered: AtomicUsize,
+    retired: AtomicUsize,
+    discarded: AtomicUsize,
+    refuse: AtomicBool,
+    epoch_live: AtomicBool,
+}
+
+struct Gate(Arc<GateState>);
+
+// SAFETY: synthetic host-only ledger; no physical-completion claims. Calls
+// never wait, and entered scopes are exact test identities, not ambient IDs.
+unsafe impl CommandAdmission for Gate {
+    fn admit(
+        &self,
+        request: CommandRequest<'_>,
+    ) -> Result<OwnedCommandPermit, CommandAdmissionError> {
+        if self.0.refuse.load(Ordering::SeqCst) {
+            return Err(CommandAdmissionError::Fenced);
+        }
+        let gate = Arc::as_ptr(&self.0) as usize;
+        let mut capture = request.scope.is_some();
+        ENTERED.with(|slot| {
+            if let Some(entered) = slot.get() {
+                if entered.gate == gate && entered.capture {
+                    if entered.service != request.service || entered.stream != request.stream {
+                        return Err(CommandAdmissionError::ForeignScope);
+                    }
+                    capture = true;
+                }
+            }
+            Ok(())
+        })?;
+        self.0.seen.lock().unwrap().push(Seen {
+            service: request.service,
+            stream: request.stream,
+            kind: request.kind,
+            explicit_scope: request.scope.is_some(),
+        });
+        self.0.live.fetch_add(1, Ordering::SeqCst);
+        Ok(OwnedCommandPermit::new(Posted {
+            state: self.0.clone(),
+            identity: Entered {
+                gate,
+                service: request.service,
+                stream: request.stream,
+                capture,
+            },
+            entered: false,
+        }))
+    }
+}
+
+struct Posted {
+    state: Arc<GateState>,
+    identity: Entered,
+    entered: bool,
+}
+
+// SAFETY: transfers the same owned synthetic host count into a same-thread
+// non-Send guard; Drop only changes atomics and never blocks or touches a device.
+unsafe impl PostedCommandPermit for Posted {
+    fn enter(mut self: Box<Self>) -> Box<dyn RunningCommandPermit> {
+        let previous = ENTERED.with(|slot| slot.replace(Some(self.identity)));
+        self.entered = true;
+        self.state.entered.fetch_add(1, Ordering::SeqCst);
+        Box::new(Running {
+            state: self.state.clone(),
+            previous,
+            _not_send: core::marker::PhantomData,
+        })
+    }
+}
+
+impl Drop for Posted {
+    fn drop(&mut self) {
+        if !self.entered {
+            self.state.live.fetch_sub(1, Ordering::SeqCst);
+            self.state.discarded.fetch_add(1, Ordering::SeqCst);
+        }
+    }
+}
+
+struct Running {
+    state: Arc<GateState>,
+    previous: Option<Entered>,
+    _not_send: core::marker::PhantomData<std::rc::Rc<()>>,
+}
+
+// SAFETY: host-only; restores exactly this thread's prior scope and does not
+// declare any physical work complete.
+unsafe impl RunningCommandPermit for Running {}
+
+impl Drop for Running {
+    fn drop(&mut self) {
+        ENTERED.with(|slot| slot.set(self.previous));
+        self.state.live.fetch_sub(1, Ordering::SeqCst);
+        self.state.retired.fetch_add(1, Ordering::SeqCst);
+    }
+}
+
+struct Epoch(Arc<GateState>);
+
+// SAFETY: synthetic host epoch only; no complete-participation issuer.
+unsafe impl CommandScopeOwner for Epoch {
+    fn validate(&self, _: ServiceId, _: StreamId) -> Result<(), CommandAdmissionError> {
+        if self.0.epoch_live.load(Ordering::SeqCst) {
+            Ok(())
+        } else {
+            Err(CommandAdmissionError::Stale)
+        }
+    }
+}
+
+struct Service {
+    hook: Option<Arc<dyn CommandAdmission>>,
+    value: usize,
+}
+
+impl DeviceService for Service {
+    fn init(_: DeviceId) -> Self {
+        Self {
+            hook: None,
+            value: 0,
+        }
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        Arc::new(())
+    }
+    fn command_admission(&self) -> Option<Arc<dyn CommandAdmission>> {
+        self.hook.clone()
+    }
+}
+
+type Task = Box<dyn FnOnce(&mut dyn Any) + Send>;
+
+struct State {
+    device: DeviceId,
+    value: Mutex<Box<dyn Any + Send>>,
+    queue: Mutex<Vec<Task>>,
+    hook: Option<Arc<dyn CommandAdmission>>,
+}
+
+#[derive(Clone)]
+struct Deferred(Arc<State>);
+
+impl Deferred {
+    fn run_one(&self) {
+        let task = self.0.queue.lock().unwrap().remove(0);
+        let mut value = self.0.value.lock().unwrap();
+        task(&mut **value);
+    }
+}
+
+// Deterministic fake transport deliberately leaves Submit queued. It tests
+// ownership across posting, not the original actor's implementation/runtime.
+impl DeviceHandleSpec for Deferred {
+    const BLOCKING: bool = false;
+    fn insert<S: DeviceService>(
+        device: DeviceId,
+        service: S,
+    ) -> Result<Self, ServiceCreationError> {
+        let hook = service.command_admission();
+        Ok(Self(Arc::new(State {
+            device,
+            value: Mutex::new(Box::new(service)),
+            queue: Mutex::new(Vec::new()),
+            hook,
+        })))
+    }
+    fn new<S: DeviceService>(device: DeviceId) -> Self {
+        Self::insert(device, S::init(device)).unwrap()
+    }
+    fn device_id(&self) -> DeviceId {
+        self.0.device
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        Arc::new(())
+    }
+    fn command_admission(&self) -> Option<Arc<dyn CommandAdmission>> {
+        self.0.hook.clone()
+    }
+    fn flush_queue(&self) {}
+    fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
+        &self,
+        task: T,
+    ) -> Result<R, CallError> {
+        let mut value = self.0.value.lock().unwrap();
+        Ok(task(&mut **value))
+    }
+    fn submit<T: FnOnce(&mut dyn Any) + Send + 'static>(&self, task: T) {
+        self.0.queue.lock().unwrap().push(Box::new(task));
+    }
+    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
+        Ok(task())
+    }
+}
+
+type Handle = DeviceHandle<Service, Deferred>;
+
+fn make_handle(hook: Option<Arc<dyn CommandAdmission>>) -> Handle {
+    Handle::insert(
+        super::super::next_test_device_id(),
+        Service { hook, value: 0 },
+    )
+    .unwrap()
+}
+
+fn make_gate() -> (Arc<dyn CommandAdmission>, Arc<GateState>) {
+    let state = Arc::new(GateState::default());
+    state.epoch_live.store(true, Ordering::SeqCst);
+    (Arc::new(Gate(state.clone())), state)
+}
+
+fn scope(
+    hook: Arc<dyn CommandAdmission>,
+    state: Arc<GateState>,
+    handle: &Handle,
+    stream: StreamId,
+) -> CommandScope {
+    // SAFETY: exact test bridge + synthetic epoch, no hardware/capture claim.
+    unsafe { CommandScope::from_original_epoch(hook, handle.service_id(), stream, Epoch(state)) }
+}
+
+#[test]
+fn default_none_uses_original_calls_and_stays_unproven() {
+    let handle = make_handle(None);
+    handle.submit(|value| value.value = 7);
+    assert_eq!(handle.handle.0.queue.lock().unwrap().len(), 1);
+    handle.handle.run_one();
+    assert_eq!(handle.submit_blocking(|value| value.value).unwrap(), 7);
+    assert_eq!(handle.exclusive(|| 19).unwrap(), 19);
+    assert_eq!(
+        handle.command_participation(),
+        CommandParticipation::Unproven
+    );
+    assert_eq!(handle.command_stream, None);
+}
+
+#[test]
+fn queued_closure_owns_permit_until_actual_execution() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    handle.submit(|value| value.value += 1);
+    assert_eq!(state.live.load(Ordering::SeqCst), 1);
+    assert_eq!(state.entered.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 0);
+    handle.handle.run_one();
+    assert_eq!(state.entered.load(Ordering::SeqCst), 1);
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 1);
+    assert_eq!(handle.handle.0.queue.lock().unwrap().len(), 0);
+}
+
+#[test]
+fn discarded_queued_closure_retires_only_unentered_host_owner() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    handle.submit(|_| panic!("discarded task must never execute"));
+    handle.handle.0.queue.lock().unwrap().clear();
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.discarded.load(Ordering::SeqCst), 1);
+    assert_eq!(state.entered.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 0);
+}
+
+#[test]
+fn refusal_precedes_enqueue_and_callback() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    state.refuse.store(true, Ordering::SeqCst);
+    let called = AtomicBool::new(false);
+    let error = handle
+        .submit_blocking(|_| called.store(true, Ordering::SeqCst))
+        .unwrap_err();
+    assert_eq!(
+        error.message(),
+        Some(CommandAdmissionError::Fenced.message())
+    );
+    assert!(!called.load(Ordering::SeqCst));
+    assert!(handle.handle.0.queue.lock().unwrap().is_empty());
+    assert!(state.seen.lock().unwrap().is_empty());
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+}
+
+#[test]
+fn unwind_restores_entered_scope_and_retires_host_owner() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
+        let _ = handle.exclusive(|| panic!("fixture body"));
+    }));
+    assert!(result.is_err());
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 1);
+    ENTERED.with(|entered| assert!(entered.get().is_none()));
+}
+
+#[test]
+fn ambient_stream_never_manufactures_explicit_binding() {
+    let (hook, state) = make_gate();
+    let handle = make_handle(Some(hook));
+    StreamId { value: 1701 }.executes(|| handle.submit_blocking(|_| ()).unwrap());
+    let seen = state.seen.lock().unwrap();
+    assert_eq!(seen.len(), 1);
+    assert_eq!(seen[0].stream, None);
+    assert!(!seen[0].explicit_scope);
+    assert_eq!(seen[0].service, handle.service_id());
+    assert_eq!(seen[0].kind, CommandKind::SubmitBlocking);
+}
+
+#[test]
+fn exact_scope_checks_unbound_foreign_hook_service_stream_and_stale_epoch() {
+    let (hook, state) = make_gate();
+    let mut handle = make_handle(Some(hook.clone()));
+    let stream = StreamId { value: 1702 };
+    let mut owner = scope(hook.clone(), state.clone(), &handle, stream);
+    assert!(
+        handle
+            .exclusive_with_command_scope(&mut owner, || ())
+            .is_err()
+    );
+    // SAFETY: this synthetic handle issues no device/stream operations.
+    unsafe { handle.set_command_stream(stream) };
+    let (foreign, _) = make_gate();
+    assert!(
+        owner
+            .validate(&foreign, handle.service_id(), Some(stream))
+            .is_err()
+    );
+    assert!(
+        owner
+            .validate(
+                &hook,
+                ServiceId {
+                    device: handle.device_id(),
+                    service: TypeId::of::<()>()
+                },
+                Some(stream)
+            )
+            .is_err()
+    );
+    assert!(
+        owner
+            .validate(&hook, handle.service_id(), Some(StreamId { value: 1703 }))
+            .is_err()
+    );
+    state.epoch_live.store(false, Ordering::SeqCst);
+    let error = handle
+        .exclusive_with_command_scope(&mut owner, || ())
+        .unwrap_err();
+    assert_eq!(
+        error.message(),
+        Some(CommandAdmissionError::Stale.message())
+    );
+    assert!(state.seen.lock().unwrap().is_empty());
+}
+
+#[test]
+fn original_exclusive_entry_transports_explicit_scope_and_allows_exact_nested_calls() {
+    let (hook, state) = make_gate();
+    let mut handle = make_handle(Some(hook.clone()));
+    let stream = StreamId { value: 1704 };
+    // SAFETY: synthetic host commands only; no stream driver operations.
+    unsafe { handle.set_command_stream(stream) };
+    let mut owner = scope(hook, state.clone(), &handle, stream);
+    handle
+        .exclusive_with_command_scope(&mut owner, || {
+            ENTERED.with(|entered| assert!(entered.get().unwrap().capture));
+            handle.submit_blocking(|value| value.value = 9).unwrap();
+            assert_eq!(state.live.load(Ordering::SeqCst), 1);
+        })
+        .unwrap();
+    let seen = state.seen.lock().unwrap();
+    assert_eq!(seen.len(), 2);
+    assert!(seen[0].explicit_scope);
+    assert!(!seen[1].explicit_scope);
+    assert!(
+        seen.iter()
+            .all(|r| r.service == handle.service_id() && r.stream == Some(stream))
+    );
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retired.load(Ordering::SeqCst), 2);
+}
+
+#[test]
+fn clone_and_seen_as_preserve_exact_service_stream_and_cached_hook() {
+    let (hook, state) = make_gate();
+    let mut handle = make_handle(Some(hook));
+    let stream = StreamId { value: 1705 };
+    // SAFETY: host-only synthetic commands.
+    unsafe { handle.set_command_stream(stream) };
+    let erased = handle.clone().seen_as::<dyn Any>(|value| value);
+    erased
+        .submit_blocking(|value| value.downcast_mut::<Service>().unwrap().value = 11)
+        .unwrap();
+    assert_eq!(erased.service_id(), handle.service_id());
+    let seen = state.seen.lock().unwrap();
+    assert_eq!(seen[0].service, handle.service_id());
+    assert_eq!(seen[0].stream, Some(stream));
+    assert_eq!(
+        erased.command_participation(),
+        CommandParticipation::Unproven
+    );
+}
+
+#[test]
+fn unsafe_registered_hook_does_not_issue_complete_participation() {
+    let (hook, _) = make_gate();
+    let handle = make_handle(Some(hook));
+    assert_eq!(
+        handle.command_participation(),
+        CommandParticipation::Unproven
+    );
+}
+
+// The registry below is ONLY fixture input for the three original transports.
+// It is not a provider discovery mechanism or a second physical queue.
+struct Registered {
+    hook: Arc<dyn CommandAdmission>,
+    state: Arc<GateState>,
+}
+
+static REGISTERED: std::sync::OnceLock<Mutex<std::collections::HashMap<DeviceId, Registered>>> =
+    std::sync::OnceLock::new();
+
+fn registered() -> &'static Mutex<std::collections::HashMap<DeviceId, Registered>> {
+    REGISTERED.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
+}
+
+struct OriginalService {
+    hook: Arc<dyn CommandAdmission>,
+    state: Arc<GateState>,
+    value: usize,
+}
+
+impl DeviceService for OriginalService {
+    fn init(device: DeviceId) -> Self {
+        ENTERED.with(|entered| {
+            let entered = entered.get().expect("actual constructor must be prepaid");
+            assert_eq!(entered.service, ServiceId::of::<Self>(device));
+            assert!(!entered.capture);
+        });
+        let map = registered().lock().unwrap();
+        let entry = map.get(&device).unwrap();
+        Self {
+            hook: entry.hook.clone(),
+            state: entry.state.clone(),
+            value: 0,
+        }
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        self.state.clone()
+    }
+    fn command_admission(&self) -> Option<Arc<dyn CommandAdmission>> {
+        Some(self.hook.clone())
+    }
+    fn bootstrap_command_admission(device: DeviceId) -> Option<Arc<dyn CommandAdmission>> {
+        Some(
+            registered()
+                .lock()
+                .unwrap()
+                .get(&device)
+                .unwrap()
+                .hook
+                .clone(),
+        )
+    }
+}
+
+fn original_handle<I: DeviceHandleSpec>(device: DeviceId) -> DeviceHandle<OriginalService, I> {
+    let (hook, state) = make_gate();
+    assert!(
+        registered()
+            .lock()
+            .unwrap()
+            .insert(device, Registered { hook, state })
+            .is_none()
+    );
+    DeviceHandle::new(device)
+}
+
+fn exercise_original_transport<I: DeviceHandleSpec + Sync>() {
+    let fixture = super::super::DeviceFixture::new(
+        original_handle::<I>,
+        DeviceHandle::<OriginalService, I>::shutdown,
+    );
+    // The reentrant transport initializes lazily at this exact original call.
+    let state = fixture
+        .submit_blocking(|service| service.state.clone())
+        .unwrap();
+    let hook = registered()
+        .lock()
+        .unwrap()
+        .get(&fixture.device_id())
+        .unwrap()
+        .hook
+        .clone();
+    let mut pinned: DeviceHandle<OriginalService, I> = (*fixture).clone();
+    let stream = StreamId { value: 1706 };
+    // SAFETY: host-only service; no CUDA or logical driver stream is created.
+    unsafe { pinned.set_command_stream(stream) };
+    // SAFETY: synthetic epoch with same fixture hook/service/explicit stream.
+    let mut owner = unsafe {
+        CommandScope::from_original_epoch(hook, pinned.service_id(), stream, Epoch(state.clone()))
+    };
+    pinned
+        .exclusive_with_command_scope(&mut owner, || {
+            pinned
+                .submit_blocking(|service| service.value = 23)
+                .unwrap();
+            assert_eq!(pinned.submit_blocking(|service| service.value).unwrap(), 23);
+        })
+        .unwrap();
+    assert_eq!(state.live.load(Ordering::SeqCst), 0);
+    assert!(
+        state
+            .seen
+            .lock()
+            .unwrap()
+            .iter()
+            .any(|seen| seen.kind == CommandKind::LocateOrInitialize)
+    );
+    assert_eq!(
+        pinned.command_participation(),
+        CommandParticipation::Unproven
+    );
+    registered()
+        .lock()
+        .unwrap()
+        .remove(&fixture.device_id())
+        .unwrap();
+}
+
+#[test]
+fn original_channel_transports_owned_bootstrap_and_explicit_reentry() {
+    exercise_original_transport::<super::super::channel::ChannelDeviceHandle>();
+}
+
+#[test]
+fn original_mutex_transports_owned_bootstrap_and_explicit_reentry() {
+    exercise_original_transport::<super::super::mutex::MutexDeviceHandle>();
+}
+
+#[test]
+fn original_reentrant_transports_lazy_bootstrap_and_explicit_reentry() {
+    exercise_original_transport::<super::super::reentrant::ReentrantMutexDeviceHandle>();
+}
diff --git a/cubecl-common-0.11.0/src/device/handle/admission.rs b/cubecl-common-0.11.0/src/device/handle/admission.rs
new file mode 100644
--- /dev/null
+++ b/cubecl-common-0.11.0/src/device/handle/admission.rs
@@ -0,0 +1,265 @@
+//! Optional host-command ownership; never a device-completion certificate.
+use crate::device::{DeviceId, DeviceService, ServiceId};
+use alloc::boxed::Box;
+use core::fmt;
+use cubecl_environment::{stream::StreamId, sync::Arc};
+
+/// No complete CUDA participation issuer exists in this transport slice.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum CommandParticipation {
+    /// Constructors, external events/storage and final retirement are unproven.
+    Unproven,
+}
+
+/// The original actor operation, before entering its transport or registry lock.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum CommandKind {
+    /// May create the service; an inserted service was constructed earlier.
+    LocateOrInitialize,
+    /// Install an already constructed service; does not cover its construction.
+    Install,
+    /// A posted closure, including its captures, until host execution finishes.
+    Submit,
+    /// A host-blocking actor call, not a device synchronization guarantee.
+    SubmitBlocking,
+    /// The original actor's reentrant exclusive closure.
+    Exclusive,
+}
+
+/// An admission refusal occurs before enqueue, never after a command is posted.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum CommandAdmissionError {
+    /// No audited bridge for the requested operation.
+    Unproven,
+    /// Hook, service or explicit logical stream differs.
+    ForeignScope,
+    /// Original physical queue epoch was revoked.
+    Stale,
+    /// Original physical card is fenced.
+    Fenced,
+    /// The original request was cancelled before admission.
+    Cancelled,
+}
+
+impl CommandAdmissionError {
+    pub(crate) const fn message(self) -> &'static str {
+        match self {
+            Self::Unproven => "device command admission: participation unproven",
+            Self::ForeignScope => "device command admission: foreign capture scope",
+            Self::Stale => "device command admission: stale original queue epoch",
+            Self::Fenced => "device command admission: original physical card fenced",
+            Self::Cancelled => "device command admission: cancelled before enqueue",
+        }
+    }
+}
+
+impl fmt::Display for CommandAdmissionError {
+    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+        f.write_str(self.message())
+    }
+}
+
+/// Exact original service and optional explicitly pinned Client stream.
+pub struct CommandRequest<'a> {
+    /// Actual transport registry service; not a PCI-card identity.
+    pub service: ServiceId,
+    /// `None` stays unbound; no inference from the caller's ambient StreamId.
+    pub stream: Option<StreamId>,
+    /// Host operation being admitted.
+    pub kind: CommandKind,
+    /// Supplied only by an explicit original pinned-client scope.
+    pub scope: Option<&'a CommandScope>,
+}
+
+/// A cached bridge into the SAME original physical QueueState/ResourceArbiter.
+///
+/// # Safety
+///
+/// Admission may wait only before enqueue/registry lock. Temporary capture close
+/// parks an ordinary caller rather than rejecting previously accepted work. A
+/// refusal must imply that neither enqueue nor driver effects occurred. Entered
+/// original commands may make nested actor calls without waiting for their own
+/// permit to drain. Capture reentry requires the explicit validated scope, never
+/// merely a thread/stream ID. No new physical registry, scheduler or byte ledger.
+/// All callbacks, hook destruction and permit Drops must avoid waiting on
+/// actor/drop work needed by drain. This first slice MUST NOT register a bridge
+/// that closes admission: external events/storage/final destruction still lack
+/// pre-admitted retirement. A later audited bridge may implement the waiting
+/// contract above only AFTER all participation is integrated and qualified.
+/// This interface alone does NOT cover device-terminal ownership.
+pub unsafe trait CommandAdmission: Send + Sync + 'static {
+    /// Returns an owned host-command permit, before any transport enqueue.
+    fn admit(
+        &self,
+        request: CommandRequest<'_>,
+    ) -> Result<OwnedCommandPermit, CommandAdmissionError>;
+}
+
+/// Queued ownership moves into the actual original actor closure.
+pub struct OwnedCommandPermit {
+    owner: Box<dyn PostedCommandPermit>,
+}
+
+impl OwnedCommandPermit {
+    /// Seal an original bridge's owned permit; no complete capability is issued.
+    pub fn new(owner: impl PostedCommandPermit + 'static) -> Self {
+        Self {
+            owner: Box::new(owner),
+        }
+    }
+
+    pub(super) fn owner_enter_for_transport(self) -> Box<dyn RunningCommandPermit> {
+        self.owner.enter()
+    }
+
+    pub(crate) fn run<R>(self, task: impl FnOnce() -> R) -> R {
+        let _entered = self.owner.enter();
+        task()
+    }
+}
+
+/// Owner retained while queued, including transport discard/panic paths.
+///
+/// # Safety
+///
+/// Drop is nonblocking and performs no driver calls. `enter` is nonblocking,
+/// infallible for admitted work, and establishes only the original command's
+/// actor-local reentry, including an explicitly admitted capture scope if any.
+/// It transfers all ownership into the returned guard. Unexpected enter/panic
+/// must retain/fence through the original queue rather than declare completion.
+pub unsafe trait PostedCommandPermit: Send {
+    /// Called on the original execution thread, not on the posting thread.
+    fn enter(self: Box<Self>) -> Box<dyn RunningCommandPermit>;
+}
+
+/// Host execution owner; intentionally not Send or a hardware Ready event.
+///
+/// # Safety
+///
+/// Drop is nonblocking, releases actor-local scope on the same thread, and
+/// retires only HOST-command ownership. Any posted CUDA work retains a separate
+/// original flight/event owner until known terminal completion or quarantine.
+pub unsafe trait RunningCommandPermit {}
+
+/// Original capture-epoch owner; no implementation is supplied here.
+///
+/// # Safety
+///
+/// Own the actual noncloneable capture epoch/ticket and original physical key.
+/// Validation and Drop are nonblocking; revoked/unknown work stays under the
+/// original queue fence/quarantine. Caller-supplied IDs alone cannot implement
+/// this contract. An admitted permit must retain the actual epoch through its
+/// entire queued/executing lifetime, not borrow this object past `admit`.
+pub unsafe trait CommandScopeOwner: Send + Sync {
+    /// Validate actual epoch and exact service/stream, before enqueue.
+    fn validate(&self, service: ServiceId, stream: StreamId) -> Result<(), CommandAdmissionError>;
+}
+
+/// Move-only authority supplied by an original pinned Client/capture epoch.
+/// No safe/public complete-participation factory exists in this slice.
+pub struct CommandScope {
+    admission: Arc<dyn CommandAdmission>,
+    service: ServiceId,
+    stream: StreamId,
+    owner: Box<dyn CommandScopeOwner>,
+}
+
+impl CommandScope {
+    /// Bridge an already-owned original epoch; does not grant CUDA eligibility.
+    ///
+    /// # Safety
+    ///
+    /// `owner` must own the same original QueueState/key/epoch used by
+    /// `admission`, and service/stream must come from the actual pinned Client.
+    /// All same-card lifecycle participation remains a separate CLOSED gate.
+    pub unsafe fn from_original_epoch(
+        admission: Arc<dyn CommandAdmission>,
+        service: ServiceId,
+        stream: StreamId,
+        owner: impl CommandScopeOwner + 'static,
+    ) -> Self {
+        Self {
+            admission,
+            service,
+            stream,
+            owner: Box::new(owner),
+        }
+    }
+
+    /// Exact original service; metadata alone is not physical ownership proof.
+    pub fn service_id(&self) -> ServiceId {
+        self.service
+    }
+
+    /// Explicit pinned Client logical stream, never ambient StreamId::current.
+    pub fn stream_id(&self) -> StreamId {
+        self.stream
+    }
+
+    pub(crate) fn validate(
+        &self,
+        admission: &Arc<dyn CommandAdmission>,
+        service: ServiceId,
+        stream: Option<StreamId>,
+    ) -> Result<(), CommandAdmissionError> {
+        if !Arc::ptr_eq(&self.admission, admission)
+            || self.service != service
+            || stream != Some(self.stream)
+        {
+            return Err(CommandAdmissionError::ForeignScope);
+        }
+        self.owner.validate(service, self.stream)
+    }
+}
+
+pub(super) fn admit(
+    hook: Option<&Arc<dyn CommandAdmission>>,
+    request: CommandRequest<'_>,
+) -> Result<Option<OwnedCommandPermit>, CommandAdmissionError> {
+    if let Some(scope) = request.scope {
+        scope.validate(
+            hook.ok_or(CommandAdmissionError::Unproven)?,
+            request.service,
+            request.stream,
+        )?;
+    }
+    hook.map(|hook| hook.admit(request)).transpose()
+}
+
+pub(super) fn bootstrap<S: DeviceService>(
+    device: DeviceId,
+    kind: CommandKind,
+) -> Result<
+    (
+        Option<Arc<dyn CommandAdmission>>,
+        Option<OwnedCommandPermit>,
+    ),
+    CommandAdmissionError,
+> {
+    // MUST be called before CHANNELS/GLOBAL/DEVICE_REGISTRY or actor locks.
+    let hook = S::bootstrap_command_admission(device);
+    let permit = admit(
+        hook.as_ref(),
+        CommandRequest {
+            service: ServiceId::of::<S>(device),
+            stream: None,
+            kind,
+            scope: None,
+        },
+    )?;
+    Ok((hook, permit))
+}
+
+pub(super) fn same_hook(
+    expected: &Option<Arc<dyn CommandAdmission>>,
+    actual: &Option<Arc<dyn CommandAdmission>>,
+) -> bool {
+    match (expected, actual) {
+        (None, None) => true,
+        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
+        _ => false,
+    }
+}
+
+#[cfg(test)]
+mod tests;
diff --git a/cubecl-common-0.11.0/src/device/handle/base.rs b/cubecl-common-0.11.0/src/device/handle/base.rs
--- a/cubecl-common-0.11.0/src/device/handle/base.rs
+++ b/cubecl-common-0.11.0/src/device/handle/base.rs
@@ -7,12 +7,13 @@
 ///
 /// When a task panics on a device's runner thread, the panic payload is
 /// captured and carried here instead of being logged and dropped.
-/// A `CallError` without a payload means the runner channel disconnected
-/// before the task could produce a result.
+/// A `CallError` without a payload or admission refusal means the runner
+/// channel disconnected before the task could produce a result.
 pub struct CallError {
     /// The captured panic payload, or `None` if the call failed because the
     /// runner channel disconnected
     payload: Option<Box<dyn Any + Send>>,
+    admission: Option<super::CommandAdmissionError>,
 }
 
 impl CallError {
@@ -23,6 +24,7 @@
     pub(crate) fn from_panic(payload: Box<dyn Any + Send>) -> Self {
         Self {
             payload: Some(payload),
+            admission: None,
         }
     }
 
@@ -30,12 +32,31 @@
     /// a result (no panic payload available).
     #[cfg_attr(not(multi_threading), allow(dead_code))]
     pub(crate) fn disconnected() -> Self {
-        Self { payload: None }
+        Self {
+            payload: None,
+            admission: None,
+        }
+    }
+
+    /// Refusal before enqueue, distinct from actor panic or disconnection.
+    pub(crate) fn from_admission(error: super::CommandAdmissionError) -> Self {
+        Self {
+            payload: None,
+            admission: Some(error),
+        }
+    }
+
+    /// Typed pre-enqueue refusal, if this call never reached the actor.
+    pub fn admission_error(&self) -> Option<super::CommandAdmissionError> {
+        self.admission
     }
 
     /// Returns the panic message if the failure was a panic whose payload is a
     /// string (the common case for `panic!`, indexing, `unwrap`, ...).
     pub fn message(&self) -> Option<&str> {
+        if let Some(error) = self.admission {
+            return Some(error.message());
+        }
         let payload = self.payload.as_ref()?;
         if let Some(s) = payload.downcast_ref::<&'static str>() {
             Some(*s)
@@ -51,7 +72,10 @@
     /// The payload can be handed to [`std::panic::resume_unwind`] to re-raise
     /// the original panic on the caller's thread.
     pub fn into_panic(self) -> Option<Box<dyn Any + Send>> {
-        self.payload
+        match self.admission {
+            Some(error) => Some(Box::new(error.message())),
+            None => self.payload,
+        }
     }
 
     /// Re-raises the failure on the current thread, diverging.
@@ -91,6 +115,9 @@
 
 impl core::fmt::Debug for CallError {
     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
+        if let Some(error) = self.admission {
+            return write!(f, "CallError(command refused before enqueue: {error})");
+        }
         match self.message() {
             Some(message) => write!(
                 f,
@@ -140,6 +167,13 @@
     /// Retrieves the server utilities for this thread.
     fn utilities(&self) -> ServerUtilitiesHandle;
 
+    /// Cached metadata only: do not acquire the service actor or initialize it.
+    fn command_admission(
+        &self,
+    ) -> Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>> {
+        None
+    }
+
     /// Doesn't flush the service state, but flushes any task enqueued in the communication
     /// channel.
     ///
diff --git a/cubecl-common-0.11.0/src/device/handle/channel.rs b/cubecl-common-0.11.0/src/device/handle/channel.rs
--- a/cubecl-common-0.11.0/src/device/handle/channel.rs
+++ b/cubecl-common-0.11.0/src/device/handle/channel.rs
@@ -62,6 +62,12 @@
 
     fn utilities(&self) -> ServerUtilitiesHandle {
         self.state.utilities()
+    }
+
+    fn command_admission(
+        &self,
+    ) -> Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>> {
+        self.state.service.command_admission.clone()
     }
 
     /// Runs `task` on the device thread, blocking until it returns.
@@ -240,6 +246,7 @@
 struct ChannelService {
     type_id: TypeId,
     utilities: ServerUtilitiesHandle,
+    command_admission: Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>>,
 }
 
 #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
@@ -298,6 +305,13 @@
         device_id: DeviceId,
         service: Option<S>,
     ) -> Result<Self, ServiceCreationError> {
+        let kind = if service.is_some() {
+            super::CommandKind::Install
+        } else {
+            super::CommandKind::LocateOrInitialize
+        };
+        let (bootstrap_hook, bootstrap_permit) = super::admission::bootstrap::<S>(device_id, kind)
+            .map_err(|e| ServiceCreationError::new(e.message().into()))?;
         let type_id = TypeId::of::<S>();
         let runner_id = RunnerId {
             device: device_id,
@@ -351,6 +365,11 @@
                     "Service already initialized.".into(),
                 ));
             }
+            if !super::admission::same_hook(&bootstrap_hook, &existing.service.command_admission) {
+                return Err(ServiceCreationError::new(
+                    "Cached command bridge differs from bootstrap".into(),
+                ));
+            }
             return Ok(existing.clone());
         }
 
@@ -369,28 +388,43 @@
 
         // The service initialization function.
         let initialize_service = move || {
-            STATES.with(|state| {
-                let mut map = match state.try_borrow_mut() {
-                    Ok(map) => map,
-                    Err(err) => panic!(
-                        "The device service {:?} is already borrowed: {err}",
-                        core::any::type_name::<S>()
-                    ),
-                };
-
-                if service.is_some() && map.contains_key(&type_id) {
-                    callback.send(Err(())).unwrap();
-                } else {
-                    let service = service.unwrap_or_else(|| S::init(device_id));
-                    let utilities = service.utilities();
-
-                    map.entry(type_id)
-                        .or_insert_with(|| RefCell::new(Box::new(service)));
-                    callback
-                        .send(Ok(ChannelService { type_id, utilities }))
-                        .unwrap();
-                }
-            });
+            let initialize = || {
+                STATES.with(|state| {
+                    let mut map = match state.try_borrow_mut() {
+                        Ok(map) => map,
+                        Err(err) => panic!(
+                            "The device service {:?} is already borrowed: {err}",
+                            core::any::type_name::<S>()
+                        ),
+                    };
+
+                    if service.is_some() && map.contains_key(&type_id) {
+                        callback.send(Err(())).unwrap();
+                    } else {
+                        let service = service.unwrap_or_else(|| S::init(device_id));
+                        let utilities = service.utilities();
+                        let command_admission = service.command_admission();
+                        assert!(
+                            super::admission::same_hook(&bootstrap_hook, &command_admission),
+                            "Service command bridge differs from bootstrap"
+                        );
+
+                        map.entry(type_id)
+                            .or_insert_with(|| RefCell::new(Box::new(service)));
+                        callback
+                            .send(Ok(ChannelService {
+                                type_id,
+                                utilities,
+                                command_admission,
+                            }))
+                            .unwrap();
+                    }
+                })
+            };
+            match bootstrap_permit {
+                Some(permit) => permit.run(initialize),
+                None => initialize(),
+            }
         };
 
         // Same reason in [`send]` we need to call the function directly if we are on the runner
diff --git a/cubecl-common-0.11.0/src/device/handle/mod.rs b/cubecl-common-0.11.0/src/device/handle/mod.rs
--- a/cubecl-common-0.11.0/src/device/handle/mod.rs
+++ b/cubecl-common-0.11.0/src/device/handle/mod.rs
@@ -1,5 +1,10 @@
+mod admission;
 mod base;
 
+pub use admission::{
+    CommandAdmission, CommandAdmissionError, CommandKind, CommandParticipation, CommandRequest,
+    CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RunningCommandPermit,
+};
 pub use base::*;
 
 use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle, ServiceId};
@@ -34,6 +39,8 @@
     handle: I,
     service: ServiceId,
     cast: fn(&mut dyn Any) -> &mut S,
+    // Only Client::set_stream supplies this; None is never inferred as pinned.
+    command_stream: Option<cubecl_environment::stream::StreamId>,
 }
 
 impl<S: ?Sized, I: DeviceHandleSpec> Clone for DeviceHandle<S, I> {
@@ -42,6 +49,7 @@
             handle: self.handle.clone(),
             service: self.service,
             cast: self.cast,
+            command_stream: self.command_stream,
         }
     }
 }
@@ -60,6 +68,7 @@
             handle: I::insert::<S>(device_id, service)?,
             service: ServiceId::of::<S>(device_id),
             cast: downcast::<S>,
+            command_stream: None,
         })
     }
 
@@ -68,6 +77,7 @@
             handle: I::new::<S>(device_id),
             service: ServiceId::of::<S>(device_id),
             cast: downcast::<S>,
+            command_stream: None,
         }
     }
 }
@@ -85,6 +95,7 @@
             handle: self.handle,
             service: self.service,
             cast,
+            command_stream: self.command_stream,
         }
     }
 
@@ -107,12 +118,33 @@
         task: T,
     ) -> Result<R, CallError> {
         let cast = self.cast;
-        self.handle.submit_blocking(move |state| task(cast(state)))
+        let hook = self.handle.command_admission();
+        if hook.is_none() {
+            return self.handle.submit_blocking(move |state| task(cast(state)));
+        }
+        let permit = self.command_permit(hook.as_ref(), CommandKind::SubmitBlocking, None)?;
+        self.handle.submit_blocking(move |state| {
+            permit
+                .expect("registered hook returns owned permit")
+                .run(|| task(cast(state)))
+        })
     }
 
     pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
         let cast = self.cast;
-        self.handle.submit(move |state| task(cast(state)))
+        let hook = self.handle.command_admission();
+        if hook.is_none() {
+            self.handle.submit(move |state| task(cast(state)));
+            return;
+        }
+        let permit = self
+            .command_permit(hook.as_ref(), CommandKind::Submit, None)
+            .unwrap_or_resume();
+        self.handle.submit(move |state| {
+            permit
+                .expect("registered hook returns owned permit")
+                .run(|| task(cast(state)))
+        })
     }
 
     pub fn flush_queue(&self) {
@@ -120,7 +152,63 @@
     }
 
     pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
-        self.handle.exclusive(task)
+        let hook = self.handle.command_admission();
+        if hook.is_none() {
+            return self.handle.exclusive(task);
+        }
+        let permit = self.command_permit(hook.as_ref(), CommandKind::Exclusive, None)?;
+        self.handle.exclusive(move || {
+            permit
+                .expect("registered hook returns owned permit")
+                .run(task)
+        })
+    }
+
+    /// This first transport hook never issues complete CUDA participation.
+    pub fn command_participation(&self) -> CommandParticipation {
+        CommandParticipation::Unproven
+    }
+
+    /// Mirror the existing Client's explicit logical stream; not an authority.
+    ///
+    /// # Safety
+    /// Must match the actual stream argument on every command from this handle.
+    pub unsafe fn set_command_stream(&mut self, stream: cubecl_environment::stream::StreamId) {
+        self.command_stream = Some(stream);
+    }
+
+    /// Run on the original actor with explicit original capture-epoch ownership.
+    /// Does not begin capture or grant complete same-card participation.
+    pub fn exclusive_with_command_scope<R: Send, T: FnOnce() -> R + Send>(
+        &self,
+        scope: &mut CommandScope,
+        task: T,
+    ) -> Result<R, CallError> {
+        let hook = self.handle.command_admission();
+        let permit = self.command_permit(hook.as_ref(), CommandKind::Exclusive, Some(scope))?;
+        self.handle.exclusive(move || {
+            permit
+                .expect("validated scope requires registered hook")
+                .run(task)
+        })
+    }
+
+    fn command_permit(
+        &self,
+        hook: Option<&cubecl_environment::sync::Arc<dyn CommandAdmission>>,
+        kind: CommandKind,
+        scope: Option<&CommandScope>,
+    ) -> Result<Option<OwnedCommandPermit>, CallError> {
+        admission::admit(
+            hook,
+            CommandRequest {
+                service: self.service,
+                stream: self.command_stream,
+                kind,
+                scope,
+            },
+        )
+        .map_err(CallError::from_admission)
     }
 
     /// Stops the background runner threads for `device_id`, blocking until they
diff --git a/cubecl-common-0.11.0/src/device/handle/mutex.rs b/cubecl-common-0.11.0/src/device/handle/mutex.rs
--- a/cubecl-common-0.11.0/src/device/handle/mutex.rs
+++ b/cubecl-common-0.11.0/src/device/handle/mutex.rs
@@ -19,6 +19,7 @@
     service: Arc<Mutex<Box<dyn Any + Send>>>,
     device_id: DeviceId,
     utilities: ServerUtilitiesHandle,
+    command_admission: Option<Arc<dyn super::CommandAdmission>>,
 }
 
 /// The global storage for all device services.
@@ -33,6 +34,10 @@
     const BLOCKING: bool = true;
 
     fn new<S: DeviceService>(device_id: DeviceId) -> Self {
+        let (bootstrap_hook, bootstrap_permit) =
+            super::admission::bootstrap::<S>(device_id, super::CommandKind::LocateOrInitialize)
+                .unwrap_or_else(|e| panic!("{e}"));
+        let _entered = bootstrap_permit.map(|permit| permit.owner_enter_for_transport());
         let mut guard = DEVICE_REGISTRY.lock();
         if guard.is_none() {
             *guard = Some(HashMap::new());
@@ -49,13 +54,23 @@
             .or_insert_with(|| {
                 let state = S::init(device_id);
                 let utilities = state.utilities();
+                let command_admission = state.command_admission();
+                assert!(
+                    super::admission::same_hook(&bootstrap_hook, &command_admission),
+                    "Service command bridge differs from bootstrap"
+                );
                 MutexDeviceState {
                     service: Arc::new(Mutex::new(Box::new(state))),
                     device_id,
                     utilities,
+                    command_admission,
                 }
             })
             .clone();
+        assert!(
+            super::admission::same_hook(&bootstrap_hook, &state.command_admission),
+            "Cached command bridge differs from bootstrap"
+        );
 
         Self { state, device_id }
     }
@@ -66,6 +81,10 @@
 
     fn utilities(&self) -> ServerUtilitiesHandle {
         self.state.utilities.clone()
+    }
+
+    fn command_admission(&self) -> Option<Arc<dyn super::CommandAdmission>> {
+        self.state.command_admission.clone()
     }
 
     fn flush_queue(&self) {}
@@ -89,6 +108,10 @@
         device_id: DeviceId,
         service: S,
     ) -> Result<Self, ServiceCreationError> {
+        let (bootstrap_hook, bootstrap_permit) =
+            super::admission::bootstrap::<S>(device_id, super::CommandKind::Install)
+                .map_err(|e| ServiceCreationError::new(e.message().into()))?;
+        let _entered = bootstrap_permit.map(|permit| permit.owner_enter_for_transport());
         let mut guard = DEVICE_REGISTRY.lock();
         if guard.is_none() {
             *guard = Some(HashMap::new());
@@ -108,10 +131,16 @@
             .entry(type_id)
             .or_insert_with(|| {
                 let utilities = service.utilities();
+                let command_admission = service.command_admission();
+                assert!(
+                    super::admission::same_hook(&bootstrap_hook, &command_admission),
+                    "Service command bridge differs from bootstrap"
+                );
                 MutexDeviceState {
                     service: Arc::new(Mutex::new(Box::new(service))),
                     device_id,
                     utilities,
+                    command_admission,
                 }
             })
             .clone();
diff --git a/cubecl-common-0.11.0/src/device/handle/reentrant.rs b/cubecl-common-0.11.0/src/device/handle/reentrant.rs
--- a/cubecl-common-0.11.0/src/device/handle/reentrant.rs
+++ b/cubecl-common-0.11.0/src/device/handle/reentrant.rs
@@ -18,15 +18,18 @@
     type_name: &'static str,
     /// Builds the service on first use, since the map is filled lazily.
     init: fn(DeviceId) -> ReentrantMutexDeviceState,
+    command_admission: Option<Arc<dyn super::CommandAdmission>>,
 }
 
 /// The state for a service of type `S` on `device_id`, built on first use.
 fn init_state<S: DeviceService>(device_id: DeviceId) -> ReentrantMutexDeviceState {
     let service = S::init(device_id);
     let utilities = service.utilities();
+    let command_admission = service.command_admission();
     ReentrantMutexDeviceState {
         service: Cell::new(Some(Box::new(service))),
         utilities,
+        command_admission,
     }
 }
 
@@ -59,6 +62,10 @@
             .clone()
     }
 
+    fn command_admission(&self) -> Option<Arc<dyn super::CommandAdmission>> {
+        self.command_admission.clone()
+    }
+
     fn flush_queue(&self) {}
 
     fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
@@ -88,6 +95,7 @@
             type_id: self.type_id,
             type_name: self.type_name,
             init: self.init,
+            command_admission: self.command_admission.clone(),
         }
     }
 }
@@ -121,6 +129,21 @@
         state_new: S,
     ) -> Result<Self, alloc::string::String> {
         let lock = Self::locate::<S>(device);
+        let install = super::admission::admit(
+            lock.command_admission.as_ref(),
+            super::CommandRequest {
+                service: crate::device::ServiceId::of::<S>(device),
+                stream: None,
+                kind: super::CommandKind::Install,
+                scope: None,
+            },
+        )
+        .map_err(|e| alloc::string::String::from(e.message()))?;
+        let _entered = install.map(|permit| permit.owner_enter_for_transport());
+        let actual_hook = state_new.command_admission();
+        if !super::admission::same_hook(&lock.command_admission, &actual_hook) {
+            return Err("Service command bridge differs from bootstrap".into());
+        }
         let id = TypeId::of::<S>();
 
         let state = lock.lock.lock.lock();
@@ -139,6 +162,7 @@
             ReentrantMutexDeviceState {
                 service: Cell::new(Some(any)),
                 utilities,
+                command_admission: actual_hook,
             },
         );
 
@@ -165,6 +189,32 @@
     ///
     /// If the same state type is locked multiple times on the same thread.
     fn with_lock<R>(&self, f: impl FnOnce(&mut dyn Any) -> R) -> R {
+        // Only a cold guarded service needs a bootstrap permit. Inspect cached
+        // metadata, release the lock, THEN admit before the actual init lock.
+        // Warm scoped calls retain their explicit Client stream; no ambient
+        // stream is inferred for a second nested admission.
+        let initialize = if self.command_admission.is_some() && {
+            let state = self.lock.lock.lock();
+            let map = state.map.borrow();
+            !map.contains_key(&self.type_id)
+        } {
+            super::admission::admit(
+                self.command_admission.as_ref(),
+                super::CommandRequest {
+                    service: crate::device::ServiceId {
+                        device: self.device_id,
+                        service: self.type_id,
+                    },
+                    stream: None,
+                    kind: super::CommandKind::LocateOrInitialize,
+                    scope: None,
+                },
+            )
+            .unwrap_or_else(|e| panic!("{e}"))
+        } else {
+            None
+        };
+        let _entered = initialize.map(|permit| permit.owner_enter_for_transport());
         let key = self.type_id;
         let state = self.lock.lock.lock();
 
@@ -174,7 +224,17 @@
         let entry = {
             let mut map = state.map.borrow_mut();
             map.entry(key)
-                .or_insert_with(|| (self.init)(self.device_id))
+                .or_insert_with(|| {
+                    let initialized = (self.init)(self.device_id);
+                    assert!(
+                        super::admission::same_hook(
+                            &self.command_admission,
+                            &initialized.command_admission
+                        ),
+                        "Service command bridge differs from bootstrap"
+                    );
+                    initialized
+                })
                 .service
                 .take()
         };
@@ -244,10 +304,13 @@
     /// `None` means the state is currently borrowed by a `with_lock` call.
     service: Cell<Option<Box<dyn Any + Send + 'static>>>,
     utilities: ServerUtilitiesHandle,
+    command_admission: Option<Arc<dyn super::CommandAdmission>>,
 }
 
 impl DeviceStateLock {
     fn locate<S: DeviceService>(device: DeviceId) -> ReentrantMutexDeviceHandle {
+        // Cached metadata lookup MUST stay outside GLOBAL/device locks.
+        let command_admission = S::bootstrap_command_admission(device);
         let mut global = GLOBAL.lock();
 
         let locator_state = match &mut global.state {
@@ -282,6 +345,7 @@
             type_id: TypeId::of::<S>(),
             type_name: core::any::type_name::<S>(),
             init: init_state::<S>,
+            command_admission,
         }
     }
 }
diff --git a/cubecl-runtime-0.11.0/src/client.rs b/cubecl-runtime-0.11.0/src/client.rs
--- a/cubecl-runtime-0.11.0/src/client.rs
+++ b/cubecl-runtime-0.11.0/src/client.rs
@@ -22,7 +22,7 @@
 mod lazy;
 use cubecl_common::{
     bytes::{AllocationProperty, Bytes},
-    device::{DeviceId, ServiceId},
+    device::{CommandParticipation, CommandScope, DeviceId, ServiceId},
     device_handle::{CallResultExt, DeviceHandle},
     profile::ProfileDuration,
 };
@@ -252,6 +252,32 @@
         self.device.service_id()
     }
 
+    /// Transport hooks alone cannot authorize production CUDA capture.
+    pub fn command_participation(&self) -> CommandParticipation {
+        self.device.command_participation()
+    }
+
+    /// Existing actor serialization with an explicitly owned original epoch.
+    /// An unpinned Client is refused before enqueue; does not begin CUDA capture.
+    pub fn exclusive_with_command_scope<R: Send, T: FnOnce() -> R + Send>(
+        &self,
+        scope: &mut CommandScope,
+        task: T,
+    ) -> Result<R, ServerError> {
+        if self.stream_id != Some(scope.stream_id()) || self.service_id() != scope.service_id() {
+            return Err(ServerError::Generic {
+                reason: "Command scope requires the same explicitly pinned original Client".into(),
+                backtrace: BackTrace::capture(),
+            });
+        }
+        self.device
+            .exclusive_with_command_scope(scope, task)
+            .map_err(|err| ServerError::Generic {
+                reason: format!("{err:?}"),
+                backtrace: BackTrace::capture(),
+            })
+    }
+
     /// Whether the server behind this client is an `S`. The client is erased
     /// over its server type, so a caller naming one has to be checked here,
     /// before a downcast on the device thread turns the mismatch into a panic.
@@ -291,6 +317,8 @@
     /// This is highly unsafe and should probably only be used by the CubeCL/Burn projects for now.
     pub unsafe fn set_stream(&mut self, stream_id: StreamId) {
         self.stream_id = Some(stream_id);
+        // SAFETY: this is the same explicit stream used by every Client command.
+        unsafe { self.device.set_command_stream(stream_id) };
     }
 
     fn do_read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/HOST_HARNESS_ONLY.patch

Original UTF-8 bytes: 1371; SHA256 `408576c0539ffed2cd1f1fb01439f99012953c68ef492839969ffd6ec73dbbc3`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/src/format.rs b/src/format.rs
--- a/src/format.rs
+++ b/src/format.rs
@@ -110,7 +110,7 @@
     name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
 }
 
-#[cfg(test)]
+#[cfg(all(test, not(zg_host_admission_only)))]
 #[cfg(feature = "std")]
 mod tests {
     use alloc::string::ToString;
diff --git a/src/benchmark.rs b/src/benchmark.rs
--- a/src/benchmark.rs
+++ b/src/benchmark.rs
@@ -398,7 +398,7 @@
     })
 }
 
-#[cfg(test)]
+#[cfg(all(test, not(zg_host_admission_only)))]
 #[cfg(feature = "std")]
 mod tests {
     use super::*;
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -52,11 +52,6 @@
 [lib]
 name = "cubecl_common"
 path = "src/lib.rs"
-
-[[bench]]
-name = "device_handle"
-path = "benches/device_handle.rs"
-harness = false
 
 [dependencies.bytemuck]
 version = "1.16.1"
@@ -143,18 +138,6 @@
 optional = true
 default-features = false
 
-[dev-dependencies.criterion]
-version = "0.8"
-features = ["html_reports"]
-
-[dev-dependencies.tempfile]
-version = "3.20"
-
-[dev-dependencies.test-log]
-version = "^0.2"
-features = ["trace"]
-default-features = false
-
 [build-dependencies.cfg_aliases]
 version = "0.2.2"
 
@@ -198,3 +181,6 @@
 [lints.rustdoc]
 broken_intra_doc_links = "warn"
 invalid_html_tags = "warn"
+
+# Standalone ROOT-owned host fixture harness; original source deps unchanged.
+[workspace]

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/frozen/HOST_HARNESS_ONLY.patch

Original UTF-8 bytes: 1371; SHA256 `408576c0539ffed2cd1f1fb01439f99012953c68ef492839969ffd6ec73dbbc3`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/src/format.rs b/src/format.rs
--- a/src/format.rs
+++ b/src/format.rs
@@ -110,7 +110,7 @@
     name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
 }
 
-#[cfg(test)]
+#[cfg(all(test, not(zg_host_admission_only)))]
 #[cfg(feature = "std")]
 mod tests {
     use alloc::string::ToString;
diff --git a/src/benchmark.rs b/src/benchmark.rs
--- a/src/benchmark.rs
+++ b/src/benchmark.rs
@@ -398,7 +398,7 @@
     })
 }
 
-#[cfg(test)]
+#[cfg(all(test, not(zg_host_admission_only)))]
 #[cfg(feature = "std")]
 mod tests {
     use super::*;
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -52,11 +52,6 @@
 [lib]
 name = "cubecl_common"
 path = "src/lib.rs"
-
-[[bench]]
-name = "device_handle"
-path = "benches/device_handle.rs"
-harness = false
 
 [dependencies.bytemuck]
 version = "1.16.1"
@@ -143,18 +138,6 @@
 optional = true
 default-features = false
 
-[dev-dependencies.criterion]
-version = "0.8"
-features = ["html_reports"]
-
-[dev-dependencies.tempfile]
-version = "3.20"
-
-[dev-dependencies.test-log]
-version = "^0.2"
-features = ["trace"]
-default-features = false
-
 [build-dependencies.cfg_aliases]
 version = "0.2.2"
 
@@ -198,3 +181,6 @@
 [lints.rustdoc]
 broken_intra_doc_links = "warn"
 invalid_html_tags = "warn"
+
+# Standalone ROOT-owned host fixture harness; original source deps unchanged.
+[workspace]

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/RETIREMENT.patch

Original UTF-8 bytes: 52029; SHA256 `d4e875f79a3b56e07c74625bdc25f63c2dcee1d66f2c178e0b0e187a23251eb9`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/cubecl-common-0.11.0/src/device/base.rs b/cubecl-common-0.11.0/src/device/base.rs
--- a/cubecl-common-0.11.0/src/device/base.rs
+++ b/cubecl-common-0.11.0/src/device/base.rs
@@ -7,7 +7,10 @@
 // Public contract exports through device::base, preserving device_handle ABI.
 pub use super::handle::{
     CommandAdmission, CommandAdmissionError, CommandKind, CommandParticipation, CommandRequest,
-    CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RunningCommandPermit,
+    CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RetirementAuthority,
+    RetirementExecution, RetirementExecutionOwner, RetirementFailure, RetirementHandle,
+    RetirementKind, RetirementPrepareError, RetirementRequest, RetirementReservation,
+    RetirementTask, RunningCommandPermit,
 };
 
 /// The device id.
diff --git a/cubecl-common-0.11.0/src/device/handle/admission.rs b/cubecl-common-0.11.0/src/device/handle/admission.rs
--- a/cubecl-common-0.11.0/src/device/handle/admission.rs
+++ b/cubecl-common-0.11.0/src/device/handle/admission.rs
@@ -93,6 +93,15 @@
         &self,
         request: CommandRequest<'_>,
     ) -> Result<OwnedCommandPermit, CommandAdmissionError>;
+
+    /// Same original QueueState owns the resource's prepaid cleanup work.
+    /// Default CLOSED; command presence alone cannot supply retirement.
+    fn retirement_authority(
+        &self,
+        _request: super::RetirementRequest,
+    ) -> Result<Arc<dyn super::RetirementAuthority>, CommandAdmissionError> {
+        Err(CommandAdmissionError::Unproven)
+    }
 }
 
 /// Queued ownership moves into the actual original actor closure.
diff --git a/cubecl-common-0.11.0/src/device/handle/base.rs b/cubecl-common-0.11.0/src/device/handle/base.rs
--- a/cubecl-common-0.11.0/src/device/handle/base.rs
+++ b/cubecl-common-0.11.0/src/device/handle/base.rs
@@ -174,6 +174,18 @@
         None
     }
 
+    /// Prepaid ORIGINAL-runner storage, before constructing the resource.
+    /// Unsupported transports must refuse before effects, never ordinary-submit
+    /// a destructor that could block or reenter a borrowed service.
+    fn reserve_retirement(
+        &self,
+        _request: super::RetirementRequest,
+        _authority: cubecl_environment::sync::Arc<dyn super::RetirementAuthority>,
+        _task: cubecl_environment::sync::Arc<dyn super::RetirementTask>,
+    ) -> Result<super::RetirementReservation, super::RetirementPrepareError> {
+        Err(super::RetirementPrepareError::Unsupported)
+    }
+
     /// Doesn't flush the service state, but flushes any task enqueued in the communication
     /// channel.
     ///
diff --git a/cubecl-common-0.11.0/src/device/handle/channel.rs b/cubecl-common-0.11.0/src/device/handle/channel.rs
--- a/cubecl-common-0.11.0/src/device/handle/channel.rs
+++ b/cubecl-common-0.11.0/src/device/handle/channel.rs
@@ -68,6 +68,17 @@
         &self,
     ) -> Option<cubecl_environment::sync::Arc<dyn super::CommandAdmission>> {
         self.state.service.command_admission.clone()
+    }
+
+    fn reserve_retirement(
+        &self,
+        request: super::RetirementRequest,
+        authority: cubecl_environment::sync::Arc<dyn super::RetirementAuthority>,
+        task: cubecl_environment::sync::Arc<dyn super::RetirementTask>,
+    ) -> Result<super::RetirementReservation, super::RetirementPrepareError> {
+        self.state
+            .client
+            .reserve_retirement(request, authority, task)
     }
 
     /// Runs `task` on the device thread, blocking until it returns.
@@ -927,6 +938,21 @@
             (Self { state }, thread)
         }
 
+        pub(super) fn reserve_retirement(
+            &self,
+            request: crate::device::handle::RetirementRequest,
+            authority: Arc<dyn crate::device::handle::RetirementAuthority>,
+            task: Arc<dyn crate::device::handle::RetirementTask>,
+        ) -> Result<
+            crate::device::handle::RetirementReservation,
+            crate::device::handle::RetirementPrepareError,
+        > {
+            self.state
+                .retirements
+                .get_or_init(crate::device::handle::retirement::RetirementMailbox::new)
+                .reserve(request, authority, task)
+        }
+
         /// Signals the server to run any remaining tasks and stop once every
         /// client is dropped. Queued tasks may themselves hold clients, so the
         /// server keeps draining until the last one is gone.
@@ -979,6 +1005,8 @@
         shutdown: AtomicBool,
         /// The runner id (for debugging purposes).
         runner_id: RunnerId,
+        // Lazy only for an explicitly admitted resource; None allocates nothing.
+        retirements: std::sync::OnceLock<Arc<crate::device::handle::retirement::RetirementMailbox>>,
     }
 
     impl State {
@@ -1071,6 +1099,7 @@
                 enqueued_count: AtomicU32::new(0),
                 shutdown: AtomicBool::new(false),
                 runner_id,
+                retirements: std::sync::OnceLock::new(),
             });
 
             Self {
@@ -1085,6 +1114,10 @@
         fn start(&mut self) {
             let mut idle_count: u32 = 0;
             loop {
+                // Same ORIGINAL runner, outside every task/service borrow.
+                if let Some(mailbox) = self.state.retirements.get() {
+                    mailbox.pump();
+                }
                 if self.ready_to_execute {
                     self.execute_tasks();
                     idle_count = 0;
@@ -1160,6 +1193,10 @@
             let server_buf = 1 - self.client_buf;
             for task in &mut self.buffers[server_buf].tasks {
                 task.run();
+                // Drop publication never waits on the full32 ordinary queue.
+                if let Some(mailbox) = self.state.retirements.get() {
+                    mailbox.pump();
+                }
             }
             self.ready_to_execute = false;
         }
diff --git a/cubecl-common-0.11.0/src/device/handle/mod.rs b/cubecl-common-0.11.0/src/device/handle/mod.rs
--- a/cubecl-common-0.11.0/src/device/handle/mod.rs
+++ b/cubecl-common-0.11.0/src/device/handle/mod.rs
@@ -1,11 +1,18 @@
 mod admission;
 mod base;
+mod retirement;
 
 pub use admission::{
     CommandAdmission, CommandAdmissionError, CommandKind, CommandParticipation, CommandRequest,
     CommandScope, CommandScopeOwner, OwnedCommandPermit, PostedCommandPermit, RunningCommandPermit,
 };
 pub use base::*;
+
+pub use retirement::{
+    RetirementAuthority, RetirementExecution, RetirementExecutionOwner, RetirementFailure,
+    RetirementHandle, RetirementKind, RetirementPrepareError, RetirementRequest,
+    RetirementReservation, RetirementTask,
+};
 
 use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle, ServiceId};
 use core::any::Any;
@@ -162,6 +169,33 @@
                 .expect("registered hook returns owned permit")
                 .run(task)
         })
+    }
+
+    /// Prepare action + original-runner slot BEFORE physical construction.
+    /// None preserves the original API/path without action or mailbox allocation.
+    pub fn prepare_retirement<T: RetirementTask>(
+        &self,
+        kind: RetirementKind,
+        make_task: impl FnOnce() -> cubecl_environment::sync::Arc<T>,
+    ) -> Result<
+        Option<(RetirementReservation, cubecl_environment::sync::Arc<T>)>,
+        RetirementPrepareError,
+    > {
+        let Some(hook) = self.handle.command_admission() else {
+            return Ok(None);
+        };
+        let request = RetirementRequest {
+            service: self.service,
+            stream: self.command_stream,
+            kind,
+        };
+        let authority = hook
+            .retirement_authority(request)
+            .map_err(RetirementPrepareError::Admission)?;
+        let task = make_task();
+        let erased: cubecl_environment::sync::Arc<dyn RetirementTask> = task.clone();
+        let reserved = self.handle.reserve_retirement(request, authority, erased)?;
+        Ok(Some((reserved, task)))
     }
 
     /// This first transport hook never issues complete CUDA participation.
diff --git a/cubecl-common-0.11.0/src/device/handle/retirement/tests.rs b/cubecl-common-0.11.0/src/device/handle/retirement/tests.rs
new file mode 100644
--- /dev/null
+++ b/cubecl-common-0.11.0/src/device/handle/retirement/tests.rs
@@ -0,0 +1,430 @@
+// Host transport fixtures only. No physical card, CUDA, Ready or model claim.
+use super::*;
+use core::sync::atomic::AtomicUsize;
+use std::sync::{Mutex, mpsc};
+use std::vec::Vec;
+
+#[derive(Default)]
+struct State {
+    open: AtomicBool,
+    started: AtomicUsize,
+    entered: AtomicBool,
+    finished: AtomicUsize,
+    unknown: AtomicUsize,
+    fenced: AtomicBool,
+    close_on_start: AtomicBool,
+    expected_actor: Mutex<Option<std::thread::ThreadId>>,
+    retained: Mutex<Vec<Arc<dyn core::any::Any + Send + Sync>>>,
+}
+
+struct Authority {
+    state: Arc<State>,
+    execution: Arc<Execution>,
+}
+
+struct Execution(Arc<State>);
+
+// SAFETY: synthetic host counters only; all owner storage is preallocated.
+// No physical QueueState binding/Complete issuance is claimed by this fixture.
+unsafe impl RetirementAuthority for Authority {
+    fn try_start(
+        &self,
+        _: RetirementRequest,
+    ) -> Result<Option<RetirementExecution>, CommandAdmissionError> {
+        if self.state.fenced.load(Ordering::SeqCst) {
+            return Err(CommandAdmissionError::Fenced);
+        }
+        if !self.state.open.load(Ordering::SeqCst) {
+            return Ok(None);
+        }
+        self.state.started.fetch_add(1, Ordering::SeqCst);
+        if self.state.close_on_start.load(Ordering::SeqCst) {
+            self.state.open.store(false, Ordering::SeqCst);
+        }
+        // SAFETY: models already-admitted host work; no hardware authority.
+        Ok(Some(unsafe {
+            RetirementExecution::from_original_work(self.execution.clone())
+        }))
+    }
+
+    fn fence_unknown(&self, owner: Option<Arc<dyn core::any::Any + Send + Sync>>) {
+        self.state.fenced.store(true, Ordering::SeqCst);
+        if let Some(owner) = owner {
+            self.state.retained.lock().unwrap().push(owner);
+        }
+    }
+}
+
+// SAFETY: exact original actor ThreadId, when supplied, is checked from actual
+// host transport execution. These counters have no GPU-terminal meaning.
+unsafe impl RetirementExecutionOwner for Execution {
+    fn enter(&self) -> Result<(), CommandAdmissionError> {
+        if self
+            .0
+            .expected_actor
+            .lock()
+            .unwrap()
+            .is_some_and(|expected| expected != std::thread::current().id())
+        {
+            return Err(CommandAdmissionError::ForeignScope);
+        }
+        self.0.entered.store(true, Ordering::SeqCst);
+        Ok(())
+    }
+    fn known_terminal(&self) {
+        self.0.entered.store(false, Ordering::SeqCst);
+        self.0.finished.fetch_add(1, Ordering::SeqCst);
+    }
+    fn unknown(&self, _: bool) {
+        self.0.entered.store(false, Ordering::SeqCst);
+        self.0.unknown.fetch_add(1, Ordering::SeqCst);
+        self.0.fenced.store(true, Ordering::SeqCst);
+    }
+}
+
+#[derive(Default)]
+struct Observed {
+    calls: AtomicUsize,
+    drops: AtomicUsize,
+}
+
+enum Outcome {
+    Known,
+    Unknown,
+    Panic,
+}
+
+struct Task {
+    state: Arc<State>,
+    observed: Arc<Observed>,
+    outcome: Outcome,
+}
+
+// SAFETY: no native resource; execute observes entered host ownership; Drop
+// only counts, never invokes a driver. Error/panic remains mailbox-owned.
+unsafe impl RetirementTask for Task {
+    fn execute(&self) -> Result<(), RetirementFailure> {
+        assert!(self.state.entered.load(Ordering::SeqCst));
+        self.observed.calls.fetch_add(1, Ordering::SeqCst);
+        match self.outcome {
+            Outcome::Known => Ok(()),
+            Outcome::Unknown => Err(RetirementFailure::Unknown),
+            Outcome::Panic => panic!("host retirement fixture"),
+        }
+    }
+}
+
+impl Drop for Task {
+    fn drop(&mut self) {
+        self.observed.drops.fetch_add(1, Ordering::SeqCst);
+    }
+}
+
+fn authority() -> (Arc<Authority>, Arc<State>) {
+    let state = Arc::new(State::default());
+    state.open.store(true, Ordering::SeqCst);
+    (
+        Arc::new(Authority {
+            state: state.clone(),
+            execution: Arc::new(Execution(state.clone())),
+        }),
+        state,
+    )
+}
+
+fn request() -> RetirementRequest {
+    RetirementRequest {
+        service: ServiceId::of::<Task>(crate::device::DeviceId {
+            type_id: 0,
+            index_id: 0,
+        }),
+        stream: Some(StreamId { value: 7 }),
+        kind: RetirementKind::Graph,
+    }
+}
+
+fn task(state: &Arc<State>, outcome: Outcome) -> (Arc<Task>, Arc<Observed>) {
+    let observed = Arc::new(Observed::default());
+    (
+        Arc::new(Task {
+            state: state.clone(),
+            observed: observed.clone(),
+            outcome,
+        }),
+        observed,
+    )
+}
+
+#[test]
+fn unarmed_failure_cancels_before_resource_creation() {
+    let mailbox = RetirementMailbox::new();
+    let (authority, state) = authority();
+    let (task, observed) = task(&state, Outcome::Known);
+    let reserved = mailbox.reserve(request(), authority, task).unwrap();
+    drop(reserved);
+    mailbox.pump();
+    assert_eq!(observed.calls.load(Ordering::SeqCst), 0);
+    assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
+    assert_eq!(mailbox.free.load(Ordering::Acquire), u64::MAX);
+}
+
+#[test]
+fn armed_drop_only_publishes_and_actor_runs_once_after_open() {
+    let mailbox = RetirementMailbox::new();
+    let (authority, state) = authority();
+    state.open.store(false, Ordering::SeqCst);
+    let (task, observed) = task(&state, Outcome::Known);
+    let armed = mailbox.reserve(request(), authority, task).unwrap().arm();
+    drop(armed);
+    assert_eq!(observed.calls.load(Ordering::SeqCst), 0);
+    mailbox.pump();
+    assert_eq!(observed.calls.load(Ordering::SeqCst), 0);
+    state.open.store(true, Ordering::SeqCst);
+    mailbox.pump();
+    mailbox.pump();
+    assert_eq!(observed.calls.load(Ordering::SeqCst), 1);
+    assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
+    assert_eq!(state.finished.load(Ordering::SeqCst), 1);
+    assert_eq!(mailbox.free.load(Ordering::Acquire), u64::MAX);
+}
+
+#[test]
+fn preclose_owned_cleanup_finishes_without_a_second_admission() {
+    let mailbox = RetirementMailbox::new();
+    let (authority, state) = authority();
+    state.close_on_start.store(true, Ordering::SeqCst);
+    let (task, observed) = task(&state, Outcome::Known);
+    drop(mailbox.reserve(request(), authority, task).unwrap().arm());
+    mailbox.pump();
+    assert!(!state.open.load(Ordering::SeqCst));
+    assert_eq!(state.started.load(Ordering::SeqCst), 1);
+    assert_eq!(state.finished.load(Ordering::SeqCst), 1);
+    assert_eq!(observed.calls.load(Ordering::SeqCst), 1);
+}
+
+#[test]
+fn bounded_capacity_refuses_before_construction_and_reuses_with_new_generation() {
+    let mailbox = RetirementMailbox::new();
+    let (authority, state) = authority();
+    let mut reserved = Vec::new();
+    for _ in 0..SLOTS {
+        let (task, _) = task(&state, Outcome::Known);
+        reserved.push(mailbox.reserve(request(), authority.clone(), task).unwrap());
+    }
+    let (extra, _) = task(&state, Outcome::Known);
+    assert!(matches!(
+        mailbox.reserve(request(), authority.clone(), extra),
+        Err(RetirementPrepareError::Capacity)
+    ));
+    let old = reserved[0].binding.as_ref().unwrap().generation;
+    drop(reserved.remove(0));
+    let (next, _) = task(&state, Outcome::Known);
+    let new = mailbox.reserve(request(), authority, next).unwrap();
+    assert!(new.binding.as_ref().unwrap().generation > old);
+}
+
+#[test]
+fn exhausted_identity_refuses_without_claiming_a_free_slot() {
+    let mailbox = RetirementMailbox::new();
+    mailbox.next.store(u64::MAX, Ordering::Release);
+    let (authority, state) = authority();
+    let (task, _) = task(&state, Outcome::Known);
+    assert!(matches!(
+        mailbox.reserve(request(), authority, task),
+        Err(RetirementPrepareError::IdentityExhausted)
+    ));
+    assert_eq!(mailbox.free.load(Ordering::Acquire), u64::MAX);
+}
+
+fn unknown_case(outcome: Outcome) {
+    let mailbox = RetirementMailbox::new();
+    let (authority, state) = authority();
+    let (first, observed) = task(&state, outcome);
+    drop(
+        mailbox
+            .reserve(request(), authority.clone(), first)
+            .unwrap()
+            .arm(),
+    );
+    mailbox.pump();
+    assert_eq!(observed.calls.load(Ordering::SeqCst), 1);
+    assert_eq!(observed.drops.load(Ordering::SeqCst), 0);
+    assert_eq!(state.finished.load(Ordering::SeqCst), 0);
+    assert_eq!(state.unknown.load(Ordering::SeqCst), 1);
+    assert!(state.fenced.load(Ordering::SeqCst));
+    assert_eq!(state.retained.lock().unwrap().len(), 1);
+    let (later, later_observed) = task(&state, Outcome::Known);
+    drop(mailbox.reserve(request(), authority, later).unwrap().arm());
+    mailbox.pump();
+    mailbox.pump();
+    assert_eq!(later_observed.calls.load(Ordering::SeqCst), 0);
+    assert_eq!(later_observed.drops.load(Ordering::SeqCst), 0);
+    assert_eq!(state.retained.lock().unwrap().len(), 1);
+    drop(mailbox);
+    assert_eq!(observed.drops.load(Ordering::SeqCst), 0);
+}
+
+#[test]
+fn failed_completion_retains_all_owners_and_never_retries() {
+    unknown_case(Outcome::Unknown);
+}
+
+#[test]
+fn actor_panic_retains_all_owners_and_never_retries() {
+    unknown_case(Outcome::Panic);
+}
+
+#[test]
+fn nested_pump_cannot_reenter_running_payload() {
+    struct Reentrant {
+        mailbox: Arc<RetirementMailbox>,
+        calls: Arc<AtomicUsize>,
+    }
+    // SAFETY: host-only nested transport probe; no native resource.
+    unsafe impl RetirementTask for Reentrant {
+        fn execute(&self) -> Result<(), RetirementFailure> {
+            self.calls.fetch_add(1, Ordering::SeqCst);
+            self.mailbox.pump();
+            Ok(())
+        }
+    }
+    let mailbox = RetirementMailbox::new();
+    let (authority, state) = authority();
+    let calls = Arc::new(AtomicUsize::new(0));
+    let task = Arc::new(Reentrant {
+        mailbox: mailbox.clone(),
+        calls: calls.clone(),
+    });
+    drop(mailbox.reserve(request(), authority, task).unwrap().arm());
+    mailbox.pump();
+    assert_eq!(calls.load(Ordering::SeqCst), 1);
+    assert_eq!(state.finished.load(Ordering::SeqCst), 1);
+    assert_eq!(mailbox.free.load(Ordering::Acquire), u64::MAX);
+}
+
+use crate::device::handle::{DeviceFixture, DeviceHandle, DeviceHandleSpec};
+use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle};
+
+struct HostService(usize);
+impl DeviceService for HostService {
+    fn init(_: DeviceId) -> Self {
+        Self(0)
+    }
+    fn utilities(&self) -> ServerUtilitiesHandle {
+        Arc::new(())
+    }
+}
+
+type Original = DeviceHandle<HostService, crate::device::handle::channel::ChannelDeviceHandle>;
+
+struct OriginalTask {
+    handle: Original,
+    state: Arc<State>,
+}
+
+// SAFETY: actual original actor, synthetic HOST service only. No GPU identity
+// or completion is claimed; callback is checked by the real transport borrow.
+unsafe impl RetirementTask for OriginalTask {
+    fn execute(&self) -> Result<(), RetirementFailure> {
+        assert!(self.state.entered.load(Ordering::SeqCst));
+        self.handle
+            .submit_blocking(|service| service.0 += 1)
+            .map_err(|_| RetirementFailure::Unknown)
+    }
+}
+
+fn original_fixture() -> DeviceFixture<Original> {
+    DeviceFixture::new(Original::new, Original::shutdown)
+}
+
+fn original_reservation(
+    handle: &Original,
+    state: &Arc<State>,
+    authority: Arc<Authority>,
+) -> RetirementReservation {
+    let task = Arc::new(OriginalTask {
+        handle: handle.clone(),
+        state: state.clone(),
+    });
+    handle
+        .handle
+        .reserve_retirement(
+            RetirementRequest {
+                service: handle.service_id(),
+                stream: None,
+                kind: RetirementKind::Graph,
+            },
+            authority,
+            task,
+        )
+        .unwrap()
+}
+
+#[test]
+fn default_none_never_constructs_a_retirement_task() {
+    let handle = original_fixture();
+    let (_, state) = authority();
+    let (task, _) = task(&state, Outcome::Known);
+    let called = AtomicBool::new(false);
+    let result = handle
+        .prepare_retirement(RetirementKind::Graph, || {
+            called.store(true, Ordering::SeqCst);
+            task
+        })
+        .unwrap();
+    assert!(result.is_none());
+    assert!(!called.load(Ordering::SeqCst));
+}
+
+#[test]
+fn original_runner_pumps_only_after_current_service_borrow_ends() {
+    let handle = original_fixture();
+    let (authority, state) = authority();
+    *state.expected_actor.lock().unwrap() = Some(
+        handle
+            .submit_blocking(|_| std::thread::current().id())
+            .unwrap(),
+    );
+    let armed = original_reservation(&handle, &state, authority).arm();
+    handle.submit_blocking(move |_| drop(armed)).unwrap();
+    assert_eq!(handle.submit_blocking(|service| service.0).unwrap(), 1);
+    assert_eq!(state.finished.load(Ordering::SeqCst), 1);
+}
+
+#[test]
+fn original_full32_ordinary_queue_cannot_block_prepaid_last_drop() {
+    let handle = original_fixture();
+    let (authority, state) = authority();
+    *state.expected_actor.lock().unwrap() = Some(
+        handle
+            .submit_blocking(|_| std::thread::current().id())
+            .unwrap(),
+    );
+    let armed = original_reservation(&handle, &state, authority).arm();
+    let (entered, wait_entered) = mpsc::channel();
+    let (release, wait_release) = mpsc::channel();
+    handle.submit(move |_| {
+        entered.send(()).unwrap();
+        wait_release.recv().unwrap();
+    });
+    handle.flush_queue();
+    wait_entered
+        .recv_timeout(std::time::Duration::from_secs(2))
+        .unwrap();
+    // The original actor is blocked in the first buffer; fill its other32.
+    for _ in 0..32 {
+        handle.submit(|_| ());
+    }
+    let (done, wait_done) = mpsc::channel();
+    let dropper = std::thread::spawn(move || {
+        drop(armed);
+        let _ = done.send(());
+    });
+    let published = wait_done.recv_timeout(std::time::Duration::from_secs(2));
+    // Release even when the assertion would fail; no stranded owned thread.
+    release.send(()).unwrap();
+    dropper.join().unwrap();
+    assert!(published.is_ok(), "Drop waited on the ordinary full queue");
+    assert_eq!(handle.submit_blocking(|service| service.0).unwrap(), 1);
+    assert_eq!(state.finished.load(Ordering::SeqCst), 1);
+}
diff --git a/cubecl-common-0.11.0/src/device/handle/retirement.rs b/cubecl-common-0.11.0/src/device/handle/retirement.rs
new file mode 100644
--- /dev/null
+++ b/cubecl-common-0.11.0/src/device/handle/retirement.rs
@@ -0,0 +1,451 @@
+//! Prepaid destruction on the ORIGINAL actor; no worker, driver or card lookup.
+use super::CommandAdmissionError;
+use crate::device::ServiceId;
+use core::fmt;
+use cubecl_environment::{stream::StreamId, sync::Arc};
+
+/// Resource whose actual constructor must reserve retirement first.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum RetirementKind {
+    /// Runtime-owned graph/executable.
+    Graph,
+    /// Event, including foreign-host fence/profiler owners.
+    Event,
+    /// Actual device allocation, not a borrowed managed handle.
+    DeviceMemory,
+    /// Actual pinned host allocation.
+    PinnedMemory,
+    /// Original CUDA stream.
+    Stream,
+    /// Final original service/runner state.
+    Service,
+    /// Original retained context.
+    Context,
+}
+
+/// Original actor and explicitly pinned stream; not physical-card discovery.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub struct RetirementRequest {
+    /// Exact existing service/transport identity.
+    pub service: ServiceId,
+    /// Explicit Client pin only; None is never inferred from ambient state.
+    pub stream: Option<StreamId>,
+    /// Resource owned by this prepaid slot.
+    pub kind: RetirementKind,
+}
+
+/// Refusal BEFORE actual resource construction/capture-end side effects.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum RetirementPrepareError {
+    /// Transport lacks an original-runner prepaid mailbox.
+    Unsupported,
+    /// Same original gate refused the reservation.
+    Admission(CommandAdmissionError),
+    /// Bounded actor mailbox is full.
+    Capacity,
+    /// Slot identity cannot advance without reuse.
+    IdentityExhausted,
+}
+
+impl fmt::Display for RetirementPrepareError {
+    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+        write!(f, "retirement preparation refused: {self:?}")
+    }
+}
+
+/// Cleanup failed/unknown. The original resource and actor stay retained.
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum RetirementFailure {
+    /// No trusted terminal outcome; do not retry or destroy from Drop.
+    Unknown,
+}
+
+/// Preallocated action; called only by the ORIGINAL runner after normal borrows.
+///
+/// # Safety
+/// Own the actual original actor/resource; execute only that resource's checked
+/// destruction on that actor. Drop itself must never call a driver or wait.
+/// Ok means known terminal destruction, not merely enqueue/host acceptance.
+/// Err/panic leaves enough original ownership for process quarantine. Never
+/// derive Ready/root authority or substitute model math. Resource identifiers
+/// are filled only after the successful original constructor, not guessed.
+pub unsafe trait RetirementTask: Send + Sync + 'static {
+    /// Known terminal cleanup or unknown; no retry is performed on failure.
+    fn execute(&self) -> Result<(), RetirementFailure>;
+}
+
+/// Bridge into the SAME physical QueueState/ResourceArbiter, never a scheduler.
+///
+/// # Safety
+/// try_start never waits. Under the original queue mutex it either parks, or
+/// admits this cleanup as owned work BEFORE a capture can finish draining. A
+/// pre-close cleanup continuation stays executable during Closing; new work
+/// during capture parks. Preserve original fair ordering/cancellation/epoch.
+/// Use the actual original actor/stream binding, not the numeric request alone.
+/// Unknown must fence the same original physical key and put the supplied
+/// owner in the EXISTING original completion quarantine (not a new registry).
+/// None means that mailbox owner was already retained. No driver, executor,
+/// discovery or recovery is performed here. All required
+/// state/owners must be prepaid before the actual resource is constructed.
+pub unsafe trait RetirementAuthority: Send + Sync + 'static {
+    /// Original QueueState authorizes without blocking the actor.
+    fn try_start(
+        &self,
+        request: RetirementRequest,
+    ) -> Result<Option<RetirementExecution>, CommandAdmissionError>;
+    /// Fence original card and retain owner through ORIGINAL quarantine.
+    /// Nonblocking/nonallocating control flow; never unwind or call a driver.
+    fn fence_unknown(&self, owner: Option<Arc<dyn core::any::Any + Send + Sync>>);
+}
+
+/// Original admitted cleanup work and actor-local reentry owner.
+///
+/// # Safety
+/// Created only from the same original queue's actual admitted work. All
+/// callbacks are nonblocking/nonallocating, never unwind, and execute on the
+/// original actor.
+/// enter validates actual actor/epoch and installs prepaid reentry without
+/// ordinary re-admission. known_terminal restores scope/retires original work;
+/// unknown restores any entered scope, retains/fences through original owners.
+/// Host closure acceptance alone must NEVER call known_terminal.
+pub unsafe trait RetirementExecutionOwner: Send + Sync + 'static {
+    /// Enter actual original actor scope; do not infer it from logical IDs.
+    fn enter(&self) -> Result<(), CommandAdmissionError>;
+    /// Checked resource destruction completed, on the original actor.
+    fn known_terminal(&self);
+    /// Work/driver completion is unknown; retain original owner and fence.
+    fn unknown(&self, entered: bool);
+}
+
+/// Move-only, actor-local cleanup continuation; not a hardware Ready token.
+pub struct RetirementExecution {
+    owner: Arc<dyn RetirementExecutionOwner>,
+    entered: bool,
+    terminal: bool,
+    _not_send: core::marker::PhantomData<alloc::rc::Rc<()>>,
+}
+
+impl RetirementExecution {
+    /// Attach an already prepaid original work owner; grants no eligibility.
+    ///
+    /// # Safety
+    /// owner owns the exact work admitted atomically with capture drain on the
+    /// same QueueState/key/epoch; it is not an independently created ledger.
+    pub unsafe fn from_original_work(owner: Arc<dyn RetirementExecutionOwner>) -> Self {
+        Self {
+            owner,
+            entered: false,
+            terminal: false,
+            _not_send: core::marker::PhantomData,
+        }
+    }
+
+    #[cfg(feature = "std")]
+    fn execute(mut self, task: &dyn RetirementTask) -> Result<(), RetirementFailure> {
+        self.owner.enter().map_err(|_| RetirementFailure::Unknown)?;
+        self.entered = true;
+        task.execute()?;
+        self.owner.known_terminal();
+        self.terminal = true;
+        Ok(())
+    }
+}
+
+impl Drop for RetirementExecution {
+    fn drop(&mut self) {
+        if !self.terminal {
+            self.owner.unknown(self.entered);
+        }
+    }
+}
+
+#[cfg(feature = "std")]
+pub use mailbox::{RetirementHandle, RetirementReservation};
+
+#[cfg(not(feature = "std"))]
+/// No prepaid original-runner mailbox is available on this transport.
+pub struct RetirementReservation {
+    unavailable: core::convert::Infallible,
+}
+
+#[cfg(not(feature = "std"))]
+impl RetirementReservation {
+    /// Unreachable without a supported original-runner reservation.
+    pub fn arm(self) -> RetirementHandle {
+        match self.unavailable {}
+    }
+}
+
+#[cfg(not(feature = "std"))]
+/// No prepaid original-runner mailbox is available on this transport.
+pub struct RetirementHandle {
+    _unavailable: core::convert::Infallible,
+}
+
+#[cfg(feature = "std")]
+pub(super) use mailbox::RetirementMailbox;
+
+#[cfg(feature = "std")]
+mod mailbox {
+    use super::*;
+    use core::{
+        cell::UnsafeCell,
+        sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
+    };
+    use std::panic::{AssertUnwindSafe, catch_unwind};
+
+    const SLOTS: usize = 64;
+    const FREE: u8 = 0;
+    const RESERVED: u8 = 1;
+    const PUBLISHED: u8 = 2;
+    const RUNNING: u8 = 3;
+    const QUARANTINED: u8 = 4;
+
+    struct Payload {
+        request: RetirementRequest,
+        authority: Arc<dyn RetirementAuthority>,
+        task: Arc<dyn RetirementTask>,
+    }
+
+    struct Slot {
+        generation: AtomicU64,
+        state: AtomicU8,
+        payload: UnsafeCell<Option<Payload>>,
+    }
+
+    // SAFETY: one reservation owns writes while RESERVED. Release publication
+    // transfers ownership to the sole ORIGINAL runner; only it pumps/takes it.
+    // A free slot cannot be reused until payload removal precedes free Release.
+    unsafe impl Sync for Slot {}
+
+    /// Lazy original-runner transport storage, not a new worker/card registry.
+    pub(in crate::device::handle) struct RetirementMailbox {
+        slots: [Slot; SLOTS],
+        free: AtomicU64,
+        published: AtomicU64,
+        next: AtomicU64,
+        pinned_unknown: AtomicBool,
+        pumping: AtomicBool,
+    }
+
+    impl RetirementMailbox {
+        pub(in crate::device::handle) fn new() -> Arc<Self> {
+            Arc::new(Self {
+                slots: core::array::from_fn(|_| Slot {
+                    generation: AtomicU64::new(0),
+                    state: AtomicU8::new(FREE),
+                    payload: UnsafeCell::new(None),
+                }),
+                free: AtomicU64::new(u64::MAX),
+                published: AtomicU64::new(0),
+                next: AtomicU64::new(1),
+                pinned_unknown: AtomicBool::new(false),
+                pumping: AtomicBool::new(false),
+            })
+        }
+
+        pub(in crate::device::handle) fn reserve(
+            self: &Arc<Self>,
+            request: RetirementRequest,
+            authority: Arc<dyn RetirementAuthority>,
+            task: Arc<dyn RetirementTask>,
+        ) -> Result<RetirementReservation, RetirementPrepareError> {
+            let generation = self
+                .next
+                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
+                .map_err(|_| RetirementPrepareError::IdentityExhausted)?;
+            let mut free = self.free.load(Ordering::Acquire);
+            let index = loop {
+                if free == 0 {
+                    return Err(RetirementPrepareError::Capacity);
+                }
+                let index = free.trailing_zeros() as usize;
+                match self.free.compare_exchange_weak(
+                    free,
+                    free & !(1u64 << index),
+                    Ordering::AcqRel,
+                    Ordering::Acquire,
+                ) {
+                    Ok(_) => break index,
+                    Err(current) => free = current,
+                }
+            };
+            let slot = &self.slots[index];
+            debug_assert_eq!(slot.state.load(Ordering::Acquire), FREE);
+            // SAFETY: successful free-bit acquisition owns this unused slot.
+            unsafe {
+                *slot.payload.get() = Some(Payload {
+                    request,
+                    authority,
+                    task,
+                })
+            };
+            slot.generation.store(generation, Ordering::Relaxed);
+            slot.state.store(RESERVED, Ordering::Release);
+            Ok(RetirementReservation {
+                binding: Some(Binding {
+                    mailbox: self.clone(),
+                    index,
+                    generation,
+                }),
+            })
+        }
+
+        /// ONLY the original Server::start/execute_tasks calls this. No other
+        /// thread may pump, including a Drop, gate opener or external GC worker.
+        pub(in crate::device::handle) fn pump(self: &Arc<Self>) {
+            // Sole runner is the required call site. Also encode exclusive
+            // payload access if an accidental/reentrant pump reaches here.
+            if self
+                .pumping
+                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
+                .is_err()
+            {
+                return;
+            }
+            struct PumpGuard<'a>(&'a AtomicBool);
+            impl Drop for PumpGuard<'_> {
+                fn drop(&mut self) {
+                    self.0.store(false, Ordering::Release);
+                }
+            }
+            let _pump = PumpGuard(&self.pumping);
+            let mut mask = self.published.load(Ordering::Acquire);
+            while mask != 0 {
+                let index = mask.trailing_zeros() as usize;
+                let bit = 1u64 << index;
+                mask &= !bit;
+                self.published.fetch_and(!bit, Ordering::AcqRel);
+                let slot = &self.slots[index];
+                if slot.state.load(Ordering::Acquire) != PUBLISHED {
+                    continue;
+                }
+                // SAFETY: the reservation relinquished writes on publication;
+                // the sole original runner owns all subsequent slot access.
+                let payload = unsafe { (&*slot.payload.get()).as_ref().unwrap() };
+                if self.pinned_unknown.load(Ordering::Acquire) {
+                    self.quarantine(slot, &payload.authority);
+                    continue;
+                }
+                let start = catch_unwind(AssertUnwindSafe(|| {
+                    payload.authority.try_start(payload.request)
+                }));
+                let execution = match start {
+                    Ok(Ok(Some(execution))) => execution,
+                    Ok(Ok(None)) => {
+                        self.published.fetch_or(bit, Ordering::Release);
+                        continue;
+                    }
+                    _ => {
+                        self.quarantine(slot, &payload.authority);
+                        continue;
+                    }
+                };
+                slot.state.store(RUNNING, Ordering::Release);
+                let result = catch_unwind(AssertUnwindSafe(|| {
+                    execution.execute(payload.task.as_ref())
+                }));
+                if !matches!(result, Ok(Ok(()))) {
+                    self.quarantine(slot, &payload.authority);
+                    continue;
+                }
+                // SAFETY: sole runner; no producer can reserve/reuse this slot
+                // until after the following payload take and free Release.
+                let retired = unsafe { (&mut *slot.payload.get()).take().unwrap() };
+                drop(retired);
+                slot.state.store(FREE, Ordering::Release);
+                self.free.fetch_or(bit, Ordering::Release);
+            }
+        }
+
+        fn quarantine(self: &Arc<Self>, slot: &Slot, authority: &Arc<dyn RetirementAuthority>) {
+            slot.state.store(QUARANTINED, Ordering::Release);
+            // EXACT existing completion-quarantine owner type; no new card
+            // registry/executor. Retain the whole bounded actor mailbox once.
+            let owner = if !self.pinned_unknown.swap(true, Ordering::AcqRel) {
+                Some(self.clone() as Arc<dyn core::any::Any + Send + Sync>)
+            } else {
+                None
+            };
+            let backup = owner.clone();
+            if catch_unwind(AssertUnwindSafe(|| authority.fence_unknown(owner))).is_err() {
+                // Broken unsafe bridge: retain rather than invoke native Drop.
+                // This fallback is NOT a claimed physical-fence proof.
+                if let Some(owner) = backup {
+                    core::mem::forget(owner);
+                }
+            }
+        }
+    }
+
+    struct Binding {
+        mailbox: Arc<RetirementMailbox>,
+        index: usize,
+        generation: u64,
+    }
+
+    /// Prepaid storage/action, cancelled if construction never starts.
+    #[must_use]
+    pub struct RetirementReservation {
+        binding: Option<Binding>,
+    }
+
+    impl RetirementReservation {
+        /// MUST arm BEFORE the original resource-creating call. Subsequent
+        /// failure publishes a checked no-resource/unknown action, never frees
+        /// the prepaid slot while a real resource may already exist.
+        pub fn arm(mut self) -> RetirementHandle {
+            RetirementHandle {
+                binding: self.binding.take(),
+            }
+        }
+    }
+
+    impl Drop for RetirementReservation {
+        fn drop(&mut self) {
+            let Some(binding) = self.binding.take() else {
+                return;
+            };
+            let slot = &binding.mailbox.slots[binding.index];
+            assert_eq!(slot.generation.load(Ordering::Acquire), binding.generation);
+            assert_eq!(slot.state.load(Ordering::Acquire), RESERVED);
+            // SAFETY: this unarmed reservation still owns the slot exclusively;
+            // no original resource may have been constructed through it yet.
+            let unused = unsafe { (&mut *slot.payload.get()).take().unwrap() };
+            drop(unused);
+            slot.state.store(FREE, Ordering::Release);
+            binding
+                .mailbox
+                .free
+                .fetch_or(1u64 << binding.index, Ordering::Release);
+        }
+    }
+
+    /// Last Drop only atomically publishes an already-owned slot/action.
+    /// No allocation, lock, driver, ordinary enqueue or queue-full wait.
+    #[must_use]
+    pub struct RetirementHandle {
+        binding: Option<Binding>,
+    }
+
+    impl Drop for RetirementHandle {
+        fn drop(&mut self) {
+            let Some(binding) = self.binding.take() else {
+                return;
+            };
+            let slot = &binding.mailbox.slots[binding.index];
+            assert_eq!(slot.generation.load(Ordering::Acquire), binding.generation);
+            assert_eq!(slot.state.load(Ordering::Acquire), RESERVED);
+            slot.state.store(PUBLISHED, Ordering::Release);
+            binding
+                .mailbox
+                .published
+                .fetch_or(1u64 << binding.index, Ordering::Release);
+        }
+    }
+
+    #[cfg(test)]
+    mod tests {
+        include!("retirement/tests.rs");
+    }
+}
diff --git a/cubecl-cuda/src/compute/capture.rs b/cubecl-cuda/src/compute/capture.rs
--- a/cubecl-cuda/src/compute/capture.rs
+++ b/cubecl-cuda/src/compute/capture.rs
@@ -14,14 +14,16 @@
 use cudarc::driver::sys::{CUgraph, CUgraphExec, CUgraphNode, CUresult, CUstream};
 
 /// An instantiated CUDA executable graph, destroyed on drop.
-pub struct Executable(CUgraphExec);
+pub struct Executable(Option<CUgraphExec>);
 
 impl Drop for Executable {
     fn drop(&mut self) {
         // SAFETY: produced by `cuGraphInstantiateWithFlags` and destroyed
         // exactly once here; the handle is never copied out.
-        unsafe {
-            cudarc::driver::sys::cuGraphExecDestroy(self.0);
+        if let Some(exec) = self.0 {
+            unsafe {
+                cudarc::driver::sys::cuGraphExecDestroy(exec);
+            }
         }
     }
 }
@@ -53,7 +55,12 @@
     fn upload(exec: &Executable, stream: &mut Stream) {
         // SAFETY: `exec` was instantiated above and `sys` is this stream; the
         // upload is enqueued stream-ordered.
-        let uploaded = unsafe { cudarc::driver::sys::cuGraphUpload(exec.0, stream.sys) };
+        let uploaded = unsafe {
+            cudarc::driver::sys::cuGraphUpload(
+                exec.0.expect("live captured executable"),
+                stream.sys,
+            )
+        };
         if let Err(err) = checked("cuGraphUpload", uploaded) {
             log::warn!(
                 "Pre-uploading the captured graph failed; \
@@ -65,8 +72,23 @@
     fn replay(exec: &Executable, stream: &mut Stream) -> Result<(), ServerError> {
         // SAFETY: `exec` is a valid instantiated graph; launching it on the
         // stream re-runs the recorded sequence.
-        let status = unsafe { cudarc::driver::sys::cuGraphLaunch(exec.0, stream.sys) };
+        let status = unsafe {
+            cudarc::driver::sys::cuGraphLaunch(
+                exec.0.expect("live captured executable"),
+                stream.sys,
+            )
+        };
         checked("cuGraphLaunch", status)
+    }
+
+    fn destroy_checked(exec: &mut Executable) -> Result<(), ServerError> {
+        let Some(raw) = exec.0 else { return Ok(()) };
+        // SAFETY: original actor owns this exact live executable; the caller
+        // completed its trusted stream barrier. Failure retains the pointer.
+        let status = unsafe { cudarc::driver::sys::cuGraphExecDestroy(raw) };
+        checked("cuGraphExecDestroy", status)?;
+        exec.0 = None;
+        Ok(())
     }
 }
 
@@ -123,7 +145,7 @@
             cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
         );
         cudarc::driver::sys::cuGraphDestroy(graph);
-        instantiated.map(|_| Executable(exec))
+        instantiated.map(|_| Executable(Some(exec)))
     }
 }
 
diff --git a/cubecl-cuda/src/compute/server.rs b/cubecl-cuda/src/compute/server.rs
--- a/cubecl-cuda/src/compute/server.rs
+++ b/cubecl-cuda/src/compute/server.rs
@@ -338,6 +338,22 @@
             // stream is exactly the contamination the claim exists to avoid.
             failed_writing(self, stream_id, written, err);
         }
+    }
+
+    fn graph_destroy_checked(
+        &mut self,
+        graph: GraphId,
+        stream_id: StreamId,
+    ) -> Result<(), ServerError> {
+        if !self.graphs.contains(graph) {
+            return Ok(());
+        }
+        // ANY failed barrier is unknown here, including poisoned-device status.
+        // Keep graph/executable/page/source pins; caller retains original actor
+        // and fences through the SAME original physical QueueState.
+        cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id))?;
+        let stream = self.streams.try_stream_mut(&stream_id);
+        self.graphs.destroy_checked(graph, stream)
     }
 
     fn sync(
diff --git a/cubecl-runtime-0.11.0/src/client.rs b/cubecl-runtime-0.11.0/src/client.rs
--- a/cubecl-runtime-0.11.0/src/client.rs
+++ b/cubecl-runtime-0.11.0/src/client.rs
@@ -22,7 +22,10 @@
 mod lazy;
 use cubecl_common::{
     bytes::{AllocationProperty, Bytes},
-    device::{CommandParticipation, CommandScope, DeviceId, ServiceId},
+    device::{
+        CommandParticipation, CommandScope, DeviceId, RetirementFailure, RetirementHandle,
+        RetirementKind, RetirementTask, ServiceId,
+    },
     device_handle::{CallResultExt, DeviceHandle},
     profile::ProfileDuration,
 };
@@ -84,6 +87,49 @@
     id: GraphId,
     device: DeviceHandle<dyn Server>,
     stream_id: StreamId,
+    retirement: Option<RetirementHandle>,
+}
+
+/// The mailbox keeps this exact original actor alive through known cleanup or
+/// permanent unknown quarantine. No destructor on this task calls a driver.
+struct GraphRetirement {
+    device: DeviceHandle<dyn Server>,
+    stream_id: StreamId,
+    id: cubecl_environment::sync::Mutex<Option<GraphId>>,
+    failure: cubecl_environment::sync::Mutex<Option<ServerError>>,
+    call_failure: cubecl_environment::sync::Mutex<Option<cubecl_common::device_handle::CallError>>,
+}
+
+// SAFETY: original actor mailbox calls this after task/service borrows end.
+// Only checked terminal success clears id; unknown keeps task + Client + graph
+// map/pins intact. The gate must enter prepaid original actor reentry first.
+unsafe impl RetirementTask for GraphRetirement {
+    fn execute(&self) -> Result<(), RetirementFailure> {
+        let mut owned = self.id.lock();
+        // Armed before end_capture: no returned id does NOT prove that the
+        // original constructor made no driver-side resource or side effect.
+        let Some(id) = *owned else {
+            return Err(RetirementFailure::Unknown);
+        };
+        let stream_id = self.stream_id;
+        match self
+            .device
+            .submit_blocking(move |server| server.graph_destroy_checked(id, stream_id))
+        {
+            Ok(Ok(())) => {
+                *owned = None;
+                Ok(())
+            }
+            Ok(Err(error)) => {
+                *self.failure.lock() = Some(error);
+                Err(RetirementFailure::Unknown)
+            }
+            Err(error) => {
+                *self.call_failure.lock() = Some(error);
+                Err(RetirementFailure::Unknown)
+            }
+        }
+    }
 }
 
 impl Graph {
@@ -150,6 +196,11 @@
 
 impl Drop for GraphHandle {
     fn drop(&mut self) {
+        if let Some(prepaid) = self.retirement.take() {
+            // Only atomic publication; no ordinary enqueue, driver or wait.
+            drop(prepaid);
+            return;
+        }
         let id = self.id;
         let stream_id = self.stream_id;
         // Destroying the raw executable must happen on the server actor (the
@@ -1408,16 +1459,40 @@
     /// [`replay`](Graph::replay).
     pub fn stop_capture(&self) -> Result<Graph, ServerError> {
         let stream_id = self.stream_id();
+        let prepared = self
+            .device
+            .prepare_retirement(RetirementKind::Graph, || {
+                Arc::new(GraphRetirement {
+                    device: self.device.clone(),
+                    stream_id,
+                    id: cubecl_environment::sync::Mutex::new(None),
+                    failure: cubecl_environment::sync::Mutex::new(None),
+                    call_failure: cubecl_environment::sync::Mutex::new(None),
+                })
+            })
+            .map_err(|error| ServerError::Generic {
+                reason: format!("{error}"),
+                backtrace: BackTrace::capture(),
+            })?;
+        let (retirement, task) = match prepared {
+            Some((reserved, task)) => (Some(reserved.arm()), Some(task)),
+            None => (None, None),
+        };
         let id = self
             .device
             .submit_blocking(move |server| server.end_capture(stream_id))
             .unwrap_or_resume()?;
+        if let Some(task) = task {
+            // The slot was armed BEFORE end_capture; this is its actual result.
+            *task.id.lock() = Some(id);
+        }
 
         Ok(Graph {
             inner: Arc::new(GraphHandle {
                 id,
                 device: self.device.clone(),
                 stream_id,
+                retirement,
             }),
         })
     }
diff --git a/cubecl-runtime-0.11.0/src/server/base.rs b/cubecl-runtime-0.11.0/src/server/base.rs
--- a/cubecl-runtime-0.11.0/src/server/base.rs
+++ b/cubecl-runtime-0.11.0/src/server/base.rs
@@ -783,6 +783,18 @@
         let _ = (graph, stream_id);
     }
 
+    /// Strict prepaid retirement: retain executable/pins on ANY failed wait or
+    /// destroy. No default delegation to legacy graph_destroy (which cannot
+    /// report terminal outcome). Called on the original admitted actor only.
+    fn graph_destroy_checked(
+        &mut self,
+        graph: GraphId,
+        stream_id: StreamId,
+    ) -> Result<(), ServerError> {
+        let _ = (graph, stream_id);
+        Err(ServerError::graph_capture_unsupported())
+    }
+
     /// Structured per-pool report of the given stream's **main GPU** memory:
     /// each pool's shape, usage, and high-water marks, in allocation-routing
     /// order. The read side of a measured memory plan — see
diff --git a/cubecl-server-0.11.0/src/command/capture.rs b/cubecl-server-0.11.0/src/command/capture.rs
--- a/cubecl-server-0.11.0/src/command/capture.rs
+++ b/cubecl-server-0.11.0/src/command/capture.rs
@@ -73,6 +73,14 @@
     ///
     /// The driver's refusal to enqueue the replay.
     fn replay(exec: &Self::Executable, stream: &mut Self::Stream) -> Result<(), ServerError>;
+
+    /// Checked explicit destruction. On failure keep the executable untouched
+    /// for original quarantine; on success its later Drop makes no driver call.
+    /// Default refuses, so legacy-only drivers cannot claim strict retirement.
+    fn destroy_checked(exec: &mut Self::Executable) -> Result<(), ServerError> {
+        let _ = exec;
+        Err(ServerError::graph_capture_unsupported())
+    }
 }
 
 /// An instantiated graph, and everything its window pinned for it.
@@ -197,6 +205,24 @@
         if let Some(stream) = stream {
             stream.info_cache().graph_release(id);
         }
+    }
+
+    /// Caller established trusted terminal ordering. Keep map, original page
+    /// guards, host-source bytes and info pins on ANY driver failure.
+    pub fn destroy_checked(
+        &mut self,
+        id: GraphId,
+        stream: Option<&mut D::Stream>,
+    ) -> Result<(), ServerError> {
+        let Some(graph) = self.graphs.get_mut(&id) else {
+            return Ok(());
+        };
+        D::destroy_checked(&mut graph.exec)?;
+        self.graphs.remove(&id);
+        if let Some(stream) = stream {
+            stream.info_cache().graph_release(id);
+        }
+        Ok(())
     }
 }
 

````

