# Draft Patches — volume 23

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/frozen/RETIREMENT.patch

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

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/NEW_FILES.review.patch

Original UTF-8 bytes: 73289; SHA256 `f17bb18a8bb5ab5e5a57499aa1089d889fa1d1df6b21d9ef824f6d161ae3e05b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ container-v1/README.md
@@ -0,0 +1,148 @@
+# Native container activity diagnostic — source only
+
+Author: batch_design_review. Parent plan: serving-and-speculation.
+
+Own scope: this ignored child only. Immutable parents: CUPTI V1 f9f5c522, V2
+932e471e and genuine matched-benchmark V8 b0c50e8a. No MAIN/parent edits, engine,
+math, queue, source/model, timing workload or production profile changes.
+
+## Files / ROOT execution
+
+- `capacity-v1/`: separate bounded C++ revision + streamed validator + six
+  synthetic fixtures. Capacity freeze4337ad63; new actual ROOT build required.
+- `container_contracts.py`, `parent-bindings.json`: exact additive closure.
+- `launch_derivative.py`: exact V8 argv + diagnostic-only mounts/env/bootstrap
+  transport and finite file-size cap. Original model flags untouched.
+- `native_exec.py`: run unchanged V8 bootstrap; its one final native exec uses
+  `os.execve` with explicit collector env. Outer Python/Podman LD_PRELOAD remains
+  empty. Stock always runs original bootstrap unchanged.
+- `namespace_control.py`: original owner nonce/name/CID -> public inspect init
+  HPID -> pidfd/start time/namespace readbacks -> host SO_PEERCRED. No signal,
+  reap, setns, new privilege, foreign PID adoption, context or stream.
+- `run_container.py`: four fresh original V8/53ee-owned processes: native plain,
+  native observed, stock plain, stock public Torch-profiler observed.
+- `container_activity.py`, `capacity_v1_loader.py`, `validate_container.py`:
+  independent full raw activity/wire/namespace/owner/canonical-argv validation.
+- `test_contracts.py`: seven prepared synthetic transport/control negatives.
+  No synthetic fixture establishes physical collection or model mathematics.
+
+ROOT first reviews/freezes, then executes only after timing quiet + reservation:
+
+```text
+python3 capacity-v1/test_activity.py
+python3 test_contracts.py
+```
+
+Build `capacity-v1/collector_capacity.cpp` with immutable parent ROOT_BUILD.md
+procedure and original approved headers/compiler. Record actual dependency/ELF
+closure, argv/source/build/artifact hashes. Fresh artifact directory contains
+`collector.so` and actual `build.json` with:
+
+```json
+{
+  "status": "ROOT_COMPILED_COLLECTOR_ONLY",
+  "freeze_sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd",
+  "collector_source_sha256": "ACTUAL_CAPACITY_CPP_SHA256",
+  "collector_sha256": "ACTUAL_COMPILED_ARTIFACT_SHA256",
+  "exit_code": 0
+}
+```
+
+Do not prewrite success. The accepted e918ff2f host V1 artifact is intentionally
+refused for this revision. Build can preserve original callback/API/math bytes;
+only fixed limits128→512MiB, records1M→4M, schema and explicit capacity metadata
+change. No runtime environment override can raise limits.
+
+Use an existing genuine V8 launch directory (exact protocol/workload/launches/
+input-attestation), not a shortened host-control prompt. Set ROOT explicitly:
+
+```text
+ZG_BENCH_HOST_ROOT=/home/pvl/spaces/abird/src/zerogravity
+python3 run_container.py --run --own-freeze OWN_FREEZE_SHA256
+  --launch-directory ABSOLUTE_ORIGINAL_V8_LAUNCH_DIRECTORY
+  --collector-artifact-directory ABSOLUTE_FRESH_ROOT_CAPACITY_BUILD_DIRECTORY
+  --concurrency ORIGINAL_PROTOCOL_CONCURRENCY
+  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
+python3 validate_container.py --own-freeze OWN_FREEZE_SHA256
+  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
+```
+
+Invocation is direct argv under ROOT's existing original process supervisor,
+not shell interpolation; export the explicit host-root variable. Author has
+not imported, run fixtures, compiled or executed any provider/container/workload.
+Only Python AST parsing and small source/metadata hashing/freezing occurred.
+
+## Identity, window, policy
+
+- Actual container init HPID comes from nonce/name/CID-bound public inspect;
+  Podman Popen PID remains its original cleanup owner, not the collector PID.
+- Exact host peer HPID + ordinary host UID must match SO_PEERCRED. pidfd remains
+  live; public container binding, proc start time and namespaces match before/
+  after each START/STOP. Output retains actual UID/GID maps and namespace IDs.
+- Reported `getpid()` is namespacePID1. Raw CUPTI driver IDs are never rewritten:
+  exactly one ID, equal to authenticated HPID OR namespacePID1; reject mixed or
+  foreign IDs and preserve which representation actually occurred. Metadata and
+  driver identities use separate validator arguments.
+- Original `--network=host` supplies abstract UNIX reachability; actual net-ns
+  equality and user-namespace peer translation must pass. Any hidepid/proc,
+  pidfd, mapped-file, permission or provider denial refuses; no privilege repair.
+- Actual native/collector/CUPTI mapped inode/dev/full SHA are checked through
+  `/proc/HPID/root`; CUPTI C++ still resolves only official versioned symbols
+  from exact original provider. Remaining original native/provider dependencies
+  retain existing V8/ROOT build closure; no substitute DSO.
+- Both native modes inherit exact V8 bootstrap/source/affinity/cgroup checks.
+  Stock uses original profiler-config and public `/start_profile`/`stop_profile`;
+  no stock math/module/queue patches or CUPTI preload.
+- `cold_process`: no collector warmup request. `warmed_process`: ONE unchanged
+  V8 unmeasured request with first original prompt and original max_tokens.
+  OS/storage/device caches are not evicted or claimed cold. Official collector
+  initialization overhead itself is diagnostic, not a canonical timing sample.
+- Native START follows original terminal/idle health; all canonical requests
+  then complete; native terminal/idle health precedes STOP/forced flush.
+  All source native dispatch counters remain distinct from CUPTI records.
+- Same canonical request count/prompts/arrivals/concurrency/max_tokens, CPU4,
+  RAM12GiB/swap-total12GiB, device, model/provider limits, stock eager/default
+  route and generation policy. No one-token collector warmup or workload edit.
+- Four actual owner receipts/monitor/containers/wholegroups/readers/EOF must
+  retire under unchanged reviewed53ee WNOWAIT cleanup before next process.
+  Primary/stop/close/post-source failures remain fatal with original receipts.
+
+## Capacity and remaining physical gates
+
+512MiB raw /4,000,000 records are finite, not a guarantee every canonical wave
+fits. Any buffer refusal/drop/overflow/partial record/API/timestamp mismatch
+refuses all acceptance. Entire native file is streamed twice, exact duplicate/
+correlation/totals checked. Validation sets are finite but may use several GiB
+of separate host RAM: ROOT reserves headroom before validation; not engine
+provider admission. Kernel count, successful direct launch APIs, graph launch
+APIs and kernels executed from graphs stay separate.
+
+Diagnostic-only per-file soft=hard RLIMIT_FSIZE is576MiB. Each owned sample tree
+must be<=768MiB (capacity README's native-sample bound); four-sample aggregate is
+<=1GiB to hold512MiB native raw + original stock trace<=256MiB + metadata/logs.
+Tree limits are post-write acceptance checks, not a kernel filesystem quota.
+File limit is enforced by original Podman public ulimit; native wrapper records
+actual readback. Original CPU/RAM/model limits are unchanged. Stock trace retains
+original256MiB compressed/expanded bounds; exceeding any bound is refusal.
+
+Before claiming a method: ROOT must qualify actual new build/13fixtures,
+namespace/user-peer/mapped closure, exact plain/observed wire IDs/text/finish/
+usage, real workload window and no activity loss, actual caps, final process/
+container/group/pipe/monitor closure. Negative tests must include foreign peer,
+wrong nonce/namespace/stale PID, provider permissions/cold refusal, failed STOP,
+interruption/receipt I/O, actual overflow/drop behavior and no accepted partial
+trace. Synthetic corrupted metadata fixtures validate refusal only; they do not
+establish actual callback overflow/CUPTI drop behavior.
+
+Root source estimate236270 dispatches for C1/48prompt/output8 suggests old128MiB
+cap too small. This is source metadata, NOT a canonical hardware measurement or
+speed result. Never run old undersized artifact and accept a prefix of records.
+
+Actual small host V2 PASS64328 kernels remains scoped to that old method. No
+container, larger-capacity, matched counts or new performance claim follows.
+Stock public Torch profiler currently does NOT prove losslessness/all-worker/
+complete initialization coverage; validator deliberately returns
+`matched_complete_hardware_kernel_counts_qualified=false` even if both recorded
+workload scopes pass. These diagnostic data do not establish numerical profile
+parity, source-correct goodput or a timing speedup. Opaque CUPTI internal device
+allocation admission remains unqualified and observed resource cost is separate.
--- /dev/null
+++ container-v1/STATIC_AUDIT.json
@@ -0,0 +1,62 @@
+{
+  "CXX_changes_normalize_byte_exact_to_immutable_V1": true,
+  "actual_new_collector_build_or_runtime_qualified": false,
+  "container_summarizer_changes_only_explicit_namespace_and_driver_arguments": true,
+  "critical_V8_source_pins": [
+    {
+      "bytes": 11893,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/prepare_launch_v8.py",
+      "sha256": "abaa0dfe32bb4aff3c1df0062fc54e7ebb0e73be1676ae290f5e5adb2725dc2d"
+    },
+    {
+      "bytes": 4706,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/server_boot_v8.py",
+      "sha256": "2288dfbe131b88b6c4fa6860d9eb3b070f4223f262fb2352a1f7f7cd0ce76baf"
+    },
+    {
+      "bytes": 9968,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/run_compare_v8.py",
+      "sha256": "fa08b592357217d58dd438014f6986976cb591e36fad8fd601c3fb83bf388adb"
+    },
+    {
+      "bytes": 8621,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/run_diagnostic_v8.py",
+      "sha256": "7c002f4803aca6daee0b772756395388cd5704c49fc5e3602c3309207bd27c7f"
+    },
+    {
+      "bytes": 9685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_plan_v8.py",
+      "sha256": "346f8f98753018e6ddacaff291dec48def8ce024be6e9c4670c154fc028bedcb"
+    },
+    {
+      "bytes": 11840,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_container_v4.py",
+      "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+    },
+    {
+      "bytes": 10387,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/protocol_v8.py",
+      "sha256": "61de8d581e9353c685bec518eab5f974df10de8baa57353e867606887b9ff98e"
+    },
+    {
+      "bytes": 5708,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/observations_v8.py",
+      "sha256": "23a783abbead3da819f9d0b046643010b669a6f14f2abb1e0752bd4ae35e4a9d"
+    },
+    {
+      "bytes": 13235,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/cgroup_readback_v3.py",
+      "sha256": "e29e76e06eb8477ffc279f1e3bf4277007c3f539e20053eddd76a3fda1df9a7b"
+    },
+    {
+      "bytes": 3387,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/audit_cuda_trace_v8.py",
+      "sha256": "15449b7fec9045d88beecf4cfb2025af39aa36083bc156015c75a5f1354ce747"
+    }
+  ],
+  "fixtures_prepared_not_executed": 13,
+  "original_CUPTI_V1_V2_and_capacity_frozen_bytes_unchanged": true,
+  "parent_manifests_exact": true,
+  "python_AST_files": 11,
+  "scope": "SMALL_SOURCE_ONLY_NO_IMPORTS_TESTS_BUILDS_RUNTIME"
+}
--- /dev/null
+++ container-v1/TODO.md
@@ -0,0 +1,25 @@
+# CUPTI container diagnostic / capacity revision
+
+Parent: serving-and-speculation. Owner: batch_design_review.
+Scope: ONLY this ignored child; no production/parent/plan/README changes.
+
+- [x] Read immutable CUPTI V1/V2 + exact V8 canonical launch/bootstrap/owner/
+  diagnostic/trace/cgroup source; keep original53ee lifecycle unchanged.
+- [x] Prepare separate512MiB/4M C++ cap revision and six source-only negatives;
+  exact callback/API/math unchanged, new compiled artifact mandatory.
+- [x] Prepare final-native-exec-only seam; host PID/namespace PID/raw driver PID
+  distinct; original owner/caps/source/full canonical workload/cold-warm policy.
+- [x] Prepare four-process plain/observed native+stock transparency and independent
+  complete activity/argv/owner validator; seven container transport fixtures.
+- [x] AST parse only; no imports, fixtures, compilation or runtime by author.
+- [ ] Independent source review / ROOT actual fixture+strict C++build+dependency
+  closure / actual process namespace+permissions/metadata/STOP/loss qualification.
+- [ ] ROOT actual canonical workload/negative failure+overflow/retirement tests.
+- [ ] Stock collector loss/completeness gate; no matched complete HW claim yet.
+- [ ] Any actual measured gain/numerical equivalence/fullmodel claim is external;
+  none established by this source proposal.
+
+Failures: no executed test/build/runtime. Source-only inspection corrected a
+post-START receipt-I/O cleanup gap before freeze; STOP flag now arms immediately
+on acknowledged START. Post-run source/affinity errors downgrade report before
+independent validation rather than leaving an executable-success status.
--- /dev/null
+++ container-v1/capacity_v1_loader.py
@@ -0,0 +1,10 @@
+"""Load own frozen capacity helper without changing its bytes or sys.path."""
+import importlib.util
+from pathlib import Path
+import sys
+
+path=Path(__file__).resolve().parent/'capacity-v1/activity.py'
+spec=importlib.util.spec_from_file_location('zg_cupti_capacity_v1_activity',path)
+activity=importlib.util.module_from_spec(spec)
+sys.modules[spec.name]=activity
+spec.loader.exec_module(activity)
--- /dev/null
+++ container-v1/container_activity.py
@@ -0,0 +1,72 @@
+"""Capacity validator with separate authenticated namespace/driver process IDs.
+
+Raw rows and metadata are never rewritten. Apart from these explicit identity
+arguments, same source algorithm as the frozen capacity validator.
+"""
+import collections
+from capacity_v1_loader import activity
+MAX_RECORDS=activity.MAX_RECORDS
+KERNEL_CBIDS=activity.KERNEL_CBIDS
+GRAPH_CBIDS=activity.GRAPH_CBIDS
+require=activity.require
+validate_capacity=activity.validate_capacity
+
+def summarize(rows, metadata, namespace_pid, driver_pid):
+    validate_capacity(metadata)
+    require(metadata['phase'] == 'STOPPED' and metadata['api_version'] == 28
+            and metadata['pid'] == namespace_pid and 0 < metadata['begin'] < metadata['end'],
+            'real complete native CUPTI window')
+    require(metadata['failures'] == metadata['dropped'] == metadata['first_cupti_error'] == 0,
+            'no parse/buffer/timestamp/activity loss or CUPTI error')
+    require(metadata['timing_sample'] is False
+            and metadata['actual_device_allocation_bound_qualified'] is False,
+            'diagnostic-only explicit allocation/timing scope')
+    require(metadata['host_callback_pool_bytes'] == 16 * 1024**2
+            and metadata['device_buffer_requested_bytes'] == 1024**2
+            and metadata['device_pool_limit'] == 8 and metadata['device_preallocate'] == 2,
+            'actual requested/read-back buffer controls')
+    kernel_count = api_count = graph_kernels = direct_calls = graph_calls = 0
+    kernel_correlations, launch_correlations, devices, unique = set(), set(), set(), set()
+    names = collections.Counter()
+    for row in rows:
+        require(metadata['begin'] <= row['start'] <= row['end'] <= metadata['end']
+                and row['start'] > 0, 'complete activity within declared workload window')
+        require(type(row['correlation']) is int and row['correlation'] > 0,
+                'actual nonempty driver/kernel correlation')
+        if row['kind'] == 'kernel':
+            require(row['name'] and len(row['name'].encode()) < 512
+                    and all(type(n) is int and n > 0 for n in row['grid'] + row['block'])
+                    and len(row['grid']) == len(row['block']) == 3, 'actual complete kernel geometry/name')
+            key = ('kernel', row['context'], row['stream'], row['grid_id'], row['start'], row['end'])
+            kernel_count += 1
+            graph_kernels += row['graph'] != 0
+            kernel_correlations.add(row['correlation'])
+            devices.add(row['device'])
+            names[row['name']] += 1
+            require(len(names) <= 2048, 'bounded actual kernel-name inventory')
+        else:
+            require(row['kind'] == 'driver' and row['process'] == driver_pid, 'actual original-process driver API')
+            key = ('driver', row['process'], row['thread'], row['correlation'], row['cbid'], row['start'], row['end'])
+            api_count += 1
+            if row['cbid'] in KERNEL_CBIDS | GRAPH_CBIDS:
+                require(row['result'] == 0, 'actual successful public launch API')
+                launch_correlations.add(row['correlation'])
+                direct_calls += row['cbid'] in KERNEL_CBIDS
+                graph_calls += row['cbid'] in GRAPH_CBIDS
+        require(key not in unique, 'no duplicated CUPTI activity record')
+        unique.add(key)
+        require(len(unique) <= MAX_RECORDS, 'bounded retained activity')
+    require(kernel_count == metadata['kernels'] > 0 and api_count == metadata['driver_apis'] > 0
+            and len(unique) == metadata['records'], 'all retained activity records/totals, not startup only')
+    require(launch_correlations and kernel_correlations <= launch_correlations,
+            'all actual native GPU kernels have retained successful public launch correlations')
+    require(len(devices) == 1, 'one declared physical GPU')
+    return {
+        'actual_gpu_kernel_records': kernel_count,
+        'driver_kernel_launch_api_calls': direct_calls,
+        'driver_graph_launch_api_calls': graph_calls,
+        'kernel_records_from_graphs': graph_kernels,
+        'kernel_names': dict(names),
+        'scope': 'official concurrent-kernel activity records in this native process/window; API calls separate',
+    }
+
--- /dev/null
+++ container-v1/container_contracts.py
@@ -0,0 +1,95 @@
+"""ROOT-only diagnostic closure. No process/GPU work on import."""
+import hashlib
+import json
+from pathlib import Path
+
+HERE=Path(__file__).resolve().parent
+FILE_LIMIT=576*1024**2
+TREE_LIMIT=1024*1024**2
+
+
+def require(ok,why):
+    if not ok: raise ValueError(why)
+
+
+def sha(path):
+    with Path(path).open('rb') as stream:
+        return hashlib.file_digest(stream,'sha256').hexdigest()
+
+
+def read(path):
+    with Path(path).open('rb') as stream: raw=stream.read(4*1024**2+1)
+    require(len(raw)<=4*1024**2,'bounded diagnostic metadata')
+    return json.loads(raw)
+
+
+def write(path,value):
+    with Path(path).open('x') as stream:
+        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
+        stream.write('\n')
+
+
+def check_file(path,item):
+    path=Path(path)
+    require(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes']
+            and sha(path)==item['sha256'],'exact prepared file: '+str(path))
+
+
+def frozen(expected):
+    require(sha(HERE/'freeze-manifest.json')==expected,'explicit own prepared source identity')
+    manifest=read(HERE/'freeze-manifest.json')
+    for item in manifest['files']: check_file(HERE/item['file'],item)
+    parent=read(HERE/'parent-bindings.json')
+    check_file(HERE/'capacity-v1/freeze-manifest.json',parent['capacity_manifest'])
+    capacity=read(HERE/'capacity-v1/freeze-manifest.json')
+    for item in capacity['files']: check_file(HERE/'capacity-v1'/item['file'],item)
+    return parent
+
+
+def host_parents(parent):
+    for key in ('cupti_v1_manifest','cupti_v2_manifest','benchmark_v8_manifest'):
+        item=parent[key]; check_file(item['path'],item)
+    for key in ('cupti_v1_manifest','cupti_v2_manifest'):
+        item=parent[key]; root=Path(item['path']).parent
+        manifest=read(item['path'])
+        for row in manifest['files']: check_file(root/row['file'],row)
+        for row in manifest.get('source_pins',[]): check_file(row['path'],row)
+    helper=Path(parent['parents']['benchmark'])/'owned_container_v4.py'
+    require(sha(helper)==parent['owned_helper_sha256'],'same reviewed reserved-leader helper')
+
+
+def build_binding(path,artifact,parent):
+    value=read(path)
+    require(value['status']=='ROOT_COMPILED_COLLECTOR_ONLY' and value['exit_code']==0
+            and value['freeze_sha256']==parent['capacity_manifest']['sha256']
+            and value['collector_source_sha256']==sha(HERE/'capacity-v1/collector_capacity.cpp')
+            and value['collector_sha256']==sha(artifact),
+            'actual ROOT new capacity source/build/artifact binding; old V1 binary cannot substitute')
+    return value
+
+
+def fingerprint(result):
+    require(result['completed_requests']==result['offered_requests']>0
+            and result['unsuccessful_requests']==0 and result['errors']=={},'complete real HTTP denominator')
+    out=[]
+    require(len(result['requests'])==result['offered_requests'],'all request records')
+    for item in result['requests']:
+        ids=item['wire_token_observation']['token_ids']
+        require(item['completed'] is True and item['usage_reported_completion_tokens']>0
+                and len(ids)==item['usage_reported_completion_tokens']
+                and all(type(v) is int and v>=0 for v in ids),'complete wire IDs and usage')
+        out.append((item['input_index'],item['prompt_sha256'],item['output_sha256'],
+                    item['finish_reason'],item['usage_reported_completion_tokens'],tuple(ids)))
+    return out
+
+
+def tree_bound(path,limit=TREE_LIMIT):
+    require(limit in (768*1024**2,TREE_LIMIT),'fixed reviewed per-owner/aggregate artifact limits')
+    total=count=0
+    for entry in Path(path).rglob('*'):
+        require(not entry.is_symlink(),'owned diagnostic tree without symlinks')
+        if entry.is_dir(): continue
+        require(entry.is_file(),'regular diagnostic artifacts only')
+        size=entry.stat().st_size; count+=1; total+=size
+        require(size<=FILE_LIMIT and total<=limit and count<=256,'bounded full diagnostic artifact tree')
+    return dict(bytes=total,files=count,limit=limit,per_file_limit=FILE_LIMIT)
--- /dev/null
+++ container-v1/launch_derivative.py
@@ -0,0 +1,27 @@
+"""Pure exact-V8 command derivative; only diagnostic transport/artifact controls."""
+from pathlib import Path
+
+from container_contracts import FILE_LIMIT, HERE, require
+
+
+def derive(command,provider,mode,own_freeze,artifact):
+    require(provider in ('zerogravity','vllm') and mode in ('plain','observed'), 'finite declared sample')
+    command=list(command)
+    old='fsize=268435456:268435456'
+    require(command.count(old)==1 and command.count('/probe/server_boot_v8.py')==1,
+            'exact known V8 diagnostic file control and bootstrap')
+    # Only diagnostic artifact budget changes; model/cgroup/device/workload unchanged.
+    command[command.index(old)]=f'fsize={FILE_LIMIT}:{FILE_LIMIT}'
+    require('--network=host' in command and 'LD_PRELOAD=' in command and 'LD_AUDIT=' in command,
+            'original host network and empty outer interposition')
+    if provider=='vllm':
+        return command
+    require(command[command.index('--provider')+1]=='zerogravity','native-only wrapper')
+    position=command.index('--entrypoint')
+    command[position:position]=['-v',str(HERE)+':/collector-probe:ro',
+        '-v',str(Path(artifact))+':/collector-artifact:ro',
+        '-e','ZG_CONTAINER_DIAGNOSTIC_FREEZE='+own_freeze,
+        '-e','ZG_CONTAINER_DIAGNOSTIC_MODE='+mode,
+        '-e','ZG_CONTAINER_DIAGNOSTIC_NONCE={owned_nonce}']
+    command[command.index('/probe/server_boot_v8.py')]='/collector-probe/native_exec.py'
+    return command
--- /dev/null
+++ container-v1/namespace_control.py
@@ -0,0 +1,143 @@
+"""Host controller: authenticate nonce/CID -> actual init HPID -> SO_PEERCRED.
+
+Does not signal, reap, setns, or adopt foreign processes. Existing V8/53ee owner
+alone retires its original Podman group/container. Namespace PID and driver PID
+remain separate observed identities; neither is rewritten into a host PID.
+"""
+import json
+import os
+from pathlib import Path
+import re
+import select
+import socket
+import struct
+
+from container_contracts import read, require, sha
+
+
+def text(path,limit):
+    with Path(path).open() as stream: raw=stream.read(limit+1)
+    require(len(raw)<=limit,'bounded original process metadata')
+    return raw
+
+
+def stat_start(raw):
+    # comm may include spaces and parentheses; the last ')' ends field2.
+    at=raw.rfind(')')
+    require(at>1 and len(raw[at+2:].split())>=20,'complete Linux proc stat')
+    return int(raw[at+2:].split()[19])
+
+
+def inspect(owner,owned):
+    owner.live()
+    cid=owned.owned_container(owner.name,owner.nonce,owner.cidfile)
+    require(cid is not None,'live exact nonce/name/CID binding')
+    result=owned.command(owned.PODMAN+['container','inspect',cid],timeout=10)
+    require(result.returncode==0 and len(result.stdout)<=1024**2,'bounded public container inspect')
+    value=json.loads(result.stdout)
+    require(type(value) is list and len(value)==1,'one inspected original container')
+    row=value[0]
+    require(row['Id']==cid and row['Name'].lstrip('/')==owner.name
+            and row['Config']['Labels'][owned.LABEL]==owner.nonce
+            and row['State']['Running'] is True and type(row['State']['Pid']) is int
+            and row['State']['Pid']>0,'same original live init host PID, never Podman Popen PID')
+    return dict(cid=cid,owned_nonce=owner.nonce,owned_name=owner.name,host_init_pid=row['State']['Pid'])
+
+
+def identities(pid):
+    root=Path('/proc')/str(pid)
+    raw=text(root/'status',65536)
+    fields={key:value.strip() for line in raw.splitlines() if ':' in line for key,value in [line.split(':',1)]}
+    nspid=[int(v) for v in fields['NSpid'].split()]
+    require(nspid[0]==pid and nspid[-1]==1,'actual direct native namespace PID1')
+    return dict(start_time=stat_start(text(root/'stat',16384)), NSpid=nspid,
+        namespace_inodes={name:os.readlink(root/'ns'/name) for name in ('pid','net','user','mnt')})
+
+
+def mapped_files(pid,exec_attestation,parent):
+    root=Path('/proc')/str(pid)
+    lines=text(root/'maps',2*1024**2).splitlines()
+    expected={'native':'/runtime/zerogravity','collector':'/collector-artifact/collector.so',
+              'cupti':parent['cupti_library']['container_path']}
+    output={}
+    for key,path in expected.items():
+        actual=root/'root'/path.lstrip('/')
+        info=actual.stat()
+        rows=[]
+        for line in lines:
+            parts=line.split(None,5)
+            if len(parts)==6 and parts[5]==path:
+                major,minor=(int(v,16) for v in parts[3].split(':'))
+                require(int(parts[4])==info.st_ino and os.makedev(major,minor)==info.st_dev,
+                        'actual mapped inode/dev matches exact original file')
+                rows.append(line)
+        require(rows and actual.is_file(),'all declared executable/collector/CUPTI mappings present')
+        digest=sha(actual)
+        expected_sha=exec_attestation['native_sha256' if key=='native' else key+'_sha256']
+        require(digest==expected_sha,'full actual mapped artifact hash: '+key)
+        output[key]=dict(container_path=path,bytes=info.st_size,sha256=digest,
+            inode=info.st_ino,device=info.st_dev,mappings=rows)
+    return output
+
+
+class Controller:
+    def __init__(self,owner,owned,parent):
+        self.owner,self.owned,self.parent=owner,owned,parent
+        self.fd=None
+        self.receipt=dict(schema='zg.cupti.container.controller.v1',events=[],pidfd_closed=False)
+        try:
+            self.binding=inspect(owner,owned)
+            self.pid=self.binding['host_init_pid']
+            self.fd=os.pidfd_open(self.pid,0)
+            self.before=identities(self.pid)
+            self.live()
+            actual=read(owner.directory/'native-exec-attestation.json')
+            require(actual['mode']=='observed' and actual['pid_namespace_pid']==1
+                    and actual['nonce']==owner.nonce and actual['collector_on_outer_python'] is False
+                    and actual['collector_requested_only_at_this_exec'] is True,
+                    'actual unchanged final-exec receipt bound to original owner nonce')
+            require(actual['namespace_inodes']==self.before['namespace_inodes']
+                    and self.before['namespace_inodes']['net']==os.readlink('/proc/self/ns/net'),
+                    'actual host-network namespace shared; no setns/privilege assumption')
+            self.nonce=actual['nonce']
+            self.receipt.update(binding=self.binding,host_before=self.before,exec_attestation=actual,
+                mapped_files=mapped_files(self.pid,actual,parent),
+                host_controller_uid=os.geteuid(),reported_namespace_pid=1,
+                driver_process_id_policy='preserve raw CUPTI IDs; require namespace PID1 independently')
+        except BaseException:
+            self.close()
+            raise
+
+    def live(self):
+        require(self.fd is not None and not select.select([self.fd],[],[],0)[0],
+                'original init pidfd remains live, no recycled PID accepted')
+        require(inspect(self.owner,self.owned)==self.binding and identities(self.pid)==self.before,
+                'original container init identity and namespace binding stable')
+
+    def command(self,verb):
+        require(verb in ('START','STOP') and len(self.receipt['events'])==(0 if verb=='START' else 1),
+                'one ordered START/STOP, no repeated activity windows')
+        self.live()
+        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as connection:
+            connection.settimeout(30)
+            connection.connect('\0zg-cupti-'+self.nonce)
+            peer=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
+            require(peer[0]==self.pid and peer[1]==os.geteuid(),
+                    'SO_PEERCRED actual host init PID/UID, independently of reported namespace PID')
+            connection.sendall((verb+' '+self.nonce).encode('ascii'))
+            raw,_,flags,_=connection.recvmsg(8192)
+            require(not flags&socket.MSG_TRUNC and len(raw)<8192,'complete bounded control reply')
+            value=json.loads(raw)
+        require(value['pid']==1 and value['nonce']==self.nonce and value['api_version']==28
+                and value['phase']==('STARTED' if verb=='START' else 'STOPPED'),
+                'actual namespace PID1/API/nonce/ordered phase')
+        require(value['failures']==value['dropped']==value['first_cupti_error']==0,'no lost/partial CUPTI result')
+        self.live()
+        self.receipt['events'].append(dict(verb=verb,host_peer_pid=peer[0],host_peer_uid=peer[1],
+            reported_namespace_pid=value['pid'],reply=value))
+        return value
+
+    def close(self):
+        if self.fd is not None:
+            os.close(self.fd); self.fd=None
+        self.receipt['pidfd_closed']=True
--- /dev/null
+++ container-v1/native_exec.py
@@ -0,0 +1,81 @@
+"""Final-native-exec-only diagnostic seam around exact immutable V8 bootstrap.
+
+Outer Python and Podman never receive LD_PRELOAD. No engine argv/model math is
+changed. Stock always uses the unchanged original server_boot_v8.py directly.
+"""
+import os
+import resource
+from pathlib import Path
+import sys
+
+from container_contracts import FILE_LIMIT, HERE, build_binding, check_file, frozen, read, require, sha, write
+
+
+def bounded_text(path,limit=8192):
+    with Path(path).open() as stream: value=stream.read(limit+1)
+    require(len(value)<=limit,'bounded namespace identity')
+    return value
+
+
+def main():
+    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+            'collector absent from outer Python/Podman process')
+    expected=os.environ['ZG_CONTAINER_DIAGNOSTIC_FREEZE']
+    parent=frozen(expected)
+    require(Path('/probe/prepared-manifest-v8.json').is_file(), 'original V8 mounted separately')
+    check_file('/probe/prepared-manifest-v8.json',parent['benchmark_v8_manifest'])
+    sys.path.insert(0,'/probe')
+    import server_boot_v8 as original
+    require(Path(original.__file__).resolve()==Path('/probe/server_boot_v8.py'), 'exact original bootstrap path')
+    require(os.getpid()==1,'direct original container-init process, no extra supervisor/engine')
+    require(resource.getrlimit(resource.RLIMIT_FSIZE)==(FILE_LIMIT,FILE_LIMIT),
+            'actual diagnostic-only finite file-size soft/hard cap')
+    mode=os.environ['ZG_CONTAINER_DIAGNOSTIC_MODE']
+    require(mode in ('plain','observed'),'declared diagnostic mode')
+    saved=os.execv
+    def final_exec(path,argv):
+        require(path=='/runtime/zerogravity' and argv[0]==path and argv[1]=='serve'
+                and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'single exact final native exec; no outer preload or stock interception')
+        attestation=dict(schema='zg.cupti.container.exec.v1', mode=mode, own_freeze_sha256=expected,
+            benchmark_v8_sha256=parent['benchmark_v8_manifest']['sha256'], argv=argv,
+            native_sha256=sha(path), pid_namespace_pid=os.getpid(), uid=os.geteuid(), gid=os.getegid(),
+            uid_map=bounded_text('/proc/self/uid_map'), gid_map=bounded_text('/proc/self/gid_map'),
+            namespace_inodes={name:os.readlink('/proc/self/ns/'+name) for name in ('pid','net','user','mnt')},
+            CPU_affinity=sorted(os.sched_getaffinity(0)), file_size_limits=list(resource.getrlimit(resource.RLIMIT_FSIZE)),
+            collector_on_outer_python=False,
+            exec_transport_only=True, math_or_queue_changes=False)
+        require(attestation['native_sha256']==os.environ['ZG_NATIVE_BINARY_SHA256'], 'same V8 actual native artifact')
+        env=dict(os.environ)
+        if mode=='observed':
+            collector='/collector-artifact/collector.so'
+            build=build_binding('/collector-artifact/build.json',collector,parent)
+            library=parent['cupti_library']
+            check_file(library['container_path'],library)
+            activity=Path('/output/observed-activity')
+            activity.mkdir(mode=0o700)
+            require(activity.stat().st_uid==os.geteuid() and (activity.stat().st_mode&0o777)==0o700,
+                    'original CUPTI directory owner/mode in actual user namespace')
+            nonce=os.environ['ZG_CONTAINER_DIAGNOSTIC_NONCE']
+            require(len(nonce)==32 and all(v in '0123456789abcdef' for v in nonce),'exact controller nonce')
+            env.update(LD_PRELOAD=collector, ZG_CUPTI_OUTPUT=str(activity), ZG_CUPTI_NONCE=nonce,
+                ZG_CUPTI_LIBRARY=library['container_path'], ZG_CUPTI_TARGET_EXE=path)
+            attestation.update(collector_sha256=sha(collector), collector_build_sha256=sha('/collector-artifact/build.json'),
+                collector_capacity_freeze_sha256=parent['capacity_manifest']['sha256'],
+                cupti_sha256=library['sha256'], nonce=nonce,
+                collector_requested_only_at_this_exec=True, ROOT_build_status=build['status'])
+        write('/output/native-exec-attestation.json',attestation)
+        os.execve(path,argv,env)
+        raise AssertionError('successful exec cannot return')
+    # Scoped control transfer only; original bootstrap retains all source, affinity,
+    # cgroup and native-binary checks before its ONE final exec site.
+    os.execv=final_exec
+    try:
+        original.main()
+        raise AssertionError('original native bootstrap must exec')
+    finally:
+        os.execv=saved
+
+
+if __name__=='__main__':
+    main()
--- /dev/null
+++ container-v1/parent-bindings.json
@@ -0,0 +1,32 @@
+{
+  "benchmark_v8_manifest": {
+    "bytes": 27185,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/prepared-manifest-v8.json",
+    "sha256": "b0c50e8a9d8d647464d3837748db0ba958a7c8294ee18140474703068fca0b33"
+  },
+  "capacity_manifest": {
+    "bytes": 1195,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/freeze-manifest.json",
+    "sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd"
+  },
+  "cupti_library": {
+    "bytes": 7604800,
+    "container_path": "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib/libcupti.so.12",
+    "sha256": "2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a"
+  },
+  "cupti_v1_manifest": {
+    "bytes": 12997,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/freeze-manifest.json",
+    "sha256": "f9f5c52234ff95386daab29ac1f4435834541e08815891dc7a8bfea29a1406c5"
+  },
+  "cupti_v2_manifest": {
+    "bytes": 2799,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/freeze-manifest-v2.json",
+    "sha256": "932e471e37ce21fc6588bb5bd67423945aae5ba8ac6b14aa5185df7b194653d3"
+  },
+  "owned_helper_sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d",
+  "parents": {
+    "benchmark": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020",
+    "cupti": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020"
+  }
+}
--- /dev/null
+++ container-v1/run_container.py
@@ -0,0 +1,230 @@
+"""ROOT ONLY four fresh diagnostic processes; no canonical timing samples.
+
+Native plain/CUPTI observed, stock plain/public Torch profiler observed. Each
+uses exact V8 requests/cold-warm policy/CPU/RAM/engine limits and original owner.
+"""
+import argparse
+from dataclasses import asdict
+import gzip
+import hashlib
+import json
+import os
+from pathlib import Path
+import signal
+import sys
+import time
+import urllib.request
+
+from container_contracts import (HERE,TREE_LIMIT,build_binding,fingerprint,frozen,host_parents,
+    read,require,sha,tree_bound,write)
+from launch_derivative import derive
+from namespace_control import Controller
+
+
+def idle(bench,url):
+    deadline=time.monotonic()+30
+    previous=None
+    while time.monotonic()<deadline:
+        raw=bench.snapshot(url+'/health',None)
+        require(raw['ok'],'actual original native health')
+        state=raw['value']['admission']
+        require(state['kind']=='resident_gpu' and state['failed_gpu_quanta']==0,'healthy original GPU engine')
+        if state['active_sessions']==state['queued_requests']==state['output_credit_parked_requests']==0:
+            if previous==raw['value']: return raw
+            previous=raw['value']
+        else: previous=None
+        time.sleep(.1)
+    raise TimeoutError('native physical work did not become terminal before window boundary')
+
+
+def stock_traces(owner,audit):
+    output=[]
+    directory=owner.directory/'traces'
+    require(directory.is_dir() and not directory.is_symlink(),'actual original public profiler output')
+    paths=sorted(directory.rglob('*'))
+    require(len(paths)<=32,'bounded actual profiler artifact count')
+    for path in paths:
+        require(not path.is_symlink(),'owned original profiler files')
+        if path.is_dir(): continue
+        require(path.is_file() and path.stat().st_size<=256*1024**2,'bounded public profiler artifact')
+        if not path.name.endswith(('.pt.trace.json','.pt.trace.json.gz')): continue
+        with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
+            raw=stream.read(256*1024**2+1)
+        require(len(raw)<=256*1024**2,'bounded entire decoded public trace')
+        value=json.loads(raw)
+        if any(event.get('cat')=='kernel' for event in value.get('traceEvents',[])):
+            result=audit(value)
+            result.update(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
+                decoded_sha256=hashlib.sha256(raw).hexdigest())
+            output.append(result)
+    require(len(output)==1,'one actual TP1 CUDA worker trace, no CPU-only substitution')
+    return output
+
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run',action='store_true')
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--launch-directory',type=Path,required=True)
+    parser.add_argument('--collector-artifact-directory',type=Path,required=True)
+    parser.add_argument('--concurrency',type=int,choices=(1,2,4,8),required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    require(args.run,'ROOT reviewed explicit actual execution only')
+    parent=frozen(args.own_freeze); host_parents(parent)
+    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
+    benchmark=Path(parent['parents']['benchmark'])
+    sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_compare_v8 import lifecycle_spec,load_tool,opener
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from support_v3 import source_check
+    from owned_plan_v8 import install
+    import owned_container_v4 as owned
+    v8=parent['benchmark_v8_manifest']['sha256']
+    pins=verify_preparation(v8); verify_small_sources()
+    require(sha(owned.__file__)==parent['owned_helper_sha256'],'actual original53ee lifecycle helper import')
+    launch_dir=args.launch_directory.resolve(strict=True)
+    require(launch_dir.parent==benchmark and not args.launch_directory.is_symlink(),'original V8 owned launch directory')
+    protocol=validate_protocol(read(launch_dir/'protocol.json'))
+    require(args.concurrency in protocol['concurrencies'] and protocol['source_sha256']==pins['models'][protocol['model']]['gguf']['sha256']
+            and protocol['tokenizer_sha256']==pins['tokenizer_sha256'],'exact V8 workload/concurrency/source identity')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    conversion=Path(project['conversion_directory'])
+    commands={label:launch(protocol,pins,v8,label,conversion) for label in ('zerogravity','vllm')}
+    require(read(launch_dir/'launches.json')==commands,'exact original canonical engine commands')
+    diagnostics=read(launch_dir/'diagnostic-launches.json')
+    stock_observed=launch(protocol,pins,v8,'vllm',conversion,profiler=True)
+    require(diagnostics['cuda_activity']==stock_observed,'exact existing public stock profiler command')
+    source=pins['models'][protocol['model']]
+    source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+    require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+            and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256'],
+            'actual unchanged projector/native executable before diagnostic')
+    artifact=args.collector_artifact_directory.resolve(strict=True)
+    require(not args.collector_artifact_directory.is_symlink(),'original ROOT artifact directory')
+    build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+    output=args.output.resolve()
+    require(output.parent==HERE and not output.exists(),'fresh exclusive owned diagnostic output')
+    output.mkdir(mode=0o700)
+    compare,bench=load_tool('serve-compare.py'),load_tool('serve-bench.py')
+    install(compare.LIFECYCLE)
+    urls={label:f'http://127.0.0.1:{port}' for label,port in zip(('zerogravity','vllm'),protocol['ports'])}
+    spec=lifecycle_spec(protocol,commands)
+    plan=compare.LIFECYCLE['LifecyclePlan'](spec,urls,str(output/'owned'))
+    alias='qwen35-'+protocol['model']+'-same-gguf'
+    report=dict(status='UNQUALIFIED',schema='zg.cupti.container.diagnostic.v1',own_freeze_sha256=args.own_freeze,
+        benchmark_v8_sha256=v8,capacity_freeze_sha256=parent['capacity_manifest']['sha256'],
+        protocol=protocol,protocol_path=str(launch_dir/'protocol.json'),
+        protocol_sha256=sha(launch_dir/'protocol.json'),concurrency=args.concurrency,
+        collector_build_sha256=sha(artifact/'build.json'),collector_sha256=build['collector_sha256'],
+        owned_cleanup_source_sha256=parent['owned_helper_sha256'],samples=[],
+        performance_timing_usable=False,matched_hardware_counts_qualified=False,
+        source_correct_goodput_qualified=False,numerical_profiles_identical=False,
+        native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+    allowed=os.sched_getaffinity(0)
+    require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
+    urllib.request.install_opener(opener())
+    try:
+        os.sched_setaffinity(0,protocol['cpu_affinity'])
+        for label,mode in (('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed')):
+            base=stock_observed if label=='vllm' and mode=='observed' else commands[label]
+            spec_command=derive(base,label,mode,args.own_freeze,artifact)
+            plan.spec['providers'][label]['argv']=spec_command
+            sample=dict(provider=label,mode=mode,policy=protocol['policy'],warmup=None)
+            with plan.sample(label,lambda:compare.preflight(urls[label],alias,None)) as owner:
+                if protocol['policy']=='warmed_process':
+                    warm=bench.stream_one(urls[label],alias,protocol['requests'][0]['prompt'],
+                        protocol['max_tokens'],180,None,request_id='matched-diag-warmup',return_token_ids=True)
+                    owner.record_warmup(asdict(warm)); sample['warmup']=asdict(warm)
+                    require(warm.completed,'exact canonical unmeasured warmed_process request')
+                before=idle(bench,urls[label]) if label=='zerogravity' else None
+                if before is not None: write(owner.directory/'health-before.json',before)
+                controller=None; started=False; primary=None; cleanup=[]
+                try:
+                    if mode=='observed':
+                        if label=='zerogravity':
+                            controller=Controller(owner,owned,parent)
+                            start=controller.command('START')
+                            started=True
+                            write(owner.directory/'window-start.json',start)
+                        else:
+                            sample['profile_start']=post(urls[label],'/start_profile')
+                            started=True
+                    measured=bench.bench(urls[label],alias,protocol['requests'][0]['prompt'],len(protocol['requests']),
+                        args.concurrency,protocol['max_tokens'],180,None,workload=protocol['requests'],
+                        return_token_ids=True,max_pending=protocol['max_pending'],
+                        launch_deadline_seconds=protocol['launch_deadline_seconds'],run_id='matched-cuda-activity')
+                    fingerprint(measured); owner.record_measurement(measured)
+                    sample['observation']=measured
+                    after=idle(bench,urls[label]) if label=='zerogravity' else None
+                    if after is not None: write(owner.directory/'health-after.json',after)
+                except BaseException as error:
+                    primary=error
+                finally:
+                    if started:
+                        try:
+                            if controller is not None:
+                                stop=controller.command('STOP')
+                                write(owner.directory/'window-stop.json',stop)
+                            else: sample['profile_stop']=post(urls[label],'/stop_profile')
+                        except BaseException as error: cleanup.append('collector stop: '+repr(error))
+                    if controller is not None:
+                        try: controller.close()
+                        except BaseException as error: cleanup.append('controller close: '+repr(error))
+                        try: write(owner.directory/'namespace-controller.json',controller.receipt)
+                        except BaseException as error: cleanup.append('controller receipt: '+repr(error))
+                    require(not cleanup and primary is None,
+                            'diagnostic primary='+repr(primary)+'; collector_cleanup='+repr(cleanup))
+            # All original owner/process/group/container/monitor/pipe cleanup finished.
+            require(not plan.blocked,'complete original retirement before next provider')
+            sample['owned_receipt_path']=str(owner.receipt_path)
+            if mode=='observed' and label=='vllm': sample['actual_recorded_CUDA_activity']=stock_traces(owner,audit)
+            report['samples'].append(sample)
+            tree_bound(owner.directory,768*1024**2)
+            tree_bound(output)
+        for label in ('zerogravity','vllm'):
+            pair=[item for item in report['samples'] if item['provider']==label]
+            require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),
+                    'complete plain/observed wire IDs/text/finish/usage exact for '+label)
+        report['output_transparency_exact']=True
+        report['status']='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        report['failure']=repr(error)[:4096]
+        raise
+    finally:
+        # All process/container owners already retired; independent evidence stages
+        # must still run if affinity restoration or a post-run source scan fails.
+        failures=[]
+        try: os.sched_setaffinity(0,allowed)
+        except BaseException as error: failures.append('restore client affinity: '+repr(error))
+        report['lifecycle_manifest']=str(plan.directory/'lifecycle.json')
+        try:
+            report['lifecycle_manifest_sha256']=sha(plan.directory/'lifecycle.json')
+        except BaseException as error: failures.append('lifecycle digest: '+repr(error))
+        try:
+            frozen(args.own_freeze); host_parents(parent); verify_preparation(v8); verify_small_sources()
+        except BaseException as error: failures.append('post-run immutable source closure: '+repr(error))
+        try:
+            source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+            require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+                    and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256']
+                    and sha(artifact/'collector.so')==build['collector_sha256']
+                    and sha(artifact/'build.json')==report['collector_build_sha256'],
+                    'unchanged actual model/projector/binary/collector/build after retirement')
+        except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
+        report['finalization_errors']=failures
+        if failures: report['status']='UNQUALIFIED'
+        write(output/'report.json',report)
+        if failures:
+            raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
+
+
+
+if __name__=='__main__':
+    def interrupted(_signum,_frame): raise KeyboardInterrupt('ROOT container diagnostic interrupted')
+    saved=signal.signal(signal.SIGTERM,interrupted)
+    try: main()
+    finally: signal.signal(signal.SIGTERM,saved)
--- /dev/null
+++ container-v1/test_contracts.py
@@ -0,0 +1,92 @@
+"""ROOT-only mocked control/transport negatives; no process/CUDA/model proof."""
+import copy
+import json
+from pathlib import Path
+import struct
+import unittest
+from unittest.mock import patch
+
+from capacity_v1_loader import activity
+from container_activity import summarize
+from container_contracts import FILE_LIMIT
+from launch_derivative import derive
+from namespace_control import Controller,stat_start
+from validate_container import driver_identity
+
+
+def metadata():
+    return dict(schema='zg.cupti.window.capacity.v1',max_bytes=activity.MAX_BYTES,max_records=activity.MAX_RECORDS,
+        bytes=1,records=2,failures=0,dropped=0,first_cupti_error=0,phase='STOPPED',api_version=28,
+        pid=1,begin=10,end=100,timing_sample=False,actual_device_allocation_bound_qualified=False,
+        host_callback_pool_bytes=16*1024**2,device_buffer_requested_bytes=1024**2,
+        device_pool_limit=8,device_preallocate=2,kernels=1,driver_apis=1)
+
+
+def records(pid):
+    return [dict(kind='kernel',start=30,end=40,correlation=17,name='synthetic-only',grid=[1,1,1],
+        block=[32,1,1],context=1,stream=3,grid_id=5,graph=0,device=0),
+        dict(kind='driver',start=20,end=25,correlation=17,process=pid,thread=2,cbid=307,result=0)]
+
+
+def command():
+    return ['podman','run','--network=host','--ulimit','fsize=268435456:268435456',
+        '-e','LD_PRELOAD=','-e','LD_AUDIT=','--memory=12g','--cpus=4','--entrypoint','python3','IMAGE',
+        '/probe/server_boot_v8.py','--provider','zerogravity','--model','0.8b','--prepared-sha256','PARENT',
+        '--','serve','--batch-rows','8']
+
+
+class Contracts(unittest.TestCase):
+    def test_native_derivative_never_preloads_outer(self):
+        source=command();actual=derive(source,'zerogravity','observed','OWN',Path('/artifact'))
+        self.assertEqual(source,command())
+        self.assertIn('LD_PRELOAD=',actual)
+        self.assertNotIn('LD_PRELOAD=/collector-artifact/collector.so',actual)
+        self.assertIn('/collector-probe/native_exec.py',actual)
+        self.assertEqual(actual[actual.index('--')+1:],source[source.index('--')+1:])
+        self.assertIn(f'fsize={FILE_LIMIT}:{FILE_LIMIT}',actual)
+        self.assertIn('--memory=12g',actual);self.assertIn('--cpus=4',actual)
+
+    def test_stock_exact_bootstrap_unchanged_except_artifact_fsize(self):
+        source=command();source[source.index('zerogravity')]='vllm'
+        actual=derive(source,'vllm','observed','OWN',Path('/artifact'))
+        changed=list(source);changed[changed.index('fsize=268435456:268435456')]=f'fsize={FILE_LIMIT}:{FILE_LIMIT}'
+        self.assertEqual(actual,changed)
+
+    def test_unknown_original_boot_or_cap_refuses(self):
+        for old,new in (('/probe/server_boot_v8.py','other.py'),('fsize=268435456:268435456','fsize=1:1')):
+            source=command();source[source.index(old)]=new
+            with self.assertRaises(ValueError): derive(source,'zerogravity','plain','OWN',Path('/artifact'))
+
+    def test_namespace_and_raw_driver_ids_remain_separate(self):
+        for pid in (1,32100):
+            actual=records(pid);saved=copy.deepcopy(actual)
+            self.assertEqual(summarize(actual,metadata(),1,pid)['actual_gpu_kernel_records'],1)
+            self.assertEqual(actual,saved)
+            with self.assertRaises(ValueError): summarize(actual,metadata(),9,pid)
+            with self.assertRaises(ValueError): summarize(actual,metadata(),1,pid+1)
+
+    def test_driver_process_namespace_does_not_admit_foreign_or_mixed(self):
+        self.assertEqual(driver_identity(records(1),32100),1)
+        self.assertEqual(driver_identity(records(32100),32100),32100)
+        for raw in (records(7),records(1)+records(32100),[]):
+            with self.assertRaises(ValueError): driver_identity(raw,32100)
+
+    def test_proc_start_parser_handles_comm_spaces_parentheses(self):
+        fields=['S']+['0']*18+['123456']+['0']*8
+        self.assertEqual(stat_start('81 (name with (nested) parens) '+' '.join(fields)),123456)
+
+    def test_control_rejects_wrong_host_peer_even_with_namespace_reply(self):
+        owner=object.__new__(Controller)
+        owner.pid=32100;owner.nonce='a'*32;owner.live=lambda:None;owner.receipt={'events':[]}
+        class Socket:
+            def __enter__(self): return self
+            def __exit__(self,*args): pass
+            def settimeout(self,*args): pass
+            def connect(self,*args): pass
+            def getsockopt(self,*args): return struct.pack('3i',1,1000,1000)
+        with patch('namespace_control.socket.socket',return_value=Socket()),patch('namespace_control.os.geteuid',return_value=1000):
+            with self.assertRaises(ValueError): owner.command('START')
+        self.assertEqual(owner.receipt['events'],[])
+
+
+if __name__=='__main__': unittest.main()
--- /dev/null
+++ container-v1/validate_container.py
@@ -0,0 +1,215 @@
+"""ROOT-only independent complete-workload/namespace/activity/owner validator.
+
+Stock public Torch profiler loss completeness remains explicitly UNQUALIFIED;
+recorded kernel counts cannot establish source-correct goodput or timing speed.
+"""
+import argparse
+import hashlib
+from pathlib import Path
+import sys
+
+from capacity_v1_loader import activity
+from container_activity import summarize
+from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
+from launch_derivative import derive
+
+
+def successful_owner(receipt,parent):
+    require(receipt['status']=='OWNERS_CLOSED' and receipt['cleanup_errors']==[]
+            and receipt['primary_failure'] is None and receipt['ready'] is True
+            and receipt['owned_container_absent_after_reap'] is True,'actual complete successful owned retirement')
+    cleanup=receipt['foreground_cleanup']
+    require(cleanup['pgid']==receipt['owned_cli_pid']>0 and cleanup['leader_reaped'] is True
+            and cleanup['group_absent'] is True and cleanup['all_nonzero_signals_before_reap'] is True
+            and cleanup['kill_attempts_before_reap']==2 and cleanup['leader_reap_attempts'] in (1,2)
+            and cleanup['stage_errors']==[],'all independent group signals before reap and final absence')
+    log=receipt['log_cleanup']
+    require(log['reader_complete'] is True and log['reader_error'] is None,'actual complete original log/pipe EOF')
+    observations=receipt['physical_observations']
+    require(observations['failures']==[] and observations['samples']
+            and observations['exact_peak'] is False
+            and observations['complete_physical_allocator_accounting'] is False,
+            'actual bounded original physical observations, no invented allocator peak')
+    for row in observations['samples']:
+        if row['compute_apps']:
+            require(all(app['pid'] in row['authenticated_container_host_pids'] for app in row['compute_apps']),
+                    'every actual GPU PID bound to original public container numeric HPIDs')
+    require(receipt['GPU_quiet_after'] and all(not row['compute_apps'] for row in receipt['GPU_quiet_after'][-3:]),
+            'actual final card quiet after original owner cleanup')
+
+
+def workload(result,protocol,concurrency):
+    require(result['offered_requests']==len(protocol['requests']) and result['concurrency']==concurrency
+            and result['max_pending']==protocol['max_pending']
+            and result['launch_deadline_seconds']==protocol['launch_deadline_seconds']
+            and result['run_id']=='matched-cuda-activity','exact canonical workload denominator and scheduling policy')
+    fingerprint(result)
+    require(len(result['requests'])==len(protocol['requests']),'all original canonical offers')
+    for index,(actual,expected) in enumerate(zip(result['requests'],protocol['requests'])):
+        require(actual['input_index']==index and actual['request_id']==expected['request_id']
+                and actual['prompt_sha256']==hashlib.sha256(expected['prompt'].encode()).hexdigest()
+                and actual['max_tokens']==expected['max_tokens'], 'actual full original prompt/max-token/request identity')
+
+
+def driver_identity(rows,host_pid):
+    ids=set()
+    for row in rows:
+        if row['kind']=='driver':
+            require(type(row['process']) is int,'actual raw driver process ID')
+            ids.add(row['process'])
+            require(len(ids)==1,'single native process driver identity; never merge namespaces')
+    require(len(ids)==1 and ids<={1,host_pid},'raw driver ID equals one authenticated process representation')
+    return next(iter(ids))
+
+
+def native_activity(directory,receipt,parent,report):
+    control=read(directory/'namespace-controller.json')
+    actual=read(directory/'native-exec-attestation.json')
+    require(control['pidfd_closed'] is True and control['reported_namespace_pid']==1
+            and control['exec_attestation']==actual and actual['nonce']==receipt['owned_nonce']
+            and control['binding']['owned_nonce']==receipt['owned_nonce']
+            and control['binding']['owned_name']==receipt['owned_name'], 'original owner/exec/controller binding')
+    hp=control['binding']['host_init_pid']
+    require(hp>0 and actual['pid_namespace_pid']==1 and actual['namespace_inodes']==control['host_before']['namespace_inodes']
+            and control['host_before']['NSpid'][0]==hp and control['host_before']['NSpid'][-1]==1,
+            'host init PID and actual namespace PID preserved independently')
+    require(actual['file_size_limits']==[FILE_LIMIT,FILE_LIMIT], 'actual finite diagnostic file-size readback')
+    for key,field in (('native','native_sha256'),('collector','collector_sha256'),('cupti','cupti_sha256')):
+        require(control['mapped_files'][key]['sha256']==actual[field] and control['mapped_files'][key]['mappings'],
+                'actual complete mapped artifact identity retained')
+    require(actual['collector_on_outer_python'] is False and actual['collector_requested_only_at_this_exec'] is True
+            and actual['native_sha256']==report['protocol']['zerogravity_binary_sha256']
+            and actual['collector_sha256']==report['collector_sha256']
+            and actual['collector_capacity_freeze_sha256']==report['capacity_freeze_sha256']
+            and actual['cupti_sha256']==parent['cupti_library']['sha256'],'only final-exec exact artifact/provider closure')
+    require(len(control['events'])==2,'one complete original native activity window')
+    start,stop=read(directory/'window-start.json'),read(directory/'window-stop.json')
+    for event,verb,value in zip(control['events'],('START','STOP'),(start,stop)):
+        require(event['verb']==verb and event['reply']==value and event['host_peer_pid']==hp
+                and event['host_peer_uid']==control['host_controller_uid'] and event['reported_namespace_pid']==1,
+                'actual host SO_PEERCRED authenticated independently from CUPTI namespace identity')
+    require(start['nonce']==stop['nonce']==receipt['owned_nonce']
+            and start['begin']==stop['begin'] and start['end']==0
+            and start['start_monotonic']==stop['start_monotonic']<stop['stop_monotonic'], 'one actual timestamped window')
+    path=directory/'observed-activity/activity.jsonl'
+    require(read(directory/'observed-activity/collector-stop.json')==stop,'persisted stop exactly acknowledged')
+    digest=sha(path)
+    driver_pid=driver_identity(activity.rows(path,stop['bytes']),hp)
+    driver_ids={driver_pid}
+    result=summarize(activity.rows(path,stop['bytes']),stop,1,driver_pid)
+    require(sha(path)==digest,'immutable entire raw trace across independent complete passes')
+    before=read(directory/'health-before.json')['value']['admission']
+    after=read(directory/'health-after.json')['value']['admission']
+    for value in (before,after):
+        require(value['active_sessions']==value['queued_requests']==value['output_credit_parked_requests']==0
+                and value['failed_gpu_quanta']==0,'actual terminal original GPU work at window boundaries')
+    require(after['accepted_output_tokens']>before['accepted_output_tokens'],'actual canonical output within window')
+    result.update(raw_activity_sha256=digest,raw_bytes=path.stat().st_size,namespace_pid=1,
+        authenticated_init_host_pid=hp,actual_raw_driver_process_ids=sorted(driver_ids),
+        driver_process_representation='namespace' if driver_pid==1 else 'host',
+        source_dispatch_delta=after['completed_forward_kernel_dispatches']-before['completed_forward_kernel_dispatches'],
+        source_dispatches_are_not_driver_measurements=True,all_retained_records_lossless=True)
+    return result
+
+
+def main():
+    parser=argparse.ArgumentParser()
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    parent=frozen(args.own_freeze); host_parents(parent)
+    output=args.output.resolve(strict=True)
+    require(output.parent==HERE,'owned diagnostic output')
+    report=read(output/'report.json')
+    require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+            and report['own_freeze_sha256']==args.own_freeze
+            and report['benchmark_v8_sha256']==parent['benchmark_v8_manifest']['sha256']
+            and report['capacity_freeze_sha256']==parent['capacity_manifest']['sha256']
+            and report['output_transparency_exact'] is True and report['finalization_errors']==[]
+            and report['performance_timing_usable'] is False
+            and report['native_or_stock_math_changes'] is False,'executed bounded diagnostic-only scope')
+    benchmark=Path(parent['parents']['benchmark']); sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from cgroup_readback_v3 import validate_caps
+    from run_container import stock_traces
+    pins=verify_preparation(report['benchmark_v8_sha256']); verify_small_sources()
+    protocol=validate_protocol(report['protocol'])
+    protocol_path=Path(report['protocol_path'])
+    require(protocol_path.parent.parent==benchmark and protocol_path.name=='protocol.json'
+            and sha(protocol_path)==report['protocol_sha256'] and read(protocol_path)==protocol,
+            'actual immutable original V8 launch protocol file and complete value')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    require(sha(report['lifecycle_manifest'])==report['lifecycle_manifest_sha256'],'complete original lifecycle manifest')
+    lifecycle=read(report['lifecycle_manifest'])
+    require(lifecycle['owned_cleanup_source_sha256']==parent['owned_helper_sha256']
+            and lifecycle['adapter_source_sha256']==sha(benchmark/'owned_plan_v8.py')
+            and len(lifecycle['receipts'])==len(report['samples'])==4,'all four original owner receipts')
+    order=(('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed'))
+    native_summary=stock_summary=None
+    for sample,receipt_path,expected in zip(report['samples'],lifecycle['receipts'],order):
+        label,mode=expected
+        require((sample['provider'],sample['mode'])==expected and sample['owned_receipt_path']==receipt_path,
+                'all four actual distinct ordered samples')
+        receipt=read(receipt_path); directory=Path(receipt_path).parent
+        require(output in directory.parents,'original owned sample stays in this exclusive output')
+        tree_bound(directory,768*1024**2)
+        successful_owner(receipt,parent)
+        bootstrap=read(directory/'bootstrap.json')
+        validate_caps(bootstrap['cgroup_readback'],protocol['cpu_affinity'])
+        require(bootstrap['provider']==label and bootstrap['preparation_sha256']==report['benchmark_v8_sha256']
+                and bootstrap['stock_math_modified'] is False
+                and bootstrap['actual_cpu_affinity']==sorted(protocol['cpu_affinity']), 'original runtime caps/source bootstrap')
+        base=launch(protocol,pins,report['benchmark_v8_sha256'],label,Path(project['conversion_directory']),
+            profiler=label=='vllm' and mode=='observed')
+        # Receipt's mounted artifact path is authenticated by actual exec SHA/build
+        # below; all other launch bytes must be exact derivation of immutable V8.
+        artifact=Path(next(receipt['argv'][i+1].removesuffix(':/collector-artifact:ro')
+            for i,v in enumerate(receipt['argv'][:-1]) if v=='-v' and receipt['argv'][i+1].endswith(':/collector-artifact:ro'))) if label=='zerogravity' else HERE
+        if label=='zerogravity':
+            build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+            require(sha(artifact/'build.json')==report['collector_build_sha256']
+                    and build['collector_sha256']==report['collector_sha256'], 'independent actual ROOT build binding')
+        command=derive(base,label,mode,args.own_freeze,artifact)
+        replacements={'{owned_name}':receipt['owned_name'],'{owned_nonce}':receipt['owned_nonce'],
+            '{cidfile}':str(directory/'cid'),'{sample_directory}':str(directory)}
+        for key,value in replacements.items(): command=[part.replace(key,value) for part in command]
+        require(receipt['argv']==command,'every actual launch argument equals reviewed exact canonical derivative')
+        workload(sample['observation'],protocol,report['concurrency'])
+        require(read(directory/'measurement.json')==sample['observation'],'actual original measured HTTP observations')
+        if protocol['policy']=='cold_process':
+            require(sample['warmup'] is None and receipt['warmup'] is None,'no collector-only warmup in cold_process')
+        else:
+            require(sample['warmup']==receipt['warmup'] and sample['warmup']['completed'] is True
+                    and sample['warmup']['tokens']>0,'actual exact canonical warmed_process request')
+        if label=='zerogravity':
+            actual=read(directory/'native-exec-attestation.json')
+            require(actual['mode']==mode and actual['argv']==['/runtime/zerogravity']+bootstrap['engine_argv']
+                    and actual['own_freeze_sha256']==args.own_freeze,'same original final exec and own closure')
+            if mode=='observed': native_summary=native_activity(directory,receipt,parent,report)
+        elif mode=='observed':
+            for key,route in (('profile_start','/start_profile'),('profile_stop','/stop_profile')):
+                require(sample[key]['route']==route and sample[key]['HTTP_status']==200,'actual original public profiler controls')
+            # Reparse complete original trace after owner is gone, not stored report count.
+            class Owner: pass
+            owner=Owner(); owner.directory=directory
+            stock_summary=stock_traces(owner,audit)
+            require(stock_summary==sample['actual_recorded_CUDA_activity'],'independent entire actual stock trace replay')
+    for label in ('zerogravity','vllm'):
+        pair=[item for item in report['samples'] if item['provider']==label]
+        require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),'exact all wire token/text/finish/usage transparency')
+    require(native_summary is not None and stock_summary,'both actual canonical workload activity scopes present')
+    result=dict(status='PASS_CONTAINER_RECORDED_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY',
+        own_freeze_sha256=args.own_freeze,report_sha256=sha(output/'report.json'),
+        native=native_summary,stock_recorded=stock_summary,artifact_bounds=tree_bound(output),
+        same_canonical_workload_policy_and_caps_checked=True,plain_observed_wire_outputs_exact=True,
+        source_profiles_numerically_equal=False,source_correct_goodput_qualified=False,
+        matched_complete_hardware_kernel_counts_qualified=False,stock_collector_losslessness_verified=False,
+        timing_usable=False,device_profiler_allocation_admission_qualified=False)
+    frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
+    write(output/'validation.json',result)
+
+
+if __name__=='__main__': main()

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/NEW_DELTA.patch

Original UTF-8 bytes: 98204; SHA256 `f95fc3d2752c4b488000f847e0b50f6cede9cf8bd94e5237cf17bce908932856`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/README.md
@@ -0,0 +1,48 @@
+# CUPTI same-open-FD mapping identity — additive V3
+
+- Parent: serving-and-speculation; source-only owned proposal. Original container07a86bf6, capacity4337ad63, finalization2b82ca5f and all failed ROOT attempts stay immutable.
+- Motivation only: ROOT pre-START metadata and separate readonly-FD diagnostic show the native/collector's stat device and kernel VMA device differ while inode/full hashes agree. CUPTI's observed devices agree. Those observations do not independently qualify the target mappings or any GPU activity.
+- Eight new source/metadata files; no MAIN, collector, queue, model, CLI, workload, policy, capacity or owner changes. Author performed source/AST/byte checks only: no tests, proposed-module imports, mmap, DSO/model reads/hashes, provider/build/runtime/GPU execution.
+
+## Exact identity rule
+
+1. Authenticate the same original nonce/name/CID, actual init HPID, retained pidfd, start time and four namespaces. Preserve original host-network and SO_PEERCRED/namespace-PID separation.
+2. Open each exact `/proc/HPID/root/<declared-native-or-collector-or-CUPTI-path>` readonly/CLOEXEC/NOFOLLOW. Keep this FD through full bounded64KiB-chunk SHA256 and fstat checks. Require the original pathname still denotes that FD. No dlopen or executable mmap.
+3. Create only one temporary `MAP_PRIVATE|PROT_READ`, offset0, up to one system page, from that exact retained FD. Compare bounded complete local proc-maps snapshots before/during/after. Require one newly introduced nonoverlapping `r--p` page with the FD's exact kernel readlink/inode. Derive its device/major/minor/inode from the actual VMA, then prove that mapping disappears after explicit close. Coalescing/ambiguity/unsupported geometry refuses; no guessed address or filesystem exception.
+4. Require every actual target mapping at the original declared container path to match that derived kernel key and a bounded original file extent. At least one executable segment must exist. All target mapping rows must remain unchanged across the proof. Original stat-device identity is retained separately as `device`/`stat_device`; `mapping_device` is the independently observed kernel value. Never replace or ignore either value.
+5. Rehash the entire same FD and recheck all stat/path/readlink identities before closing it. Native additionally authenticates the original `/proc/HPID/exe` magic link, same file identity and two complete hashes through its own retained readonly FD. Preserve actual before/after observations, not inferred metadata.
+6. Check the original live process binding before/after every artifact. The original owner alone performs process/container retirement; this controller still never signals, reaps, setns, adopts or changes privileges.
+
+Independent V3 validator reconstructs the temporary VMA/key from all retained bounded local snapshots and rechecks every target mapping, same-FD hash/stat/path/exe/close receipt. These are authenticated executed-controller observations; source preparation alone proves no physical mapping.
+
+Bounds: original576MiB artifact-file cap; hashes64KiB scratch; at most one readonly page at a time; each complete local maps snapshot128KiB/2048rows; target maps retain original2MiB bound. Three snapshots per artifact remain in the bounded controller receipt. It must fit the original4MiB metadata reader before writing. Original1GiB aggregate,768MiB owner,256-file and two reserved final receipts remain exact; no artifact capacity increase.
+
+## ROOT review/execution only
+
+1. Read eight sources, `REVIEW.diff`, source pins and unchanged originals. Review full FD identity/cleanup and all-original native activity/stock profiler controls.
+2. From this directory, ROOT may run `python3 -m unittest -v test_fd_mapping_v3.py`. Prepared synthetic negatives include unequal/equal devices, mixed target key/extent, writable/shared/executable/nonzero-offset pages, preexisting/coalesced/ambiguous VMAs, remaining mappings, hash/stat/path/exe mutations, malformed bounds, and original foreign-peer refusal. Two fixtures exercise only a small synthetic ordinary file's readonly metadata mapping/refusal; never an ELF/provider/model.
+3. Existing original test_contracts.py/test_completion_v2.py remain unchanged. No collector rebuild is required for this host-only derivative; exact existing capacity collector/build and CUPTI provider bindings stay mandatory.
+4. After review/fixtures and an otherwise quiet owned GPU interval, use a fresh output directly under the original container-v1 directory:
+
+```sh
+python3 fd-mapping-v3/run_container_v3.py --run \
+  --own-freeze 07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee \
+  --mapping-freeze EXACT_NEW_FREEZE_SHA256 \
+  --launch-directory ORIGINAL_V8_LAUNCH_DIRECTORY \
+  --collector-artifact-directory ROOT_CAPACITY_BUILD_DIRECTORY \
+  --concurrency 8 --output NEW_OWNED_OUTPUT
+python3 fd-mapping-v3/validate_container_v3.py \
+  --own-freeze 07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee \
+  --mapping-freeze EXACT_NEW_FREEZE_SHA256 --output NEW_OWNED_OUTPUT
+```
+
+New mandatory `--mapping-freeze` binds these exact host sources; original `--own-freeze`, final-exec transport and all provider argv remain unchanged. V3 provenance is checked before/after execution and final writes, in both report and controller receipt. All original nonce/pidfd/start/ns/peer/path/hash, driver-ID, full HTTP transparency, native losslessness, stock trace, cgroup/caps, finite owner cleanup and final artifact reservations remain enforced. Full native/stock outputs and original loss/cleanup guards remain required after actual runtime.
+
+Acceptance stays diagnostic-only: no timing, source-correct goodput, identical numerical profile, complete matched hardware counts, stock profiler losslessness or device-profiler allocation admission claim. The new mapping source proposal does not close any of those gates.
+
+## Owned continuation
+
+- [x] Read actual ROOT metadata and original controller/runner/validator/contracts/finalization.
+- [x] Prepare additive V3 and source-free fixtures; static source/AST review only.
+- [ ] Independent peer + ROOT full source review.
+- [ ] ROOT actual fixtures, then fresh plain/observed four-owner execution and independent validation.
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/mapping_contract_v3.py
@@ -0,0 +1,79 @@
+"""Additive source closure only; imports never start a process or load a DSO."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import sys
+
+HERE = Path(__file__).resolve().parent
+PARENT = HERE.parent
+if str(PARENT) not in sys.path:
+    sys.path.insert(0, str(PARENT))
+
+from container_contracts import require
+from completion_v2 import verify_preparation as verify_finalization
+
+ORIGINAL = '07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee'
+SOURCE_CAP = 4 * 1024**2
+
+
+def pinned(path, row):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= SOURCE_CAP,
+                'bounded source/metadata file before read')
+        raw = stream.read(SOURCE_CAP + 1)
+        after = os.fstat(stream.fileno())
+    identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
+    require(identity(before) == identity(after) and len(raw) == before.st_size == row['bytes'] and
+            hashlib.sha256(raw).hexdigest() == row['sha256'], 'exact unchanged source/metadata pin')
+    return raw
+
+
+def unique(pairs):
+    result = {}
+    for name, value in pairs:
+        require(name not in result, 'unique source manifest fields')
+        result[name] = value
+    return result
+
+
+def bounded_controller_metadata(value):
+    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
+    require(len(raw) <= SOURCE_CAP, 'complete controller receipt fits original four-MiB metadata reader')
+    return value
+
+
+def verify_preparation(expected):
+    manifest_path = HERE/'freeze-manifest.json'
+    with os.fdopen(os.open(manifest_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= SOURCE_CAP,
+                'bounded regular additive source manifest before read')
+        raw = stream.read(SOURCE_CAP + 1)
+        after = os.fstat(stream.fileno())
+    identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
+    require(identity(before) == identity(after) and len(raw) == before.st_size and
+            hashlib.sha256(raw).hexdigest() == expected,
+            'explicit reviewed additive FD mapping source freeze')
+    value = json.loads(raw, object_pairs_hook=unique)
+    require(value['original_freeze_sha256'] == ORIGINAL and
+            value['math_queue_kernel_counts_or_GPU_runtime_qualified'] is False,
+            'original immutable source and diagnostic-only additive scope')
+    files = value['files']
+    require(type(files) is list and len(files) == 8 and len({row['file'] for row in files}) == 8,
+            'complete eight-file additive source closure')
+    for row in files:
+        require(row['file'] == Path(row['file']).name, 'finite local source filenames')
+        pinned(HERE/row['file'], row)
+        pinned(HERE/'frozen'/row['file'], row)
+    require(value['review_artifact']['file'] == 'REVIEW.diff', 'exact additive review artifact')
+    pinned(HERE/'REVIEW.diff', value['review_artifact'])
+    pins_row = next(row for row in files if row['file'] == 'source-pins.json')
+    pins = json.loads(pinned(HERE/'source-pins.json', pins_row), object_pairs_hook=unique)
+    for row in pins['unchanged_source_and_diagnostic_metadata']:
+        pinned(row['path'], row)
+    require(verify_finalization() == pins['original_finalization_freeze_sha256'],
+            'same reviewed final completion reservation and original parents')
+    return expected
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/namespace_control_v3.py
@@ -0,0 +1,122 @@
+"""Host controller: authenticate nonce/CID -> actual init HPID -> SO_PEERCRED.
+
+Does not signal, reap, setns, or adopt foreign processes. Existing V8/53ee owner
+alone retires its original Podman group/container. Namespace PID and driver PID
+remain separate observed identities; neither is rewritten into a host PID.
+"""
+import json
+import os
+from pathlib import Path
+import re
+import select
+import socket
+import struct
+
+from mapping_contract_v3 import verify_preparation
+from container_contracts import read, require, sha
+from verified_fd_mapping_v3 import mapped_files
+
+
+def text(path,limit):
+    with Path(path).open() as stream: raw=stream.read(limit+1)
+    require(len(raw)<=limit,'bounded original process metadata')
+    return raw
+
+
+def stat_start(raw):
+    # comm may include spaces and parentheses; the last ')' ends field2.
+    at=raw.rfind(')')
+    require(at>1 and len(raw[at+2:].split())>=20,'complete Linux proc stat')
+    return int(raw[at+2:].split()[19])
+
+
+def inspect(owner,owned):
+    owner.live()
+    cid=owned.owned_container(owner.name,owner.nonce,owner.cidfile)
+    require(cid is not None,'live exact nonce/name/CID binding')
+    result=owned.command(owned.PODMAN+['container','inspect',cid],timeout=10)
+    require(result.returncode==0 and len(result.stdout)<=1024**2,'bounded public container inspect')
+    value=json.loads(result.stdout)
+    require(type(value) is list and len(value)==1,'one inspected original container')
+    row=value[0]
+    require(row['Id']==cid and row['Name'].lstrip('/')==owner.name
+            and row['Config']['Labels'][owned.LABEL]==owner.nonce
+            and row['State']['Running'] is True and type(row['State']['Pid']) is int
+            and row['State']['Pid']>0,'same original live init host PID, never Podman Popen PID')
+    return dict(cid=cid,owned_nonce=owner.nonce,owned_name=owner.name,host_init_pid=row['State']['Pid'])
+
+
+def identities(pid):
+    root=Path('/proc')/str(pid)
+    raw=text(root/'status',65536)
+    fields={key:value.strip() for line in raw.splitlines() if ':' in line for key,value in [line.split(':',1)]}
+    nspid=[int(v) for v in fields['NSpid'].split()]
+    require(nspid[0]==pid and nspid[-1]==1,'actual direct native namespace PID1')
+    return dict(start_time=stat_start(text(root/'stat',16384)), NSpid=nspid,
+        namespace_inodes={name:os.readlink(root/'ns'/name) for name in ('pid','net','user','mnt')})
+
+
+
+
+class Controller:
+    def __init__(self,owner,owned,parent,mapping_freeze):
+        self.owner,self.owned,self.parent=owner,owned,parent
+        self.fd=None
+        self.receipt=dict(schema='zg.cupti.container.controller.v3',events=[],pidfd_closed=False,
+            fd_mapping_followup_sha256=verify_preparation(mapping_freeze))
+        try:
+            self.binding=inspect(owner,owned)
+            self.pid=self.binding['host_init_pid']
+            self.fd=os.pidfd_open(self.pid,0)
+            self.before=identities(self.pid)
+            self.live()
+            actual=read(owner.directory/'native-exec-attestation.json')
+            require(actual['mode']=='observed' and actual['pid_namespace_pid']==1
+                    and actual['nonce']==owner.nonce and actual['collector_on_outer_python'] is False
+                    and actual['collector_requested_only_at_this_exec'] is True,
+                    'actual unchanged final-exec receipt bound to original owner nonce')
+            require(actual['namespace_inodes']==self.before['namespace_inodes']
+                    and self.before['namespace_inodes']['net']==os.readlink('/proc/self/ns/net'),
+                    'actual host-network namespace shared; no setns/privilege assumption')
+            self.nonce=actual['nonce']
+            self.receipt.update(binding=self.binding,host_before=self.before,exec_attestation=actual,
+                mapped_files=mapped_files(self.pid,actual,parent,self.live),
+                host_controller_uid=os.geteuid(),reported_namespace_pid=1,
+                driver_process_id_policy='preserve raw CUPTI IDs; require namespace PID1 independently')
+        except BaseException:
+            self.close()
+            raise
+
+    def live(self):
+        require(self.fd is not None and not select.select([self.fd],[],[],0)[0],
+                'original init pidfd remains live, no recycled PID accepted')
+        require(inspect(self.owner,self.owned)==self.binding and identities(self.pid)==self.before,
+                'original container init identity and namespace binding stable')
+
+    def command(self,verb):
+        require(verb in ('START','STOP') and len(self.receipt['events'])==(0 if verb=='START' else 1),
+                'one ordered START/STOP, no repeated activity windows')
+        self.live()
+        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as connection:
+            connection.settimeout(30)
+            connection.connect('\0zg-cupti-'+self.nonce)
+            peer=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
+            require(peer[0]==self.pid and peer[1]==os.geteuid(),
+                    'SO_PEERCRED actual host init PID/UID, independently of reported namespace PID')
+            connection.sendall((verb+' '+self.nonce).encode('ascii'))
+            raw,_,flags,_=connection.recvmsg(8192)
+            require(not flags&socket.MSG_TRUNC and len(raw)<8192,'complete bounded control reply')
+            value=json.loads(raw)
+        require(value['pid']==1 and value['nonce']==self.nonce and value['api_version']==28
+                and value['phase']==('STARTED' if verb=='START' else 'STOPPED'),
+                'actual namespace PID1/API/nonce/ordered phase')
+        require(value['failures']==value['dropped']==value['first_cupti_error']==0,'no lost/partial CUPTI result')
+        self.live()
+        self.receipt['events'].append(dict(verb=verb,host_peer_pid=peer[0],host_peer_uid=peer[1],
+            reported_namespace_pid=value['pid'],reply=value))
+        return value
+
+    def close(self):
+        if self.fd is not None:
+            os.close(self.fd); self.fd=None
+        self.receipt['pidfd_closed']=True
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/run_container_v3.py
@@ -0,0 +1,240 @@
+"""ROOT ONLY four fresh diagnostic processes; no canonical timing samples.
+
+Native plain/CUPTI observed, stock plain/public Torch profiler observed. Each
+uses exact V8 requests/cold-warm policy/CPU/RAM/engine limits and original owner.
+"""
+import argparse
+from dataclasses import asdict
+import gzip
+import hashlib
+import json
+import os
+from pathlib import Path
+import signal
+import sys
+import time
+import urllib.request
+
+from mapping_contract_v3 import (verify_preparation as verify_mapping_preparation,
+    bounded_controller_metadata)
+from container_contracts import (HERE,TREE_LIMIT,build_binding,fingerprint,frozen,host_parents,
+    read,require,sha,tree_bound,write)
+from launch_derivative import derive
+from namespace_control_v3 import Controller
+from completion_v2 import verify_preparation as verify_finalization, write_completion
+
+
+def idle(bench,url):
+    deadline=time.monotonic()+30
+    previous=None
+    while time.monotonic()<deadline:
+        raw=bench.snapshot(url+'/health',None)
+        require(raw['ok'],'actual original native health')
+        state=raw['value']['admission']
+        require(state['kind']=='resident_gpu' and state['failed_gpu_quanta']==0,'healthy original GPU engine')
+        if state['active_sessions']==state['queued_requests']==state['output_credit_parked_requests']==0:
+            if previous==raw['value']: return raw
+            previous=raw['value']
+        else: previous=None
+        time.sleep(.1)
+    raise TimeoutError('native physical work did not become terminal before window boundary')
+
+
+def stock_traces(owner,audit):
+    output=[]
+    directory=owner.directory/'traces'
+    require(directory.is_dir() and not directory.is_symlink(),'actual original public profiler output')
+    paths=sorted(directory.rglob('*'))
+    require(len(paths)<=32,'bounded actual profiler artifact count')
+    for path in paths:
+        require(not path.is_symlink(),'owned original profiler files')
+        if path.is_dir(): continue
+        require(path.is_file() and path.stat().st_size<=256*1024**2,'bounded public profiler artifact')
+        if not path.name.endswith(('.pt.trace.json','.pt.trace.json.gz')): continue
+        with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
+            raw=stream.read(256*1024**2+1)
+        require(len(raw)<=256*1024**2,'bounded entire decoded public trace')
+        value=json.loads(raw)
+        if any(event.get('cat')=='kernel' for event in value.get('traceEvents',[])):
+            result=audit(value)
+            result.update(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
+                decoded_sha256=hashlib.sha256(raw).hexdigest())
+            output.append(result)
+    require(len(output)==1,'one actual TP1 CUDA worker trace, no CPU-only substitution')
+    return output
+
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run',action='store_true')
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--mapping-freeze',required=True)
+    parser.add_argument('--launch-directory',type=Path,required=True)
+    parser.add_argument('--collector-artifact-directory',type=Path,required=True)
+    parser.add_argument('--concurrency',type=int,choices=(1,2,4,8),required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    require(args.run,'ROOT reviewed explicit actual execution only')
+    verify_mapping_preparation(args.mapping_freeze)
+    verify_finalization()
+    parent=frozen(args.own_freeze); host_parents(parent)
+    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
+    benchmark=Path(parent['parents']['benchmark'])
+    sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_compare_v8 import lifecycle_spec,load_tool,opener
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from support_v3 import source_check
+    from owned_plan_v8 import install
+    import owned_container_v4 as owned
+    v8=parent['benchmark_v8_manifest']['sha256']
+    pins=verify_preparation(v8); verify_small_sources()
+    require(sha(owned.__file__)==parent['owned_helper_sha256'],'actual original53ee lifecycle helper import')
+    launch_dir=args.launch_directory.resolve(strict=True)
+    require(launch_dir.parent==benchmark and not args.launch_directory.is_symlink(),'original V8 owned launch directory')
+    protocol=validate_protocol(read(launch_dir/'protocol.json'))
+    require(args.concurrency in protocol['concurrencies'] and protocol['source_sha256']==pins['models'][protocol['model']]['gguf']['sha256']
+            and protocol['tokenizer_sha256']==pins['tokenizer_sha256'],'exact V8 workload/concurrency/source identity')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    conversion=Path(project['conversion_directory'])
+    commands={label:launch(protocol,pins,v8,label,conversion) for label in ('zerogravity','vllm')}
+    require(read(launch_dir/'launches.json')==commands,'exact original canonical engine commands')
+    diagnostics=read(launch_dir/'diagnostic-launches.json')
+    stock_observed=launch(protocol,pins,v8,'vllm',conversion,profiler=True)
+    require(diagnostics['cuda_activity']==stock_observed,'exact existing public stock profiler command')
+    source=pins['models'][protocol['model']]
+    source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+    require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+            and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256'],
+            'actual unchanged projector/native executable before diagnostic')
+    artifact=args.collector_artifact_directory.resolve(strict=True)
+    require(not args.collector_artifact_directory.is_symlink(),'original ROOT artifact directory')
+    build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+    output=args.output.resolve()
+    require(output.parent==HERE and not output.exists(),'fresh exclusive owned diagnostic output')
+    output.mkdir(mode=0o700)
+    compare,bench=load_tool('serve-compare.py'),load_tool('serve-bench.py')
+    install(compare.LIFECYCLE)
+    urls={label:f'http://127.0.0.1:{port}' for label,port in zip(('zerogravity','vllm'),protocol['ports'])}
+    spec=lifecycle_spec(protocol,commands)
+    plan=compare.LIFECYCLE['LifecyclePlan'](spec,urls,str(output/'owned'))
+    alias='qwen35-'+protocol['model']+'-same-gguf'
+    report=dict(status='UNQUALIFIED',schema='zg.cupti.container.diagnostic.v1',own_freeze_sha256=args.own_freeze,
+        benchmark_v8_sha256=v8,capacity_freeze_sha256=parent['capacity_manifest']['sha256'],
+        protocol=protocol,protocol_path=str(launch_dir/'protocol.json'),
+        protocol_sha256=sha(launch_dir/'protocol.json'),concurrency=args.concurrency,
+        collector_build_sha256=sha(artifact/'build.json'),collector_sha256=build['collector_sha256'],
+        owned_cleanup_source_sha256=parent['owned_helper_sha256'],samples=[],
+        performance_timing_usable=False,matched_hardware_counts_qualified=False,
+        source_correct_goodput_qualified=False,numerical_profiles_identical=False,
+        native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+    report['artifact_finalization_followup_sha256']=verify_finalization()
+    report['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
+    allowed=os.sched_getaffinity(0)
+    require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
+    urllib.request.install_opener(opener())
+    try:
+        os.sched_setaffinity(0,protocol['cpu_affinity'])
+        for label,mode in (('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed')):
+            base=stock_observed if label=='vllm' and mode=='observed' else commands[label]
+            spec_command=derive(base,label,mode,args.own_freeze,artifact)
+            plan.spec['providers'][label]['argv']=spec_command
+            sample=dict(provider=label,mode=mode,policy=protocol['policy'],warmup=None)
+            with plan.sample(label,lambda:compare.preflight(urls[label],alias,None)) as owner:
+                if protocol['policy']=='warmed_process':
+                    warm=bench.stream_one(urls[label],alias,protocol['requests'][0]['prompt'],
+                        protocol['max_tokens'],180,None,request_id='matched-diag-warmup',return_token_ids=True)
+                    owner.record_warmup(asdict(warm)); sample['warmup']=asdict(warm)
+                    require(warm.completed,'exact canonical unmeasured warmed_process request')
+                before=idle(bench,urls[label]) if label=='zerogravity' else None
+                if before is not None: write(owner.directory/'health-before.json',before)
+                controller=None; started=False; primary=None; cleanup=[]
+                try:
+                    if mode=='observed':
+                        if label=='zerogravity':
+                            controller=Controller(owner,owned,parent,args.mapping_freeze)
+                            start=controller.command('START')
+                            started=True
+                            write(owner.directory/'window-start.json',start)
+                        else:
+                            sample['profile_start']=post(urls[label],'/start_profile')
+                            started=True
+                    measured=bench.bench(urls[label],alias,protocol['requests'][0]['prompt'],len(protocol['requests']),
+                        args.concurrency,protocol['max_tokens'],180,None,workload=protocol['requests'],
+                        return_token_ids=True,max_pending=protocol['max_pending'],
+                        launch_deadline_seconds=protocol['launch_deadline_seconds'],run_id='matched-cuda-activity')
+                    fingerprint(measured); owner.record_measurement(measured)
+                    sample['observation']=measured
+                    after=idle(bench,urls[label]) if label=='zerogravity' else None
+                    if after is not None: write(owner.directory/'health-after.json',after)
+                except BaseException as error:
+                    primary=error
+                finally:
+                    if started:
+                        try:
+                            if controller is not None:
+                                stop=controller.command('STOP')
+                                write(owner.directory/'window-stop.json',stop)
+                            else: sample['profile_stop']=post(urls[label],'/stop_profile')
+                        except BaseException as error: cleanup.append('collector stop: '+repr(error))
+                    if controller is not None:
+                        try: controller.close()
+                        except BaseException as error: cleanup.append('controller close: '+repr(error))
+                        try: write(owner.directory/'namespace-controller.json',bounded_controller_metadata(controller.receipt))
+                        except BaseException as error: cleanup.append('controller receipt: '+repr(error))
+                    require(not cleanup and primary is None,
+                            'diagnostic primary='+repr(primary)+'; collector_cleanup='+repr(cleanup))
+            # All original owner/process/group/container/monitor/pipe cleanup finished.
+            require(not plan.blocked,'complete original retirement before next provider')
+            sample['owned_receipt_path']=str(owner.receipt_path)
+            if mode=='observed' and label=='vllm': sample['actual_recorded_CUDA_activity']=stock_traces(owner,audit)
+            report['samples'].append(sample)
+            tree_bound(owner.directory,768*1024**2)
+            tree_bound(output)
+        for label in ('zerogravity','vllm'):
+            pair=[item for item in report['samples'] if item['provider']==label]
+            require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),
+                    'complete plain/observed wire IDs/text/finish/usage exact for '+label)
+        report['output_transparency_exact']=True
+        report['status']='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        report['failure']=repr(error)[:4096]
+        raise
+    finally:
+        # All process/container owners already retired; independent evidence stages
+        # must still run if affinity restoration or a post-run source scan fails.
+        failures=[]
+        try: os.sched_setaffinity(0,allowed)
+        except BaseException as error: failures.append('restore client affinity: '+repr(error))
+        report['lifecycle_manifest']=str(plan.directory/'lifecycle.json')
+        try:
+            report['lifecycle_manifest_sha256']=sha(plan.directory/'lifecycle.json')
+        except BaseException as error: failures.append('lifecycle digest: '+repr(error))
+        try:
+            frozen(args.own_freeze); host_parents(parent); verify_preparation(v8); verify_small_sources()
+            verify_mapping_preparation(args.mapping_freeze)
+        except BaseException as error: failures.append('post-run immutable source closure: '+repr(error))
+        try:
+            source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+            require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+                    and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256']
+                    and sha(artifact/'collector.so')==build['collector_sha256']
+                    and sha(artifact/'build.json')==report['collector_build_sha256'],
+                    'unchanged actual model/projector/binary/collector/build after retirement')
+        except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
+        report['finalization_errors']=failures
+        if failures: report['status']='UNQUALIFIED'
+        write_completion(output,'report.json',report)
+        verify_mapping_preparation(args.mapping_freeze)
+        if failures:
+            raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
+
+
+
+if __name__=='__main__':
+    def interrupted(_signum,_frame): raise KeyboardInterrupt('ROOT container diagnostic interrupted')
+    saved=signal.signal(signal.SIGTERM,interrupted)
+    try: main()
+    finally: signal.signal(signal.SIGTERM,saved)
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/source-pins.json
@@ -0,0 +1,153 @@
+{
+  "actual_model_DSO_or_provider_payloads_read_or_hashed_by_author": false,
+  "diagnostics_are_metadata_only_not_actual_GPU_mapping_qualification": true,
+  "original_finalization_freeze_sha256": "2b82ca5f8335c1abbada8b9ac624eb90dfd7558b498ca3b666ab7866660358e2",
+  "schema": "zg.cupti.same-fd-mapping.sources.v3",
+  "unchanged_source_and_diagnostic_metadata": [
+    {
+      "bytes": 73289,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/NEW_FILES.review.patch",
+      "sha256": "f17bb18a8bb5ab5e5a57499aa1089d889fa1d1df6b21d9ef824f6d161ae3e05b"
+    },
+    {
+      "bytes": 8713,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/README.md",
+      "sha256": "2eee08c2526be0ef273c6b2ceb957b704f083f8c0e827ac7ee81322c5e949179"
+    },
+    {
+      "bytes": 3105,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/STATIC_AUDIT.json",
+      "sha256": "b7087eda04623a897f694b65d6b1648d65a8ffd8d2c923623b97450c62e03304"
+    },
+    {
+      "bytes": 1651,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/TODO.md",
+      "sha256": "2debefd7c25d04d1648efabed5412ee198353a3096f4fb1929c9a34536d94cca"
+    },
+    {
+      "bytes": 2149,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/CAPACITY.review.diff",
+      "sha256": "33a04887e9d7fcf9b3016dedcfef537d0185ada29824bd6e024056f3856ebd4f"
+    },
+    {
+      "bytes": 1818,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/README.md",
+      "sha256": "be65f53ffb83f50481731d59b288fefa5b1c8a739f4213f792dab6e23ffdfed0"
+    },
+    {
+      "bytes": 5550,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/activity.py",
+      "sha256": "36945a540508e6963c919a1d7d576b913de85a54b2374175755cd45e06469f28"
+    },
+    {
+      "bytes": 18769,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/collector_capacity.cpp",
+      "sha256": "83c63c0431d867fc437d876b9bf48b0e7455dfa5b91ce2ced9188c9488803838"
+    },
+    {
+      "bytes": 1195,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/freeze-manifest.json",
+      "sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd"
+    },
+    {
+      "bytes": 3121,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/test_activity.py",
+      "sha256": "92cf705d8744b0d681f514016ad63bb0bd186613085d26f1693accf8cf9d7309"
+    },
+    {
+      "bytes": 395,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity_v1_loader.py",
+      "sha256": "e1a47e4c176947eeb2f1e42e932372892ada49ecef0475fa92c9d9ce5c8c683b"
+    },
+    {
+      "bytes": 2165,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/completion_v2.py",
+      "sha256": "5192da872cea8fd97abf2ff5d584d500d3a4dd84b9aa33ba4d1c80fcd4462116"
+    },
+    {
+      "bytes": 4232,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/container_activity.py",
+      "sha256": "4f220f13dcbd75b85972bd30611ff54592e0e8393d9c653c02cde6d71513e73e"
+    },
+    {
+      "bytes": 4087,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/container_contracts.py",
+      "sha256": "bc50255dd0bee58f126b3046f4c02f315d961e37b3513ff3f3def59f4b4ee0bd"
+    },
+    {
+      "bytes": 1274,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/README.md",
+      "sha256": "80c9edee3e13b74b952f635e0a31c44551c37363ea33de5ae736eb53f493c8c1"
+    },
+    {
+      "bytes": 1976,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/prepared-manifest.json",
+      "sha256": "2b82ca5f8335c1abbada8b9ac624eb90dfd7558b498ca3b666ab7866660358e2"
+    },
+    {
+      "bytes": 4501,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/freeze-manifest.json",
+      "sha256": "07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee"
+    },
+    {
+      "bytes": 1454,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/launch_derivative.py",
+      "sha256": "d2ebea43d9f54c1ddd1a4e2021edd6b1768342e7a8e23f3410ed36f46e03d65e"
+    },
+    {
+      "bytes": 7177,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/namespace_control.py",
+      "sha256": "c51b455e05bbfc1f33bf59f8b05a84298a9e45509989443743305c5999a04bd1"
+    },
+    {
+      "bytes": 4561,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/native_exec.py",
+      "sha256": "b8ebe7f0bb6b40162a2fe9659eaeb6a7d292a0514ab25b184e214a377d17dc67"
+    },
+    {
+      "bytes": 1703,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/parent-bindings.json",
+      "sha256": "2a6204cca537c7d76d1b6536e3c357149bae21f8eae14ac83d83832c42999a66"
+    },
+    {
+      "bytes": 8061,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/root-container-map-probe-2/root-pre-start-mapping-metadata.json",
+      "sha256": "e42390c59da0b688fc3109bd08c288f8c550a2b5becf7e4d7d57f2a9c1c2bfe9"
+    },
+    {
+      "bytes": 1448,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/root-readonly-FD-mapping-diagnostic-3.json",
+      "sha256": "f4a7fdaa28bed5310e8b6354c15d35a0ab6cd83c3024f81cd57e388d6f931a13"
+    },
+    {
+      "bytes": 13958,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/run_container.py",
+      "sha256": "952865e02a59e393a615b89a5b61a15ae78beec6814612dcb5e9a54a5454aa60"
+    },
+    {
+      "bytes": 14155,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/run_container_v2.py",
+      "sha256": "da7b311f5827d2da0e9aaacfd50c607eea1c85529b2e3d88e432a1bec32a2c59"
+    },
+    {
+      "bytes": 2250,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/test_completion_v2.py",
+      "sha256": "ae33fc62fe5deb43b647f3c7424a8873d9649782e0048d7b8dc9e15ba990f5e1"
+    },
+    {
+      "bytes": 4757,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/test_contracts.py",
+      "sha256": "4907c608c67b07d2353463288e83301acc827ad25a219d141c08c94eade9c456"
+    },
+    {
+      "bytes": 15398,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/validate_container.py",
+      "sha256": "8f72272b9904a99dd7ecef0a82fab15cbcb73b99101b84762eb1b86eab813a41"
+    },
+    {
+      "bytes": 15778,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/validate_container_v2.py",
+      "sha256": "13f05cdbaf0783d247cddbd3ab6b7ad91ea21192a66f836ec150a06d3e9428bc"
+    }
+  ]
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/test_fd_mapping_v3.py
@@ -0,0 +1,212 @@
+"""ROOT-only synthetic/source-free fixtures. Author did not execute these."""
+import copy
+import hashlib
+import os
+from pathlib import Path
+import tempfile
+import unittest
+from unittest.mock import patch
+
+import mapping_contract_v3  # original source path bootstrap only
+from container_contracts import require
+import namespace_control_v3 as control
+import verified_fd_mapping_v3 as mapping
+
+
+def vma(start, end, permissions='r--p', offset=0, minor=43, inode=791,
+        path='/synthetic/artifact'):
+    return f'{start:x}-{end:x} {permissions} {offset:08x} 00:{minor:x} {inode} {path}\n'
+
+
+def fixture():
+    # Intentionally unequal synthetic devices. These are test data, never a
+    # production device exception or a claim about this machine's filesystems.
+    identity = dict(device=os.makedev(0, 42), inode=791, bytes=16384,
+                    mode=0o100444, mtime_ns=100, ctime_ns=200)
+    before = vma(0x1000, 0x2000, inode=0, minor=0, path='')
+    during = before + vma(0x3000, 0x4000)
+    proof = dict(schema='zg.cupti.verified-fd-mapping.v3',
+                 opened_path='/proc/123/root/runtime/zerogravity', hash_chunk_bytes=mapping.HASH_CHUNK,
+                 no_executable_mapping=True, readonly_private_offset_zero=True,
+                 verified_fd_closed=True, temporary_mapping_closed=True,
+                 stat_before=identity, stat_before_mapping=identity, stat_after_mapping=identity,
+                 stat_after=identity, path_stat_after=identity,
+                 descriptor_kernel_path_before='/synthetic/artifact',
+                 descriptor_kernel_path_after='/synthetic/artifact',
+                 sha256_before='1'*64, sha256_after='1'*64, page_size=4096,
+                 local_maps_before=before, local_maps_during=during, local_maps_after=before,
+                 readonly_mapping_line=vma(0x3000, 0x4000).rstrip('\n'),
+                 kernel_mapping_key=dict(device=os.makedev(0, 43), major=0, minor=43, inode=791))
+    rows = [vma(0xa000, 0xb000, path='/runtime/zerogravity').rstrip('\n'),
+            vma(0xb000, 0xd000, 'r-xp', 4096, path='/runtime/zerogravity').rstrip('\n')]
+    exe = dict(path='/proc/123/exe', executable_link='/runtime/zerogravity',
+               stat_before=identity, stat_after=identity, sha256_before='1'*64,
+               sha256_after='1'*64, verified_fd_closed=True)
+    value = dict(container_path='/runtime/zerogravity', bytes=identity['bytes'], sha256='1'*64,
+                 inode=identity['inode'], device=identity['device'], stat_device=identity['device'],
+                 mapping_device=os.makedev(0, 43), kernel_mapping_key=proof['kernel_mapping_key'],
+                 mappings=rows, mappings_after=rows, fd_mapping=proof, exe_binding=exe)
+    return proof, value
+
+
+class MappingFixtures(unittest.TestCase):
+    def check(self, value):
+        return mapping.validate_mapped_file(value, '1'*64, 123, '/runtime/zerogravity')
+
+    def test_unequal_stat_and_mapping_devices_require_exact_fd_derived_key(self):
+        proof, value = fixture()
+        identity, key = self.check(value)
+        self.assertNotEqual(identity['device'], key['device'])
+        self.assertEqual(key, proof['kernel_mapping_key'])
+
+    def test_equal_devices_are_observed_without_a_filesystem_exception(self):
+        proof, value = fixture()
+        for field in ('stat_before', 'stat_before_mapping', 'stat_after_mapping', 'stat_after', 'path_stat_after'):
+            proof[field]['device'] = os.makedev(0, 43)
+        value['device'] = value['stat_device'] = os.makedev(0, 43)
+        self.check(value)
+
+    def test_substituting_stat_device_for_mapping_device_is_refused(self):
+        _, value = fixture()
+        value['mapping_device'] = value['stat_device']
+        with self.assertRaises(ValueError):
+            self.check(value)
+
+    def test_target_foreign_device_inode_path_or_extent_is_refused(self):
+        for replacement in (dict(minor=44), dict(inode=792), dict(path='/foreign/artifact'), dict(offset=16384)):
+            _, value = fixture()
+            fields = dict(offset=4096, path='/runtime/zerogravity')
+            fields.update(replacement)
+            value['mappings'][1] = vma(0xb000, 0xd000, 'r-xp', **fields).rstrip('\n')
+            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
+                self.check(value)
+        _, value = fixture()
+        value['mappings'][1] = vma(0xb000, 0xf000, 'r-xp', 4096,
+                                  path='/runtime/zerogravity').rstrip('\n')
+        with self.assertRaises(ValueError):
+            self.check(value)
+
+    def test_all_target_rows_must_preserve_executable_mapping(self):
+        _, value = fixture()
+        value['mappings'][1] = vma(0xb000, 0xd000, 'r--p', 4096,
+                                  path='/runtime/zerogravity').rstrip('\n')
+        with self.assertRaises(ValueError):
+            self.check(value)
+
+    def test_local_writable_executable_shared_or_nonzero_offset_is_refused(self):
+        for permissions, offset in (('rw-p', 0), ('r-xp', 0), ('r--s', 0), ('r--p', 4096)):
+            proof, value = fixture()
+            proof['local_maps_during'] = proof['local_maps_before'] + vma(0x3000, 0x4000, permissions, offset)
+            with self.subTest(permissions=permissions, offset=offset), self.assertRaises(ValueError):
+                self.check(value)
+
+    def test_local_path_inode_or_hash_mismatch_is_refused(self):
+        for changes in (dict(descriptor_kernel_path_after='/foreign/file'),
+                        dict(descriptor_kernel_path_before='/foreign/file', descriptor_kernel_path_after='/foreign/file'),
+                        dict(sha256_after='2'*64), dict(sha256_before='2'*64)):
+            proof, value = fixture()
+            proof.update(changes)
+            with self.subTest(changes=changes), self.assertRaises(ValueError):
+                self.check(value)
+        proof, value = fixture()
+        proof['local_maps_during'] = proof['local_maps_before'] + vma(0x3000, 0x4000, inode=792)
+        with self.assertRaises(ValueError):
+            self.check(value)
+
+    def test_preexisting_ambiguous_or_coalesced_local_mapping_is_refused(self):
+        for mode in ('preexisting', 'ambiguous', 'coalesced'):
+            proof, value = fixture()
+            if mode == 'preexisting':
+                proof['local_maps_before'] = proof['local_maps_during']
+            elif mode == 'ambiguous':
+                proof['local_maps_during'] += vma(0x5000, 0x6000)
+            else:
+                proof['local_maps_during'] = proof['local_maps_before'] + vma(0x3000, 0x5000)
+            with self.subTest(mode=mode), self.assertRaises(ValueError):
+                self.check(value)
+
+    def test_failed_map_or_fd_retirement_and_remaining_vma_are_refused(self):
+        for changes in (dict(temporary_mapping_closed=False), dict(verified_fd_closed=False)):
+            proof, value = fixture()
+            proof.update(changes)
+            with self.subTest(changes=changes), self.assertRaises(ValueError):
+                self.check(value)
+        proof, value = fixture()
+        proof['local_maps_after'] = proof['local_maps_during']
+        with self.assertRaises(ValueError):
+            self.check(value)
+
+    def test_stat_path_mutation_and_target_vma_replacement_are_refused(self):
+        proof, value = fixture()
+        proof['stat_after'] = dict(proof['stat_after'], ctime_ns=201)
+        with self.assertRaises(ValueError):
+            self.check(value)
+        _, value = fixture()
+        value['mappings_after'] = list(value['mappings'])
+        value['mappings_after'][0] = value['mappings_after'][0].replace('a000-b000', '9000-a000')
+        with self.assertRaises(ValueError):
+            self.check(value)
+
+    def test_native_exe_foreign_path_identity_hash_and_unretired_owner_refuse(self):
+        for changes in (dict(path='/proc/124/exe'), dict(executable_link='/foreign/executable'),
+                        dict(sha256_after='2'*64), dict(verified_fd_closed=False)):
+            _, value = fixture()
+            value['exe_binding'].update(changes)
+            with self.subTest(changes=changes), self.assertRaises(ValueError):
+                self.check(value)
+
+    def test_malformed_maps_bounds_overlap_and_signed_geometry_refuse(self):
+        for raw in ('', 'not a map\n', '1000-2000 r--p 0 00:00 -1\n',
+                    vma(0x1000, 0x3000)+vma(0x2000, 0x4000),
+                    vma(0x1000, 0x2000).rstrip('\n'), 'x'*(mapping.LOCAL_MAP_BYTES+1)):
+            with self.subTest(prefix=raw[:40]), self.assertRaises(ValueError):
+                mapping.parse_maps(raw)
+
+    def test_metadata_only_real_readonly_same_fd_fixture(self):
+        # ROOT only. A temporary synthetic ordinary file, never an ELF/provider.
+        with tempfile.TemporaryDirectory(prefix='zg-fd-mapping-fixture-') as directory:
+            path = Path(directory)/'synthetic-bytes.bin'
+            raw = bytes(range(256))*128
+            path.write_bytes(raw)
+            digest = hashlib.sha256(raw).hexdigest()
+            proof = mapping.verify_opened(path, digest)
+            identity, key = mapping.validate_proof(proof, digest, str(path))
+            self.assertEqual(identity['bytes'], len(raw))
+            self.assertEqual(key['inode'], path.stat().st_ino)
+            self.assertEqual(proof['sha256_before'], proof['sha256_after'])
+            self.assertEqual(path.read_bytes(), raw)
+
+    def test_unknown_symlink_or_changed_hash_never_becomes_mapping_proof(self):
+        with tempfile.TemporaryDirectory(prefix='zg-fd-mapping-negative-') as directory:
+            path = Path(directory)/'ordinary.bin'
+            path.write_bytes(b'source-free fixture\n')
+            alias = Path(directory)/'alias.bin'
+            alias.symlink_to(path)
+            digest = hashlib.sha256(path.read_bytes()).hexdigest()
+            with self.assertRaises(OSError):
+                mapping.verify_opened(alias, digest)
+            with self.assertRaises(ValueError):
+                mapping.verify_opened(path, '0'*64)
+
+    def test_original_host_peer_refusal_remains_before_control_send(self):
+        import struct
+        owner = object.__new__(control.Controller)
+        owner.pid = 32100
+        owner.nonce = 'a'*32
+        owner.live = lambda: None
+        owner.receipt = dict(events=[])
+        class Socket:
+            def __enter__(self): return self
+            def __exit__(self, *args): pass
+            def settimeout(self, *args): pass
+            def connect(self, *args): pass
+            def getsockopt(self, *args): return struct.pack('3i', 1, 1000, 1000)
+        with patch('namespace_control_v3.socket.socket', return_value=Socket()), \
+                patch('namespace_control_v3.os.geteuid', return_value=1000):
+            with self.assertRaises(ValueError):
+                owner.command('START')
+        self.assertEqual(owner.receipt['events'], [])
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/validate_container_v3.py
@@ -0,0 +1,234 @@
+"""ROOT-only independent complete-workload/namespace/activity/owner validator.
+
+Stock public Torch profiler loss completeness remains explicitly UNQUALIFIED;
+recorded kernel counts cannot establish source-correct goodput or timing speed.
+"""
+import argparse
+import hashlib
+from pathlib import Path
+import sys
+
+from mapping_contract_v3 import verify_preparation as verify_mapping_preparation
+from capacity_v1_loader import activity
+from container_activity import summarize
+from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
+from launch_derivative import derive
+from verified_fd_mapping_v3 import validate_mapped_file
+from completion_v2 import verify_preparation as verify_finalization, write_completion
+
+
+def successful_owner(receipt,parent):
+    require(receipt['status']=='OWNERS_CLOSED' and receipt['cleanup_errors']==[]
+            and receipt['primary_failure'] is None and receipt['ready'] is True
+            and receipt['owned_container_absent_after_reap'] is True,'actual complete successful owned retirement')
+    cleanup=receipt['foreground_cleanup']
+    require(cleanup['pgid']==receipt['owned_cli_pid']>0 and cleanup['leader_reaped'] is True
+            and cleanup['group_absent'] is True and cleanup['all_nonzero_signals_before_reap'] is True
+            and cleanup['kill_attempts_before_reap']==2 and cleanup['leader_reap_attempts'] in (1,2)
+            and cleanup['stage_errors']==[],'all independent group signals before reap and final absence')
+    log=receipt['log_cleanup']
+    require(log['reader_complete'] is True and log['reader_error'] is None,'actual complete original log/pipe EOF')
+    observations=receipt['physical_observations']
+    require(observations['failures']==[] and observations['samples']
+            and observations['exact_peak'] is False
+            and observations['complete_physical_allocator_accounting'] is False,
+            'actual bounded original physical observations, no invented allocator peak')
+    for row in observations['samples']:
+        if row['compute_apps']:
+            require(all(app['pid'] in row['authenticated_container_host_pids'] for app in row['compute_apps']),
+                    'every actual GPU PID bound to original public container numeric HPIDs')
+    require(receipt['GPU_quiet_after'] and all(not row['compute_apps'] for row in receipt['GPU_quiet_after'][-3:]),
+            'actual final card quiet after original owner cleanup')
+
+
+def workload(result,protocol,concurrency):
+    require(result['offered_requests']==len(protocol['requests']) and result['concurrency']==concurrency
+            and result['max_pending']==protocol['max_pending']
+            and result['launch_deadline_seconds']==protocol['launch_deadline_seconds']
+            and result['run_id']=='matched-cuda-activity','exact canonical workload denominator and scheduling policy')
+    fingerprint(result)
+    require(len(result['requests'])==len(protocol['requests']),'all original canonical offers')
+    for index,(actual,expected) in enumerate(zip(result['requests'],protocol['requests'])):
+        require(actual['input_index']==index and actual['request_id']==expected['request_id']
+                and actual['prompt_sha256']==hashlib.sha256(expected['prompt'].encode()).hexdigest()
+                and actual['max_tokens']==expected['max_tokens'], 'actual full original prompt/max-token/request identity')
+
+
+def driver_identity(rows,host_pid):
+    ids=set()
+    for row in rows:
+        if row['kind']=='driver':
+            require(type(row['process']) is int,'actual raw driver process ID')
+            ids.add(row['process'])
+            require(len(ids)==1,'single native process driver identity; never merge namespaces')
+    require(len(ids)==1 and ids<={1,host_pid},'raw driver ID equals one authenticated process representation')
+    return next(iter(ids))
+
+
+def native_activity(directory,receipt,parent,report):
+    control=read(directory/'namespace-controller.json')
+    actual=read(directory/'native-exec-attestation.json')
+    require(control['schema']=='zg.cupti.container.controller.v3' and
+            control['fd_mapping_followup_sha256']==report['fd_mapping_followup_sha256'],
+            'actual reviewed additive same-FD mapping controller')
+    require(control['pidfd_closed'] is True and control['reported_namespace_pid']==1
+            and control['exec_attestation']==actual and actual['nonce']==receipt['owned_nonce']
+            and control['binding']['owned_nonce']==receipt['owned_nonce']
+            and control['binding']['owned_name']==receipt['owned_name'], 'original owner/exec/controller binding')
+    hp=control['binding']['host_init_pid']
+    require(hp>0 and actual['pid_namespace_pid']==1 and actual['namespace_inodes']==control['host_before']['namespace_inodes']
+            and control['host_before']['NSpid'][0]==hp and control['host_before']['NSpid'][-1]==1,
+            'host init PID and actual namespace PID preserved independently')
+    require(actual['file_size_limits']==[FILE_LIMIT,FILE_LIMIT], 'actual finite diagnostic file-size readback')
+    for key,field in (('native','native_sha256'),('collector','collector_sha256'),('cupti','cupti_sha256')):
+        require(control['mapped_files'][key]['sha256']==actual[field] and control['mapped_files'][key]['mappings'],
+                'actual complete mapped artifact identity retained')
+        path={'native':'/runtime/zerogravity','collector':'/collector-artifact/collector.so',
+              'cupti':parent['cupti_library']['container_path']}[key]
+        validate_mapped_file(control['mapped_files'][key],actual[field],hp,path)
+    require(actual['collector_on_outer_python'] is False and actual['collector_requested_only_at_this_exec'] is True
+            and actual['native_sha256']==report['protocol']['zerogravity_binary_sha256']
+            and actual['collector_sha256']==report['collector_sha256']
+            and actual['collector_capacity_freeze_sha256']==report['capacity_freeze_sha256']
+            and actual['cupti_sha256']==parent['cupti_library']['sha256'],'only final-exec exact artifact/provider closure')
+    require(len(control['events'])==2,'one complete original native activity window')
+    start,stop=read(directory/'window-start.json'),read(directory/'window-stop.json')
+    for event,verb,value in zip(control['events'],('START','STOP'),(start,stop)):
+        require(event['verb']==verb and event['reply']==value and event['host_peer_pid']==hp
+                and event['host_peer_uid']==control['host_controller_uid'] and event['reported_namespace_pid']==1,
+                'actual host SO_PEERCRED authenticated independently from CUPTI namespace identity')
+    require(start['nonce']==stop['nonce']==receipt['owned_nonce']
+            and start['begin']==stop['begin'] and start['end']==0
+            and start['start_monotonic']==stop['start_monotonic']<stop['stop_monotonic'], 'one actual timestamped window')
+    path=directory/'observed-activity/activity.jsonl'
+    require(read(directory/'observed-activity/collector-stop.json')==stop,'persisted stop exactly acknowledged')
+    digest=sha(path)
+    driver_pid=driver_identity(activity.rows(path,stop['bytes']),hp)
+    driver_ids={driver_pid}
+    result=summarize(activity.rows(path,stop['bytes']),stop,1,driver_pid)
+    require(sha(path)==digest,'immutable entire raw trace across independent complete passes')
+    before=read(directory/'health-before.json')['value']['admission']
+    after=read(directory/'health-after.json')['value']['admission']
+    for value in (before,after):
+        require(value['active_sessions']==value['queued_requests']==value['output_credit_parked_requests']==0
+                and value['failed_gpu_quanta']==0,'actual terminal original GPU work at window boundaries')
+    require(after['accepted_output_tokens']>before['accepted_output_tokens'],'actual canonical output within window')
+    result.update(raw_activity_sha256=digest,raw_bytes=path.stat().st_size,namespace_pid=1,
+        authenticated_init_host_pid=hp,actual_raw_driver_process_ids=sorted(driver_ids),
+        driver_process_representation='namespace' if driver_pid==1 else 'host',
+        source_dispatch_delta=after['completed_forward_kernel_dispatches']-before['completed_forward_kernel_dispatches'],
+        source_dispatches_are_not_driver_measurements=True,all_retained_records_lossless=True)
+    return result
+
+
+def main():
+    parser=argparse.ArgumentParser()
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--mapping-freeze',required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    verify_mapping_preparation(args.mapping_freeze)
+    verify_finalization()
+    parent=frozen(args.own_freeze); host_parents(parent)
+    output=args.output.resolve(strict=True)
+    require(output.parent==HERE,'owned diagnostic output')
+    report=read(output/'report.json')
+    require(report['fd_mapping_followup_sha256']==verify_mapping_preparation(args.mapping_freeze),
+            'actual report uses exact reviewed same-FD mapping source closure')
+    require(report.get('artifact_finalization_followup_sha256')==verify_finalization(),
+            'actual report used reviewed final receipt reservation')
+    require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+            and report['own_freeze_sha256']==args.own_freeze
+            and report['benchmark_v8_sha256']==parent['benchmark_v8_manifest']['sha256']
+            and report['capacity_freeze_sha256']==parent['capacity_manifest']['sha256']
+            and report['output_transparency_exact'] is True and report['finalization_errors']==[]
+            and report['performance_timing_usable'] is False
+            and report['native_or_stock_math_changes'] is False,'executed bounded diagnostic-only scope')
+    benchmark=Path(parent['parents']['benchmark']); sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from cgroup_readback_v3 import validate_caps
+    from run_container_v3 import stock_traces
+    pins=verify_preparation(report['benchmark_v8_sha256']); verify_small_sources()
+    protocol=validate_protocol(report['protocol'])
+    protocol_path=Path(report['protocol_path'])
+    require(protocol_path.parent.parent==benchmark and protocol_path.name=='protocol.json'
+            and sha(protocol_path)==report['protocol_sha256'] and read(protocol_path)==protocol,
+            'actual immutable original V8 launch protocol file and complete value')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    require(sha(report['lifecycle_manifest'])==report['lifecycle_manifest_sha256'],'complete original lifecycle manifest')
+    lifecycle=read(report['lifecycle_manifest'])
+    require(lifecycle['owned_cleanup_source_sha256']==parent['owned_helper_sha256']
+            and lifecycle['adapter_source_sha256']==sha(benchmark/'owned_plan_v8.py')
+            and len(lifecycle['receipts'])==len(report['samples'])==4,'all four original owner receipts')
+    order=(('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed'))
+    native_summary=stock_summary=None
+    for sample,receipt_path,expected in zip(report['samples'],lifecycle['receipts'],order):
+        label,mode=expected
+        require((sample['provider'],sample['mode'])==expected and sample['owned_receipt_path']==receipt_path,
+                'all four actual distinct ordered samples')
+        receipt=read(receipt_path); directory=Path(receipt_path).parent
+        require(output in directory.parents,'original owned sample stays in this exclusive output')
+        tree_bound(directory,768*1024**2)
+        successful_owner(receipt,parent)
+        bootstrap=read(directory/'bootstrap.json')
+        validate_caps(bootstrap['cgroup_readback'],protocol['cpu_affinity'])
+        require(bootstrap['provider']==label and bootstrap['preparation_sha256']==report['benchmark_v8_sha256']
+                and bootstrap['stock_math_modified'] is False
+                and bootstrap['actual_cpu_affinity']==sorted(protocol['cpu_affinity']), 'original runtime caps/source bootstrap')
+        base=launch(protocol,pins,report['benchmark_v8_sha256'],label,Path(project['conversion_directory']),
+            profiler=label=='vllm' and mode=='observed')
+        # Receipt's mounted artifact path is authenticated by actual exec SHA/build
+        # below; all other launch bytes must be exact derivation of immutable V8.
+        artifact=Path(next(receipt['argv'][i+1].removesuffix(':/collector-artifact:ro')
+            for i,v in enumerate(receipt['argv'][:-1]) if v=='-v' and receipt['argv'][i+1].endswith(':/collector-artifact:ro'))) if label=='zerogravity' else HERE
+        if label=='zerogravity':
+            build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+            require(sha(artifact/'build.json')==report['collector_build_sha256']
+                    and build['collector_sha256']==report['collector_sha256'], 'independent actual ROOT build binding')
+        command=derive(base,label,mode,args.own_freeze,artifact)
+        replacements={'{owned_name}':receipt['owned_name'],'{owned_nonce}':receipt['owned_nonce'],
+            '{cidfile}':str(directory/'cid'),'{sample_directory}':str(directory)}
+        for key,value in replacements.items(): command=[part.replace(key,value) for part in command]
+        require(receipt['argv']==command,'every actual launch argument equals reviewed exact canonical derivative')
+        workload(sample['observation'],protocol,report['concurrency'])
+        require(read(directory/'measurement.json')==sample['observation'],'actual original measured HTTP observations')
+        if protocol['policy']=='cold_process':
+            require(sample['warmup'] is None and receipt['warmup'] is None,'no collector-only warmup in cold_process')
+        else:
+            require(sample['warmup']==receipt['warmup'] and sample['warmup']['completed'] is True
+                    and sample['warmup']['tokens']>0,'actual exact canonical warmed_process request')
+        if label=='zerogravity':
+            actual=read(directory/'native-exec-attestation.json')
+            require(actual['mode']==mode and actual['argv']==['/runtime/zerogravity']+bootstrap['engine_argv']
+                    and actual['own_freeze_sha256']==args.own_freeze,'same original final exec and own closure')
+            if mode=='observed': native_summary=native_activity(directory,receipt,parent,report)
+        elif mode=='observed':
+            for key,route in (('profile_start','/start_profile'),('profile_stop','/stop_profile')):
+                require(sample[key]['route']==route and sample[key]['HTTP_status']==200,'actual original public profiler controls')
+            # Reparse complete original trace after owner is gone, not stored report count.
+            class Owner: pass
+            owner=Owner(); owner.directory=directory
+            stock_summary=stock_traces(owner,audit)
+            require(stock_summary==sample['actual_recorded_CUDA_activity'],'independent entire actual stock trace replay')
+    for label in ('zerogravity','vllm'):
+        pair=[item for item in report['samples'] if item['provider']==label]
+        require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),'exact all wire token/text/finish/usage transparency')
+    require(native_summary is not None and stock_summary,'both actual canonical workload activity scopes present')
+    result=dict(status='PASS_CONTAINER_RECORDED_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY',
+        own_freeze_sha256=args.own_freeze,report_sha256=sha(output/'report.json'),
+        native=native_summary,stock_recorded=stock_summary,artifact_bounds_before_validation_receipt=tree_bound(output),
+        same_canonical_workload_policy_and_caps_checked=True,plain_observed_wire_outputs_exact=True,
+        source_profiles_numerically_equal=False,source_correct_goodput_qualified=False,
+        matched_complete_hardware_kernel_counts_qualified=False,stock_collector_losslessness_verified=False,
+        timing_usable=False,device_profiler_allocation_admission_qualified=False)
+    frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
+    result['artifact_finalization_followup_sha256']=verify_finalization()
+    result['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
+    write_completion(output,'validation.json',result)
+    verify_mapping_preparation(args.mapping_freeze)
+
+
+if __name__=='__main__': main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/verified_fd_mapping_v3.py
@@ -0,0 +1,273 @@
+"""Readonly, same-open-FD mapping identity. ROOT execution only; no dlopen.
+
+st_dev and the kernel VMA device are separate observations. A unique temporary
+private readonly page supplies the latter; no filesystem/device exceptions.
+"""
+import hashlib
+import mmap
+import os
+from pathlib import Path
+import re
+import stat
+
+from mapping_contract_v3 import require
+from container_contracts import FILE_LIMIT
+
+LOCAL_MAP_BYTES = 128 * 1024
+LOCAL_MAP_ROWS = 2048
+TARGET_MAP_BYTES = 2 * 1024**2
+HASH_CHUNK = 64 * 1024
+U64_MAX = (1 << 64) - 1
+
+
+def file_identity(info):
+    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
+                mode=info.st_mode, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
+
+
+def valid_identity(value):
+    keys = {'device', 'inode', 'bytes', 'mode', 'mtime_ns', 'ctime_ns'}
+    require(type(value) is dict and set(value) == keys and
+            all(type(value[k]) is int and value[k] >= 0 for k in keys) and
+            value['inode'] > 0 and 0 < value['bytes'] <= FILE_LIMIT and
+            stat.S_ISREG(value['mode']), 'bounded exact regular-file stat identity')
+
+
+def maps_text(path, cap):
+    with Path(path).open(encoding='utf-8', errors='strict') as stream:
+        raw = stream.read(cap + 1)
+    require(len(raw.encode()) <= cap, 'bounded actual kernel maps metadata')
+    return raw
+
+
+def parse_maps(raw, cap=LOCAL_MAP_BYTES, row_limit=LOCAL_MAP_ROWS):
+    require(type(raw) is str and 0 < len(raw.encode()) <= cap and '\x00' not in raw,
+            'complete bounded kernel maps text')
+    lines = raw.splitlines()
+    require(0 < len(lines) <= row_limit and raw.endswith('\n'), 'complete bounded maps rows')
+    result = []
+    previous = 0
+    for line in lines:
+        require(0 < len(line) <= 8192, 'bounded complete kernel maps line')
+        parts = line.split(None, 5)
+        require(len(parts) in (5, 6) and re.fullmatch(r'[0-9a-f]+-[0-9a-f]+', parts[0]) and
+                re.fullmatch(r'[r-][w-][x-][ps]', parts[1]) and
+                re.fullmatch(r'[0-9a-f]+', parts[2]) and
+                re.fullmatch(r'[0-9a-f]+:[0-9a-f]+', parts[3]) and
+                re.fullmatch(r'[0-9]+', parts[4]), 'exact Linux proc maps fields')
+        start, end = (int(v, 16) for v in parts[0].split('-'))
+        major, minor = (int(v, 16) for v in parts[3].split(':'))
+        offset, inode = int(parts[2], 16), int(parts[4])
+        require(previous <= start < end <= U64_MAX and 0 <= offset <= U64_MAX and
+                0 <= inode <= U64_MAX and 0 <= major <= 0xffffffff and 0 <= minor <= 0xffffffff,
+                'ordered nonoverlapping checked mapping geometry')
+        device = os.makedev(major, minor)
+        result.append(dict(line=line, start=start, end=end, permissions=parts[1], offset=offset,
+                           major=major, minor=minor, device=device, inode=inode,
+                           path=parts[5] if len(parts) == 6 else ''))
+        previous = end
+    return result
+
+
+def kernel_path(path):
+    require(type(path) is str and path.startswith('/') and len(path.encode()) <= 8192 and
+            not path.endswith(' (deleted)') and not any(ord(v) < 32 for v in path),
+            'exact live absolute retained-FD kernel pathname')
+    return path
+
+
+def key(row):
+    return dict(device=row['device'], major=row['major'], minor=row['minor'], inode=row['inode'])
+
+
+def valid_key(value):
+    require(type(value) is dict and set(value) == {'device', 'major', 'minor', 'inode'} and
+            all(type(v) is int for v in value.values()) and
+            0 <= value['major'] <= 0xffffffff and 0 <= value['minor'] <= 0xffffffff and
+            0 < value['inode'] <= U64_MAX and value['device'] == os.makedev(value['major'], value['minor']),
+            'exact typed observed kernel mapping key')
+
+
+def local_witness(before, during, after, path, identity, page_size):
+    valid_identity(identity)
+    kernel_path(path)
+    require(type(page_size) is int and 0 < page_size <= 1024**2 and page_size & (page_size - 1) == 0,
+            'actual bounded system page size')
+    old, live, retired = (parse_maps(value) for value in (before, during, after))
+    candidates = []
+    for row in live:
+        if (row['path'] == path and row['inode'] == identity['inode'] and
+                row['permissions'] == 'r--p' and row['offset'] == 0 and
+                row['end'] - row['start'] == page_size and row['start'] % page_size == 0 and
+                all(row['end'] <= prior['start'] or prior['end'] <= row['start'] for prior in old)):
+            candidates.append(row)
+    require(len(candidates) == 1, 'one newly introduced same-FD readonly page VMA; coalescing refuses')
+    selected = candidates[0]
+    require(not any(item['device'] == selected['device'] and item['inode'] == selected['inode'] and
+                    item['end'] > selected['start'] and item['start'] < selected['end']
+                    for item in retired), 'temporary same-FD mapping is absent after explicit close')
+    return selected
+
+
+def target_rows(raw, path, mapping_key, size, page_size):
+    kernel_path(path)
+    valid_key(mapping_key)
+    rows = [row for row in parse_maps(raw, TARGET_MAP_BYTES, 32768) if row['path'] == path]
+    require(rows and any('x' in row['permissions'] for row in rows),
+            'actual executable/collector/CUPTI file mappings present')
+    for row in rows:
+        require(key(row) == mapping_key and row['offset'] < size and
+                row['end'] - row['start'] <= ((size - row['offset'] + page_size - 1) // page_size) * page_size,
+                'every target VMA matches derived same-FD kernel key and bounded original file extent')
+    return [row['line'] for row in rows]
+
+
+def hash_fd(fd, expected, identity):
+    valid_identity(identity)
+    require(re.fullmatch(r'[0-9a-f]{64}', expected) is not None, 'exact original artifact SHA256')
+    require(file_identity(os.fstat(fd)) == identity, 'same retained FD before full hash')
+    digest = hashlib.sha256()
+    offset = 0
+    while offset < identity['bytes']:
+        raw = os.pread(fd, min(HASH_CHUNK, identity['bytes'] - offset), offset)
+        require(raw, 'complete exact same-FD artifact read, no short EOF')
+        digest.update(raw)
+        offset += len(raw)
+    require(not os.pread(fd, 1, offset) and file_identity(os.fstat(fd)) == identity and
+            digest.hexdigest() == expected, 'full immutable opened-FD artifact SHA/stat, no trailing bytes')
+    return digest.hexdigest()
+
+
+def verify_opened(path, expected):
+    """Return metadata after the sole source FD and temporary readonly map retire."""
+    opened_path = Path(path)
+    proof = dict(schema='zg.cupti.verified-fd-mapping.v3',
+                 opened_path=str(opened_path), hash_chunk_bytes=HASH_CHUNK,
+                 no_executable_mapping=True, readonly_private_offset_zero=True,
+                 verified_fd_closed=False, temporary_mapping_closed=False)
+    fd = os.open(opened_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
+    try:
+        identity = file_identity(os.fstat(fd))
+        valid_identity(identity)
+        proof['stat_before'] = identity
+        require(file_identity(os.stat(opened_path)) == identity, 'original path still denotes the retained FD')
+        descriptor_path = kernel_path(os.readlink('/proc/self/fd/' + str(fd)))
+        proof['descriptor_kernel_path_before'] = descriptor_path
+        proof['sha256_before'] = hash_fd(fd, expected, identity)
+        proof['stat_before_mapping'] = file_identity(os.fstat(fd))
+        page = mmap.PAGESIZE
+        require(page == os.sysconf('SC_PAGE_SIZE') and 0 < page <= 1024**2,
+                'same actual system mmap page size')
+        proof['page_size'] = page
+        proof['local_maps_before'] = maps_text('/proc/self/maps', LOCAL_MAP_BYTES)
+        # This exact retained FD, never a reopened pathname or executable map.
+        with mmap.mmap(fd, min(page, identity['bytes']), flags=mmap.MAP_PRIVATE,
+                       prot=mmap.PROT_READ, offset=0):
+            proof['local_maps_during'] = maps_text('/proc/self/maps', LOCAL_MAP_BYTES)
+        proof['temporary_mapping_closed'] = True
+        proof['local_maps_after'] = maps_text('/proc/self/maps', LOCAL_MAP_BYTES)
+        proof['stat_after_mapping'] = file_identity(os.fstat(fd))
+        witness = local_witness(proof['local_maps_before'], proof['local_maps_during'],
+                                proof['local_maps_after'], descriptor_path, identity, page)
+        proof['readonly_mapping_line'] = witness['line']
+        proof['kernel_mapping_key'] = key(witness)
+        proof['sha256_after'] = hash_fd(fd, expected, identity)
+        proof['stat_after'] = file_identity(os.fstat(fd))
+        proof['path_stat_after'] = file_identity(os.stat(opened_path))
+        proof['descriptor_kernel_path_after'] = kernel_path(os.readlink('/proc/self/fd/' + str(fd)))
+        require(proof['descriptor_kernel_path_after'] == descriptor_path and
+                proof['path_stat_after'] == identity, 'same opened FD/path through full mapping proof')
+    finally:
+        os.close(fd)
+    proof['verified_fd_closed'] = True
+    return proof
+
+
+def validate_proof(proof, expected, opened_path):
+    require(proof['schema'] == 'zg.cupti.verified-fd-mapping.v3' and
+            proof['opened_path'] == opened_path and proof['hash_chunk_bytes'] == HASH_CHUNK and
+            proof['no_executable_mapping'] is True and proof['readonly_private_offset_zero'] is True and
+            proof['verified_fd_closed'] is True and proof['temporary_mapping_closed'] is True,
+            'complete exact readonly same-FD proof and owner retirement')
+    identity = proof['stat_before']
+    valid_identity(identity)
+    require(all(proof[name] == identity for name in
+                ('stat_before_mapping', 'stat_after_mapping', 'stat_after', 'path_stat_after')) and
+            proof['sha256_before'] == proof['sha256_after'] == expected and
+            proof['descriptor_kernel_path_before'] == proof['descriptor_kernel_path_after'],
+            'all actual opened-FD stats/hashes/path identities stable')
+    witness = local_witness(proof['local_maps_before'], proof['local_maps_during'],
+                            proof['local_maps_after'], proof['descriptor_kernel_path_before'],
+                            identity, proof['page_size'])
+    valid_key(proof['kernel_mapping_key'])
+    require(proof['readonly_mapping_line'] == witness['line'] and
+            proof['kernel_mapping_key'] == key(witness), 'independent same-FD kernel key derivation')
+    return identity, key(witness)
+
+
+def executable_binding(pid, path, expected, identity):
+    magic = '/proc/' + str(pid) + '/exe'
+    require(os.readlink(magic) == path, 'actual original native executable pathname')
+    # /proc/PID/exe is the one explicit kernel magic link; never a caller path.
+    fd = os.open(magic, os.O_RDONLY | os.O_CLOEXEC)
+    try:
+        stat_before = file_identity(os.fstat(fd))
+        require(stat_before == identity, 'actual native exe is the same verified file')
+        before = hash_fd(fd, expected, identity)
+        after = hash_fd(fd, expected, identity)
+        stat_after = file_identity(os.fstat(fd))
+        require(os.readlink(magic) == path and stat_after == identity,
+                'same native executable through complete same-FD hashes')
+    finally:
+        os.close(fd)
+    return dict(path=magic, executable_link=path, stat_before=stat_before, stat_after=stat_after,
+                sha256_before=before, sha256_after=after, verified_fd_closed=True)
+
+
+def mapped_files(pid, exec_attestation, parent, live):
+    root = Path('/proc') / str(pid)
+    expected = {'native': '/runtime/zerogravity', 'collector': '/collector-artifact/collector.so',
+                'cupti': parent['cupti_library']['container_path']}
+    output = {}
+    for name, path in expected.items():
+        live()
+        before = maps_text(root/'maps', TARGET_MAP_BYTES)
+        actual = root/'root'/path.lstrip('/')
+        digest = exec_attestation['native_sha256' if name == 'native' else name+'_sha256']
+        proof = verify_opened(actual, digest)
+        identity, mapping_key = validate_proof(proof, digest, str(actual))
+        after = maps_text(root/'maps', TARGET_MAP_BYTES)
+        rows = target_rows(before, path, mapping_key, identity['bytes'], proof['page_size'])
+        rows_after = target_rows(after, path, mapping_key, identity['bytes'], proof['page_size'])
+        require(rows == rows_after,
+                'all target artifact VMAs stable across full same-FD proof')
+        live()
+        output[name] = dict(container_path=path, bytes=identity['bytes'], sha256=digest,
+                            inode=identity['inode'], device=identity['device'],
+                            stat_device=identity['device'], mapping_device=mapping_key['device'],
+                            kernel_mapping_key=mapping_key, mappings=rows, mappings_after=rows_after,
+                            fd_mapping=proof)
+        if name == 'native':
+            output[name]['exe_binding'] = executable_binding(pid, path, digest, identity)
+            live()
+    return output
+
+
+def validate_mapped_file(value, expected, pid, path):
+    require(value['container_path'] == path, 'exact original container artifact path')
+    identity, mapping_key = validate_proof(value['fd_mapping'], expected,
+                                         '/proc/' + str(pid) + '/root' + path)
+    require(value['sha256'] == expected and value['bytes'] == identity['bytes'] and
+            value['inode'] == identity['inode'] and value['device'] == value['stat_device'] == identity['device'] and
+            value['mapping_device'] == mapping_key['device'] and value['kernel_mapping_key'] == mapping_key and
+            value['mappings'] == value['mappings_after'], 'stat and kernel devices retained separately, never substituted')
+    raw = '\n'.join(value['mappings']) + '\n'
+    require(target_rows(raw, path, mapping_key, identity['bytes'], value['fd_mapping']['page_size']) == value['mappings'],
+            'independent all-target VMA key/extent replay')
+    if path == '/runtime/zerogravity':
+        exe = value['exe_binding']
+        require(exe['path'] == '/proc/' + str(pid) + '/exe' and exe['executable_link'] == path and
+                exe['stat_before'] == exe['stat_after'] == identity and
+                exe['sha256_before'] == exe['sha256_after'] == expected and exe['verified_fd_closed'] is True,
+                'independent complete same-native-executable FD/path/hash/stat binding')
+    return identity, mapping_key
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/freeze-manifest.json
@@ -0,0 +1,72 @@
+{
+  "author_tests_mmap_or_provider_model_DSO_build_GPU_runtime_or_payload_hash": false,
+  "expected_source_files": 8,
+  "files": [
+    {
+      "bytes": 6258,
+      "file": "README.md",
+      "sha256": "05b5a2b700fc6190e2a4e9b0c7623cd27ed788d00aa99700a35c38404f421669"
+    },
+    {
+      "bytes": 3654,
+      "file": "mapping_contract_v3.py",
+      "sha256": "53317e5465c007a9345e7a86846f40b24b20ded45269b2b8fa935aac41a1f70f"
+    },
+    {
+      "bytes": 6100,
+      "file": "namespace_control_v3.py",
+      "sha256": "3cef212940bc6c57824f23346d3ad734ee6f9fd1b8ed11386bbd92cad7384f79"
+    },
+    {
+      "bytes": 14638,
+      "file": "run_container_v3.py",
+      "sha256": "e2c78f169b143db1bb7cd0b5545782e8dc98f8a1467e9479ba69c4d971a3811f"
+    },
+    {
+      "bytes": 8250,
+      "file": "source-pins.json",
+      "sha256": "69871be05bf7a095f6480c99618ea7e83b3e99e6657de78e6101967abda70cab"
+    },
+    {
+      "bytes": 10949,
+      "file": "test_fd_mapping_v3.py",
+      "sha256": "3396148c8555d71cca325d69b5137e6d98d41332a8c2a351d201f931104e70ae"
+    },
+    {
+      "bytes": 16819,
+      "file": "validate_container_v3.py",
+      "sha256": "45741dc644139831af7f0f6cab7a210633e74bddb4c7bb411966d41b4f796f04"
+    },
+    {
+      "bytes": 14554,
+      "file": "verified_fd_mapping_v3.py",
+      "sha256": "24aca7d4297a8697b153d15fbb59c4a07b8bf9e0f33fce7a35caf5d41fe4311e"
+    }
+  ],
+  "local_maps_rows": 2048,
+  "local_maps_snapshot_bytes": 131072,
+  "mapped_FD_full_SHA_fstat_path_before_after_and_readonly_page_witness": true,
+  "math_queue_kernel_counts_or_GPU_runtime_qualified": false,
+  "max_controller_receipt_bytes": 4194304,
+  "native_executable_magic_link_hash_identity_also_checked": true,
+  "no_hardcoded_device_or_filesystem_exceptions": true,
+  "original_freeze_sha256": "07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee",
+  "original_policy_workload_loss_namespace_owner_and_finalization_unchanged": true,
+  "prepared_source_free_fixture_methods": 15,
+  "review_artifact": {
+    "bytes": 11156,
+    "file": "REVIEW.diff",
+    "sha256": "1198260b9f23d5e5309aeff8f152e4a6a3552351555b479e11492c495d45034b"
+  },
+  "same_FD_hash_chunk_bytes": 65536,
+  "schema": "zg.cupti.same-open-FD-mapping.preparation.v3",
+  "source_only_checks": [
+    "AST6 only; no proposed-module imports",
+    "original complete identity/live/peer/close/stock/workload/driver functions AST-identical",
+    "all29 old source and diagnostic metadata pins exact",
+    "old container/capacity/finalization source bytes remain unchanged"
+  ],
+  "stat_device_and_same_open_FD_kernel_mapping_device_retained_separately": true,
+  "status": "SOURCE_ONLY_UNEXECUTED_REQUIRES_ROOT_AND_PEER_REVIEW",
+  "unchanged_source_and_metadata_pins": 29
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/REVIEW.diff
@@ -0,0 +1,207 @@
+--- original/namespace_control.py
++++ additive/namespace_control_v3.py
+@@ -12,7 +12,9 @@
+ import socket
+ import struct
+ 
++from mapping_contract_v3 import verify_preparation
+ from container_contracts import read, require, sha
++from verified_fd_mapping_v3 import mapped_files
+ 
+ 
+ def text(path,limit):
+@@ -54,37 +56,14 @@
+         namespace_inodes={name:os.readlink(root/'ns'/name) for name in ('pid','net','user','mnt')})
+ 
+ 
+-def mapped_files(pid,exec_attestation,parent):
+-    root=Path('/proc')/str(pid)
+-    lines=text(root/'maps',2*1024**2).splitlines()
+-    expected={'native':'/runtime/zerogravity','collector':'/collector-artifact/collector.so',
+-              'cupti':parent['cupti_library']['container_path']}
+-    output={}
+-    for key,path in expected.items():
+-        actual=root/'root'/path.lstrip('/')
+-        info=actual.stat()
+-        rows=[]
+-        for line in lines:
+-            parts=line.split(None,5)
+-            if len(parts)==6 and parts[5]==path:
+-                major,minor=(int(v,16) for v in parts[3].split(':'))
+-                require(int(parts[4])==info.st_ino and os.makedev(major,minor)==info.st_dev,
+-                        'actual mapped inode/dev matches exact original file')
+-                rows.append(line)
+-        require(rows and actual.is_file(),'all declared executable/collector/CUPTI mappings present')
+-        digest=sha(actual)
+-        expected_sha=exec_attestation['native_sha256' if key=='native' else key+'_sha256']
+-        require(digest==expected_sha,'full actual mapped artifact hash: '+key)
+-        output[key]=dict(container_path=path,bytes=info.st_size,sha256=digest,
+-            inode=info.st_ino,device=info.st_dev,mappings=rows)
+-    return output
+ 
+ 
+ class Controller:
+-    def __init__(self,owner,owned,parent):
++    def __init__(self,owner,owned,parent,mapping_freeze):
+         self.owner,self.owned,self.parent=owner,owned,parent
+         self.fd=None
+-        self.receipt=dict(schema='zg.cupti.container.controller.v1',events=[],pidfd_closed=False)
++        self.receipt=dict(schema='zg.cupti.container.controller.v3',events=[],pidfd_closed=False,
++            fd_mapping_followup_sha256=verify_preparation(mapping_freeze))
+         try:
+             self.binding=inspect(owner,owned)
+             self.pid=self.binding['host_init_pid']
+@@ -101,7 +80,7 @@
+                     'actual host-network namespace shared; no setns/privilege assumption')
+             self.nonce=actual['nonce']
+             self.receipt.update(binding=self.binding,host_before=self.before,exec_attestation=actual,
+-                mapped_files=mapped_files(self.pid,actual,parent),
++                mapped_files=mapped_files(self.pid,actual,parent,self.live),
+                 host_controller_uid=os.geteuid(),reported_namespace_pid=1,
+                 driver_process_id_policy='preserve raw CUPTI IDs; require namespace PID1 independently')
+         except BaseException:
+--- original/run_container_v2.py
++++ additive/run_container_v3.py
+@@ -15,10 +15,12 @@
+ import time
+ import urllib.request
+ 
++from mapping_contract_v3 import (verify_preparation as verify_mapping_preparation,
++    bounded_controller_metadata)
+ from container_contracts import (HERE,TREE_LIMIT,build_binding,fingerprint,frozen,host_parents,
+     read,require,sha,tree_bound,write)
+ from launch_derivative import derive
+-from namespace_control import Controller
++from namespace_control_v3 import Controller
+ from completion_v2 import verify_preparation as verify_finalization, write_completion
+ 
+ 
+@@ -66,12 +68,14 @@
+     parser=argparse.ArgumentParser(description=__doc__)
+     parser.add_argument('--run',action='store_true')
+     parser.add_argument('--own-freeze',required=True)
++    parser.add_argument('--mapping-freeze',required=True)
+     parser.add_argument('--launch-directory',type=Path,required=True)
+     parser.add_argument('--collector-artifact-directory',type=Path,required=True)
+     parser.add_argument('--concurrency',type=int,choices=(1,2,4,8),required=True)
+     parser.add_argument('--output',type=Path,required=True)
+     args=parser.parse_args()
+     require(args.run,'ROOT reviewed explicit actual execution only')
++    verify_mapping_preparation(args.mapping_freeze)
+     verify_finalization()
+     parent=frozen(args.own_freeze); host_parents(parent)
+     require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
+@@ -127,6 +131,7 @@
+         source_correct_goodput_qualified=False,numerical_profiles_identical=False,
+         native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+     report['artifact_finalization_followup_sha256']=verify_finalization()
++    report['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
+     allowed=os.sched_getaffinity(0)
+     require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
+     urllib.request.install_opener(opener())
+@@ -149,7 +154,7 @@
+                 try:
+                     if mode=='observed':
+                         if label=='zerogravity':
+-                            controller=Controller(owner,owned,parent)
++                            controller=Controller(owner,owned,parent,args.mapping_freeze)
+                             start=controller.command('START')
+                             started=True
+                             write(owner.directory/'window-start.json',start)
+@@ -177,7 +182,7 @@
+                     if controller is not None:
+                         try: controller.close()
+                         except BaseException as error: cleanup.append('controller close: '+repr(error))
+-                        try: write(owner.directory/'namespace-controller.json',controller.receipt)
++                        try: write(owner.directory/'namespace-controller.json',bounded_controller_metadata(controller.receipt))
+                         except BaseException as error: cleanup.append('controller receipt: '+repr(error))
+                     require(not cleanup and primary is None,
+                             'diagnostic primary='+repr(primary)+'; collector_cleanup='+repr(cleanup))
+@@ -209,6 +214,7 @@
+         except BaseException as error: failures.append('lifecycle digest: '+repr(error))
+         try:
+             frozen(args.own_freeze); host_parents(parent); verify_preparation(v8); verify_small_sources()
++            verify_mapping_preparation(args.mapping_freeze)
+         except BaseException as error: failures.append('post-run immutable source closure: '+repr(error))
+         try:
+             source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+@@ -221,6 +227,7 @@
+         report['finalization_errors']=failures
+         if failures: report['status']='UNQUALIFIED'
+         write_completion(output,'report.json',report)
++        verify_mapping_preparation(args.mapping_freeze)
+         if failures:
+             raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
+ 
+--- original/validate_container_v2.py
++++ additive/validate_container_v3.py
+@@ -8,10 +8,12 @@
+ from pathlib import Path
+ import sys
+ 
++from mapping_contract_v3 import verify_preparation as verify_mapping_preparation
+ from capacity_v1_loader import activity
+ from container_activity import summarize
+ from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
+ from launch_derivative import derive
++from verified_fd_mapping_v3 import validate_mapped_file
+ from completion_v2 import verify_preparation as verify_finalization, write_completion
+ 
+ 
+@@ -66,6 +68,9 @@
+ def native_activity(directory,receipt,parent,report):
+     control=read(directory/'namespace-controller.json')
+     actual=read(directory/'native-exec-attestation.json')
++    require(control['schema']=='zg.cupti.container.controller.v3' and
++            control['fd_mapping_followup_sha256']==report['fd_mapping_followup_sha256'],
++            'actual reviewed additive same-FD mapping controller')
+     require(control['pidfd_closed'] is True and control['reported_namespace_pid']==1
+             and control['exec_attestation']==actual and actual['nonce']==receipt['owned_nonce']
+             and control['binding']['owned_nonce']==receipt['owned_nonce']
+@@ -78,6 +83,9 @@
+     for key,field in (('native','native_sha256'),('collector','collector_sha256'),('cupti','cupti_sha256')):
+         require(control['mapped_files'][key]['sha256']==actual[field] and control['mapped_files'][key]['mappings'],
+                 'actual complete mapped artifact identity retained')
++        path={'native':'/runtime/zerogravity','collector':'/collector-artifact/collector.so',
++              'cupti':parent['cupti_library']['container_path']}[key]
++        validate_mapped_file(control['mapped_files'][key],actual[field],hp,path)
+     require(actual['collector_on_outer_python'] is False and actual['collector_requested_only_at_this_exec'] is True
+             and actual['native_sha256']==report['protocol']['zerogravity_binary_sha256']
+             and actual['collector_sha256']==report['collector_sha256']
+@@ -116,13 +124,17 @@
+ def main():
+     parser=argparse.ArgumentParser()
+     parser.add_argument('--own-freeze',required=True)
++    parser.add_argument('--mapping-freeze',required=True)
+     parser.add_argument('--output',type=Path,required=True)
+     args=parser.parse_args()
++    verify_mapping_preparation(args.mapping_freeze)
+     verify_finalization()
+     parent=frozen(args.own_freeze); host_parents(parent)
+     output=args.output.resolve(strict=True)
+     require(output.parent==HERE,'owned diagnostic output')
+     report=read(output/'report.json')
++    require(report['fd_mapping_followup_sha256']==verify_mapping_preparation(args.mapping_freeze),
++            'actual report uses exact reviewed same-FD mapping source closure')
+     require(report.get('artifact_finalization_followup_sha256')==verify_finalization(),
+             'actual report used reviewed final receipt reservation')
+     require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+@@ -138,7 +150,7 @@
+     from run_diagnostic_v8 import post
+     from audit_cuda_trace_v8 import audit
+     from cgroup_readback_v3 import validate_caps
+-    from run_container import stock_traces
++    from run_container_v3 import stock_traces
+     pins=verify_preparation(report['benchmark_v8_sha256']); verify_small_sources()
+     protocol=validate_protocol(report['protocol'])
+     protocol_path=Path(report['protocol_path'])
+@@ -214,7 +226,9 @@
+         timing_usable=False,device_profiler_allocation_admission_qualified=False)
+     frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
+     result['artifact_finalization_followup_sha256']=verify_finalization()
++    result['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
+     write_completion(output,'validation.json',result)
++    verify_mapping_preparation(args.mapping_freeze)
+ 
+ 
+ if __name__=='__main__': main()

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/NEW_DELTA.patch

Original UTF-8 bytes: 87141; SHA256 `b1ef8cfc926533454ad254a3ad5aa5a235f199868182f2474bdfbc2e8ef1f28f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/README.md
@@ -0,0 +1,38 @@
+# CUPTI safe-signal owner checkpoints — additive V4
+
+- Parent serving-and-speculation; ROOT-only source proposal. Preserve V3 `14bdbe64`, its eight sources, all earlier freezes/attempts and original V8 Plan/Provider +53ee cleanup byte-identically. V3 still has the inherited asynchronous SIGTERM entrypoint and must not be executed as the repaired launcher.
+- This derivative changes only host cancellation/source binding and adds control fixtures. Same V3 FD/mapping code, collector, providers, argv, policy, workloads, generation, counters, capacities and numerical checks. No MAIN changes.
+- Copy `SigtermCancellation`1428B exact `00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128`. TERM/INT handlers record a flag; never asynchronously raise during Popen return/assignment, monitor/log construction, original cleanup, signals or reaps.
+
+## Safe checkpoints
+
+- `checked_sample` checks before constructing an owner, inside a wrapped preflight after original Provider retained Popen/monitor/log, after readiness/yield, and only after original Plan.sample finally finishes retirement. Existing Provider.start catches Exception to retry readiness; synchronous pending-cancellation delivery is KeyboardInterrupt, which escapes that retry handler into the unchanged BaseException cleanup boundary.
+- Check around actual measurement and before the next provider. A recorded signal during an original bounded HTTP/readiness/idle operation is delivered at the next checkpoint; original transport timeouts remain unchanged. No per-token callbacks, global Popen monkeypatch, alternate owner or executor.
+- No checkpoints run inside cleanup. Repeated signals only record state; original TERM/grace/two-KILL-before-reap, independent reap, nonce/CID rediscovery, logs/EOF and GPU retirement complete. Existing primary/cleanup errors retain precedence. A recorded signal marks the diagnostic UNQUALIFIED; V4 validation requires false recorded state and exact guard/source metadata.
+- Original complete owner/identity/loss/workload/transparency/caps/mapping/finalization checks remain. New mandatory `--signal-freeze` binds ten V4 source/metadata files, exact guard and immutable V3 closure before/after execution/final writes.
+
+## ROOT sequence, after active timing
+
+1. Review ten files and REVIEW.diff. Author performed AST/source-byte checks only; no tests/import execution/mmap/process/container/provider/model/GPU/build or payload hashing.
+2. From this directory, ROOT may run `python3 -m unittest -v test_signal_v4.py` (three pure mock protocol methods). They establish only checkpoint/exception/cleanup ordering, no physical lifetime evidence.
+3. ROOT may run the nine real harmless cases through the existing private rootless user/network wrapper:
+
+```sh
+python3 run_harmless_v4.py --run-control-only \
+  --signal-freeze EXACT_V4_FREEZE_SHA256 --new-output FRESH_CHILD_OF_THIS_DIRECTORY
+```
+
+   These use the **actual original V8 Plan.sample/Provider and real53ee nonce/name/CID Podman cleanup**, exact existing pinned image, foreground CLI, bounded log and original EOF/TERM/KILL/reap/absence stages. No mocked container discovery or cleanup. Only GPU quiet/monitor observers are explicit stubs; their receipts cannot pass real diagnostic validation.
+
+   Cases: TERM/INT before Popen (no child), at retained-handoff/readiness, during an actually open harmless POST, at the observer stage of original Provider.close; plus postspawn/final receipt-I/O failure. Cleanup-phase signals are injected only by the allowed GPU-observer stub; original53ee cleanup is neither wrapped nor mocked. The stdlib payload is an exact existing copy: no device/model/source/native-binary mount, CUDA/NVIDIA disabled,256MiB/1CPU configured controls, pullnever/read-only/capdrop/no-new-privileges. `/v1/models` always503; no token bytes, model/native readiness, generation or GPU proof. No-body-yield assertion prevents accepting the control as a real provider. Each non-prelaunch case must have actually observed the genuine nonce/CID and actual control health before its injected event. Even the expected I/O-negative must retain real cleanup evidence separately and leave the original Plan blocked on missing final evidence.
+
+   The nine-case report is **control owner evidence only**, not provider/cgroup/math/hardware/throughput qualification. The first two cases intentionally have no child; the remaining seven must retire actual named containers, groups and pipes. Native/provider readiness must stay false in every case. No actual GPU probe is performed.
+4. After all reviews/fixtures and ROOT's quiet-resource authorization, run the same V3 diagnostic commands using `run_container_v4.py`/`validate_container_v4.py` and add `--signal-freeze EXACT_V4_FREEZE_SHA256`. Keep `--own-freeze07a86bf6…` and `--mapping-freeze14bdbe64…` unchanged. Use fresh original container-v1 outputs and genuine existing capacity collector/build.
+
+Acceptance remains diagnostic-only. No timing, complete matched hardware counts, stock-profiler losslessness, numerical equivalence or device-profiler memory admission follows from this signal fix or the control fixtures.
+
+## Owned continuation
+
+- [x] Preserve V3 and accepted guard; prepare signal-only V4 plus three mocks/nine harmless real cases.
+- [ ] ROOT + independent peer full source review.
+- [ ] ROOT actual mock/control fixtures after timing, then fresh four-owner diagnostic and independent validator.
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/harmless_http_v2.py
@@ -0,0 +1,58 @@
+"""Explicit CPU-only cancellation target. NOT a model/native serving engine."""
+import argparse
+from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
+import json
+import signal
+import threading
+import time
+
+
+class Handler(BaseHTTPRequestHandler):
+    def do_GET(self):
+        value = dict(control_only=True, model_or_GPU_loaded=False)
+        if self.path == '/phase':
+            value['post_open'] = self.server.post_open.is_set()
+        raw = json.dumps(value).encode()
+        # Original native readiness must NOT accept this target.
+        self.send_response(503 if self.path == '/v1/models' else 200)
+        self.send_header('Content-Length', str(len(raw)))
+        self.end_headers()
+        self.wfile.write(raw)
+
+    def do_POST(self):
+        length = int(self.headers.get('Content-Length', '0'))
+        if not 0 < length <= 4096:
+            self.send_error(400)
+            return
+        self.rfile.read(length)
+        self.send_response(200)
+        self.send_header('Content-Type', 'text/event-stream')
+        self.end_headers()
+        self.wfile.flush()
+        self.server.post_open.set()
+        print('CONTROL_POST_OPEN', flush=True)
+        # No IDs/output/model arithmetic. Parent uses short canonical timeout;
+        # an independently retained owner kills this harmless target afterward.
+        time.sleep(20)
+
+    def log_message(self, *_):
+        pass
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument('--port', type=int, required=True)
+    args = parser.parse_args()
+    if not 1024 <= args.port <= 65535:
+        raise SystemExit('bounded control port')
+    # Exercise original owned TERM->KILL path. Only this fresh owned container
+    # is affected; this process has no GPU device/model/source mounts.
+    signal.signal(signal.SIGTERM, signal.SIG_IGN)
+    print('HARMLESS_CONTROL_READY_NO_MODEL_NO_GPU', flush=True)
+    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
+    server.post_open = threading.Event()
+    server.serve_forever()
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/run_container_v4.py
@@ -0,0 +1,255 @@
+"""ROOT ONLY four fresh diagnostic processes; no canonical timing samples.
+
+Native plain/CUPTI observed, stock plain/public Torch profiler observed. Each
+uses exact V8 requests/cold-warm policy/CPU/RAM/engine limits and original owner.
+"""
+import argparse
+from dataclasses import asdict
+import gzip
+import hashlib
+import json
+import os
+from pathlib import Path
+import signal
+import sys
+import time
+import urllib.request
+
+from signal_contract_v4 import verify_preparation as verify_signal_preparation
+from signal_owner_v4 import checked_sample, checkpoint, signal_metadata
+from sigterm_cancellation import SigtermCancellation
+from mapping_contract_v3 import (verify_preparation as verify_mapping_preparation,
+    bounded_controller_metadata)
+from container_contracts import (HERE,TREE_LIMIT,build_binding,fingerprint,frozen,host_parents,
+    read,require,sha,tree_bound,write)
+from launch_derivative import derive
+from namespace_control_v3 import Controller
+from completion_v2 import verify_preparation as verify_finalization, write_completion
+
+
+def idle(bench,url):
+    deadline=time.monotonic()+30
+    previous=None
+    while time.monotonic()<deadline:
+        raw=bench.snapshot(url+'/health',None)
+        require(raw['ok'],'actual original native health')
+        state=raw['value']['admission']
+        require(state['kind']=='resident_gpu' and state['failed_gpu_quanta']==0,'healthy original GPU engine')
+        if state['active_sessions']==state['queued_requests']==state['output_credit_parked_requests']==0:
+            if previous==raw['value']: return raw
+            previous=raw['value']
+        else: previous=None
+        time.sleep(.1)
+    raise TimeoutError('native physical work did not become terminal before window boundary')
+
+
+def stock_traces(owner,audit):
+    output=[]
+    directory=owner.directory/'traces'
+    require(directory.is_dir() and not directory.is_symlink(),'actual original public profiler output')
+    paths=sorted(directory.rglob('*'))
+    require(len(paths)<=32,'bounded actual profiler artifact count')
+    for path in paths:
+        require(not path.is_symlink(),'owned original profiler files')
+        if path.is_dir(): continue
+        require(path.is_file() and path.stat().st_size<=256*1024**2,'bounded public profiler artifact')
+        if not path.name.endswith(('.pt.trace.json','.pt.trace.json.gz')): continue
+        with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
+            raw=stream.read(256*1024**2+1)
+        require(len(raw)<=256*1024**2,'bounded entire decoded public trace')
+        value=json.loads(raw)
+        if any(event.get('cat')=='kernel' for event in value.get('traceEvents',[])):
+            result=audit(value)
+            result.update(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
+                decoded_sha256=hashlib.sha256(raw).hexdigest())
+            output.append(result)
+    require(len(output)==1,'one actual TP1 CUDA worker trace, no CPU-only substitution')
+    return output
+
+
+def main(cancellation):
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run',action='store_true')
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--mapping-freeze',required=True)
+    parser.add_argument('--signal-freeze',required=True)
+    parser.add_argument('--launch-directory',type=Path,required=True)
+    parser.add_argument('--collector-artifact-directory',type=Path,required=True)
+    parser.add_argument('--concurrency',type=int,choices=(1,2,4,8),required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    require(args.run,'ROOT reviewed explicit actual execution only')
+    verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+    verify_mapping_preparation(args.mapping_freeze)
+    verify_finalization()
+    parent=frozen(args.own_freeze); host_parents(parent)
+    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
+    benchmark=Path(parent['parents']['benchmark'])
+    sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_compare_v8 import lifecycle_spec,load_tool,opener
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from support_v3 import source_check
+    from owned_plan_v8 import install
+    import owned_container_v4 as owned
+    v8=parent['benchmark_v8_manifest']['sha256']
+    pins=verify_preparation(v8); verify_small_sources()
+    require(sha(owned.__file__)==parent['owned_helper_sha256'],'actual original53ee lifecycle helper import')
+    launch_dir=args.launch_directory.resolve(strict=True)
+    require(launch_dir.parent==benchmark and not args.launch_directory.is_symlink(),'original V8 owned launch directory')
+    protocol=validate_protocol(read(launch_dir/'protocol.json'))
+    require(args.concurrency in protocol['concurrencies'] and protocol['source_sha256']==pins['models'][protocol['model']]['gguf']['sha256']
+            and protocol['tokenizer_sha256']==pins['tokenizer_sha256'],'exact V8 workload/concurrency/source identity')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    conversion=Path(project['conversion_directory'])
+    commands={label:launch(protocol,pins,v8,label,conversion) for label in ('zerogravity','vllm')}
+    require(read(launch_dir/'launches.json')==commands,'exact original canonical engine commands')
+    diagnostics=read(launch_dir/'diagnostic-launches.json')
+    stock_observed=launch(protocol,pins,v8,'vllm',conversion,profiler=True)
+    require(diagnostics['cuda_activity']==stock_observed,'exact existing public stock profiler command')
+    source=pins['models'][protocol['model']]
+    source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+    require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+            and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256'],
+            'actual unchanged projector/native executable before diagnostic')
+    artifact=args.collector_artifact_directory.resolve(strict=True)
+    require(not args.collector_artifact_directory.is_symlink(),'original ROOT artifact directory')
+    build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+    output=args.output.resolve()
+    require(output.parent==HERE and not output.exists(),'fresh exclusive owned diagnostic output')
+    output.mkdir(mode=0o700)
+    compare,bench=load_tool('serve-compare.py'),load_tool('serve-bench.py')
+    install(compare.LIFECYCLE)
+    urls={label:f'http://127.0.0.1:{port}' for label,port in zip(('zerogravity','vllm'),protocol['ports'])}
+    spec=lifecycle_spec(protocol,commands)
+    plan=compare.LIFECYCLE['LifecyclePlan'](spec,urls,str(output/'owned'))
+    alias='qwen35-'+protocol['model']+'-same-gguf'
+    report=dict(status='UNQUALIFIED',schema='zg.cupti.container.diagnostic.v1',own_freeze_sha256=args.own_freeze,
+        benchmark_v8_sha256=v8,capacity_freeze_sha256=parent['capacity_manifest']['sha256'],
+        protocol=protocol,protocol_path=str(launch_dir/'protocol.json'),
+        protocol_sha256=sha(launch_dir/'protocol.json'),concurrency=args.concurrency,
+        collector_build_sha256=sha(artifact/'build.json'),collector_sha256=build['collector_sha256'],
+        owned_cleanup_source_sha256=parent['owned_helper_sha256'],samples=[],
+        performance_timing_usable=False,matched_hardware_counts_qualified=False,
+        source_correct_goodput_qualified=False,numerical_profiles_identical=False,
+        native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+    report['artifact_finalization_followup_sha256']=verify_finalization()
+    report['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
+    report['signal_followup_sha256']=verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+    allowed=os.sched_getaffinity(0)
+    require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
+    urllib.request.install_opener(opener())
+    try:
+        os.sched_setaffinity(0,protocol['cpu_affinity'])
+        for label,mode in (('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed')):
+            base=stock_observed if label=='vllm' and mode=='observed' else commands[label]
+            spec_command=derive(base,label,mode,args.own_freeze,artifact)
+            plan.spec['providers'][label]['argv']=spec_command
+            sample=dict(provider=label,mode=mode,policy=protocol['policy'],warmup=None)
+            with checked_sample(plan,label,lambda:compare.preflight(urls[label],alias,None),cancellation) as owner:
+                if protocol['policy']=='warmed_process':
+                    warm=bench.stream_one(urls[label],alias,protocol['requests'][0]['prompt'],
+                        protocol['max_tokens'],180,None,request_id='matched-diag-warmup',return_token_ids=True)
+                    owner.record_warmup(asdict(warm)); sample['warmup']=asdict(warm)
+                    require(warm.completed,'exact canonical unmeasured warmed_process request')
+                checkpoint(cancellation)
+                before=idle(bench,urls[label]) if label=='zerogravity' else None
+                if before is not None: write(owner.directory/'health-before.json',before)
+                controller=None; started=False; primary=None; cleanup=[]
+                try:
+                    if mode=='observed':
+                        if label=='zerogravity':
+                            controller=Controller(owner,owned,parent,args.mapping_freeze)
+                            checkpoint(cancellation)
+                            start=controller.command('START')
+                            started=True
+                            write(owner.directory/'window-start.json',start)
+                        else:
+                            sample['profile_start']=post(urls[label],'/start_profile')
+                            started=True
+                    checkpoint(cancellation)
+                    measured=bench.bench(urls[label],alias,protocol['requests'][0]['prompt'],len(protocol['requests']),
+                        args.concurrency,protocol['max_tokens'],180,None,workload=protocol['requests'],
+                        return_token_ids=True,max_pending=protocol['max_pending'],
+                        launch_deadline_seconds=protocol['launch_deadline_seconds'],run_id='matched-cuda-activity')
+                    fingerprint(measured); owner.record_measurement(measured)
+                    sample['observation']=measured
+                    checkpoint(cancellation)
+                    after=idle(bench,urls[label]) if label=='zerogravity' else None
+                    if after is not None: write(owner.directory/'health-after.json',after)
+                except BaseException as error:
+                    primary=error
+                finally:
+                    if started:
+                        try:
+                            if controller is not None:
+                                stop=controller.command('STOP')
+                                write(owner.directory/'window-stop.json',stop)
+                            else: sample['profile_stop']=post(urls[label],'/stop_profile')
+                        except BaseException as error: cleanup.append('collector stop: '+repr(error))
+                    if controller is not None:
+                        try: controller.close()
+                        except BaseException as error: cleanup.append('controller close: '+repr(error))
+                        try: write(owner.directory/'namespace-controller.json',bounded_controller_metadata(controller.receipt))
+                        except BaseException as error: cleanup.append('controller receipt: '+repr(error))
+                    require(not cleanup and primary is None,
+                            'diagnostic primary='+repr(primary)+'; collector_cleanup='+repr(cleanup))
+            # All original owner/process/group/container/monitor/pipe cleanup finished.
+            require(not plan.blocked,'complete original retirement before next provider')
+            sample['owned_receipt_path']=str(owner.receipt_path)
+            if mode=='observed' and label=='vllm': sample['actual_recorded_CUDA_activity']=stock_traces(owner,audit)
+            report['samples'].append(sample)
+            tree_bound(owner.directory,768*1024**2)
+            tree_bound(output)
+            checkpoint(cancellation)
+        for label in ('zerogravity','vllm'):
+            pair=[item for item in report['samples'] if item['provider']==label]
+            require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),
+                    'complete plain/observed wire IDs/text/finish/usage exact for '+label)
+        checkpoint(cancellation)
+        report['output_transparency_exact']=True
+        report['status']='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        report['failure']=repr(error)[:4096]
+        raise
+    finally:
+        # All process/container owners already retired; independent evidence stages
+        # must still run if affinity restoration or a post-run source scan fails.
+        failures=[]
+        try: os.sched_setaffinity(0,allowed)
+        except BaseException as error: failures.append('restore client affinity: '+repr(error))
+        report['lifecycle_manifest']=str(plan.directory/'lifecycle.json')
+        try:
+            report['lifecycle_manifest_sha256']=sha(plan.directory/'lifecycle.json')
+        except BaseException as error: failures.append('lifecycle digest: '+repr(error))
+        try:
+            frozen(args.own_freeze); host_parents(parent); verify_preparation(v8); verify_small_sources()
+            verify_mapping_preparation(args.mapping_freeze)
+            verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+        except BaseException as error: failures.append('post-run immutable source closure: '+repr(error))
+        try:
+            source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+            require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+                    and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256']
+                    and sha(artifact/'collector.so')==build['collector_sha256']
+                    and sha(artifact/'build.json')==report['collector_build_sha256'],
+                    'unchanged actual model/projector/binary/collector/build after retirement')
+        except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
+        report['finalization_errors']=failures
+        report['signal_cancellation']=signal_metadata(cancellation)
+        if failures or cancellation.received: report['status']='UNQUALIFIED'
+        write_completion(output,'report.json',report)
+        verify_mapping_preparation(args.mapping_freeze)
+        verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+        if failures:
+            raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
+        if 'failure' not in report:
+            checkpoint(cancellation)
+
+
+
+if __name__=='__main__':
+    with SigtermCancellation() as cancellation:
+        main(cancellation)
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/run_harmless_v4.py
@@ -0,0 +1,269 @@
+"""ROOT ONLY nine real nonce/CID harmless owner cases; no model or GPU proof."""
+import argparse
+from contextlib import ExitStack
+import fcntl
+import json
+import os
+from pathlib import Path
+import signal
+import socket
+import struct
+import sys
+import urllib.error
+import urllib.request
+from unittest.mock import patch
+
+from signal_contract_v4 import HERE, MAPPING, verify_preparation
+from signal_owner_v4 import checked_sample, signal_metadata
+from sigterm_cancellation import SigtermCancellation
+from container_contracts import read, require, sha, tree_bound, write
+
+
+def private_namespace():
+    # Exact existing read-only ROOT namespace checks; no setns/network change.
+    mapping = Path('/proc/self/uid_map').read_text().split()
+    require(mapping == ['0', '1000', '1', '1', '100000', '65536'],
+            'exact existing ROOT-less Podman UID namespace required')
+    require(os.environ.get('_CONTAINERS_USERNS_CONFIGURED') == 'done' and
+            os.environ.get('_CONTAINERS_ROOTLESS_UID') == '1000',
+            'original rootless Podman namespace bootstrap required')
+    network = os.open('/proc/self/ns/net', os.O_RDONLY)
+    try:
+        owner = fcntl.ioctl(network, 0xB701)
+        try:
+            require(os.fstat(owner).st_ino == os.stat('/proc/self/ns/user').st_ino,
+                    'private network must be owned by current rootless user namespace')
+        finally:
+            os.close(owner)
+    finally:
+        os.close(network)
+    require(socket.if_nameindex() == [(1, 'lo')], 'fresh network must contain only private loopback')
+    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
+        flags = fcntl.ioctl(control, 0x8913, struct.pack('16sH22x', b'lo', 0))
+        require(struct.unpack_from('H', flags, 16)[0] & 1, 'use exact reviewed wrapper2 loopback setup first')
+    return dict(uid_map=Path('/proc/self/uid_map').read_text(),
+        user_namespace=os.readlink('/proc/self/ns/user'), network_namespace=os.readlink('/proc/self/ns/net'),
+        mount_namespace=os.readlink('/proc/self/ns/mnt'),
+        rootless_userns_configured=os.environ['_CONTAINERS_USERNS_CONFIGURED'],
+        rootless_uid=os.environ['_CONTAINERS_ROOTLESS_UID'])
+
+
+def no_gpu(*_):
+    return [dict(memory_used_mib=0, control_only=True, GPU_observation_performed=False)]
+
+
+class NoGpuMonitor:
+    def __init__(self, *_):
+        pass
+
+    def finish(self):
+        return dict(control_only=True, GPU_observation_performed=False, samples=[])
+
+
+class ControlEnd(BaseException):
+    pass
+
+
+class ControlIo(BaseException):
+    pass
+
+
+def http(opener, url):
+    with opener.open(url, timeout=2) as response:
+        raw = response.read(4097)
+        require(response.status == 200 and len(raw) <= 4096, 'bounded actual harmless HTTP response')
+        return json.loads(raw)
+
+
+def dependencies():
+    parent = read(HERE.parent/'parent-bindings.json')
+    benchmark = Path(parent['parents']['benchmark'])
+    sys.path.insert(0, str(benchmark))
+    from run_compare_v8 import load_tool
+    from owned_plan_v8 import install
+    import owned_plan_v8 as adapter
+    import owned_container_v4 as owned
+    import observations_v8 as observations
+    compare = load_tool('serve-compare.py')
+    install(compare.LIFECYCLE)
+    return benchmark, compare, adapter, owned, observations
+
+
+def specification(owned, image, port):
+    command = owned.PODMAN + ['run', '--name', '{owned_name}', '--label', owned.LABEL+'={owned_nonce}',
+        '--cidfile', '{cidfile}', '--rm', '--pull=never', '--network=host', '--read-only',
+        '--cap-drop=all', '--security-opt=no-new-privileges', '--memory=256m', '--memory-swap=256m',
+        '--cpus=1', '--pids-limit=32', '--ulimit', 'fsize=4194304:4194304',
+        '--tmpfs', '/tmp:rw,nosuid,nodev,size=16m', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
+        '-e', 'CUDA_VISIBLE_DEVICES=-1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
+        '-v', str(HERE/'harmless_http_v2.py')+':/control.py:ro',
+        '--entrypoint', 'python3', image, '/control.py', '--port', str(port)]
+    declarations = {name:'HARMLESS_CONTROL_NOT_PROVIDER_QUALIFICATION' for name in
+        ('runtime','numerical_profile','quantization','generation_policy','hardware')}
+    declarations.update(source_sha256='0'*64, tokenizer_sha256='0'*64,
+        resource_limits=dict(actual_gpu_total_mib=0, GPU_observer_is_explicit_stub=True))
+    provider = dict(argv=command, cwd=str(HERE), env=dict(ZG_EXPECTED_GPU_UUID='NO_GPU_CONTROL_FIXTURE'),
+                    stop_argv=owned.PODMAN+['stop','--ignore','--time','1','{owned_cid}'], declarations=declarations)
+    # Lifecycle requires these two keys; only the explicitly marked control
+    # slot is ever started. No real engine/readiness/output is impersonated.
+    return dict(schema='zerogravity.serving.provider-lifecycle.v1',policy='cold_process',
+                startup_timeout_seconds=60,shutdown_timeout_seconds=5,log_limit_bytes=4*1024**2,
+                providers=dict(zerogravity=provider,vllm=provider))
+
+
+def case(root, index, phase, signum, image, deps):
+    _, compare, adapter, owned, observations = deps
+    port = 18570 + index
+    url = 'http://127.0.0.1:'+str(port)
+    directory = root/f'{index:02d}-{phase}-{signum or 0}'
+    directory.mkdir(mode=0o700)
+    plan = compare.LIFECYCLE['LifecyclePlan'](specification(owned,image,port),
+        dict(zerogravity=url,vllm='http://127.0.0.1:'+str(port+100)),str(directory))
+    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
+    observed = {}
+    class PhaseMonitor(NoGpuMonitor):
+        def finish(self):
+            # Only the allowed GPU-observer stub injects a signal, at its real
+            # original Provider.close stage.53ee cleanup is never wrapped.
+            if phase == 'cleanup':
+                observed['signal_at_original_close'] = True
+                os.kill(os.getpid(),signum)
+            return super().finish()
+
+    def preflight():
+        provider = plan.current
+        require(provider is not None and provider.process is not None,
+                'actual original Provider retained Popen before callback')
+        observed['provider'] = provider
+        cid = owned.owned_container(provider.name,provider.nonce,provider.cidfile)
+        require(cid is not None, 'genuine nonce/name/CID-bound harmless container, never mocked')
+        require(http(opener,url+'/health') == dict(control_only=True,model_or_GPU_loaded=False),
+                'actual harmless payload, not native or stock readiness')
+        try:
+            opener.open(url+'/v1/models',timeout=2)
+        except urllib.error.HTTPError as error:
+            require(error.code == 503, 'native/models readiness is always503')
+            error.close()
+        else:
+            raise ValueError('harmless payload must never advertise native model readiness')
+        observed['actual_owned_cid'] = cid
+        if phase == 'readiness':
+            os.kill(os.getpid(),signum)
+            return  # checked_preflight raises synchronous KeyboardInterrupt.
+        if phase == 'request':
+            request = urllib.request.Request(url+'/v1/chat/completions',data=b'{"control_only":true}',
+                                             headers={'Content-Type':'application/json'})
+            with opener.open(request,timeout=1) as response:
+                require(response.status == 200 and http(opener,url+'/phase') ==
+                        dict(control_only=True,model_or_GPU_loaded=False,post_open=True),
+                        'actual open harmless POST before cancellation')
+                observed['actual_post_open'] = True
+                os.kill(os.getpid(),signum)
+                try:
+                    raw = response.read(1)
+                    require(raw == b'', 'harmless fixture must emit no token/content bytes')
+                except (TimeoutError,OSError):
+                    pass
+            return
+        if phase == 'cleanup':
+            raise ControlEnd('enter original retirement without accepting model readiness')
+        require(phase == 'receipt_io', 'closed harmless fixture phase')
+        def failing_save():
+            raise OSError('injected actual postspawn and final receipt I/O failure')
+        provider.save = failing_save
+        try:
+            provider.save()
+        except OSError as error:
+            raise ControlIo('actual postspawn evidence failure') from error
+
+    failure = None
+    with SigtermCancellation() as cancellation, ExitStack() as patches:
+        # These are the ONLY runtime-observer stubs. Container discovery,
+        # cleanup, Popen/log/Plan/Provider and original53ee functions stay real.
+        patches.enter_context(patch.object(observations,'quiet',side_effect=no_gpu))
+        patches.enter_context(patch.object(observations,'Monitor',PhaseMonitor))
+        if phase == 'before_popen':
+            os.kill(os.getpid(),signum)
+        try:
+            with checked_sample(plan,'zerogravity',preflight,cancellation):
+                raise AssertionError('harmless target must never reach native/model readiness')
+        except BaseException as error:
+            failure = error
+        provider = observed.get('provider')
+        if phase == 'before_popen':
+            require(isinstance(failure,KeyboardInterrupt) and provider is None and
+                    plan.receipts == [] and plan.current is None, 'cancel before any owner/Popen construction')
+        else:
+            require(provider is not None and observed.get('actual_owned_cid') and
+                    provider.receipt.get('ready') is not True and plan.current is None,
+                    'real harmless container observed; native/model readiness never accepted')
+            receipt = provider.receipt
+            foreground,log = receipt['foreground_cleanup'],receipt['log_cleanup']
+            require(receipt['cleanup_errors'] == [] and receipt['owned_container_absent_after_reap'] is True and
+                    all(foreground[k] is True for k in ('leader_reaped','group_absent','all_nonzero_signals_before_reap')) and
+                    foreground['stage_errors'] == [] and foreground['kill_attempts_before_reap'] == 2 and
+                    log['reader_complete'] is True and log['reader_error'] is None and log['truncated'] is False and
+                    owned.owned_container(provider.name,provider.nonce,provider.cidfile) is None,
+                    'actual original container/foreground/EOF retired with real discovery, no stubs')
+            if phase == 'receipt_io':
+                require(isinstance(failure,RuntimeError) and 'ControlIo' in str(failure) and
+                        'final evidence I/O failed AFTER original owner cleanup' in str(failure) and
+                        plan.blocked is True, 'primary and cleanup evidence failures remain visible and block next provider')
+            elif phase == 'cleanup':
+                require(isinstance(failure,ControlEnd) and cancellation.received and not plan.blocked and
+                        observed.get('signal_at_original_close') is True,
+                        'recorded signal cannot interrupt original cleanup or replace primary')
+            else:
+                require(isinstance(failure,KeyboardInterrupt) and cancellation.received and not plan.blocked,
+                        'safe synchronous readiness/request cancellation after real retained handoff')
+                if phase == 'request':
+                    require(observed.get('actual_post_open') is True, 'signal requires actual open POST evidence')
+        result = dict(status='PASS_HARMLESS_ORIGINAL_PLAN_OWNER_ONLY',phase=phase,signal=signum,
+            expected_failure_type=type(failure).__name__,failure=repr(failure),real_owned_child=provider is not None,
+            signal_cancellation=signal_metadata(cancellation),native_models_readiness_accepted=False,
+            actual_owned_cid=observed.get('actual_owned_cid'),actual_open_POST=observed.get('actual_post_open',False),
+            signal_at_original_close_observer_stage=observed.get('signal_at_original_close',False),
+            actual_receipt_path=str(provider.receipt_path) if provider else None,
+            receipt_snapshot_after_original_retirement=provider.receipt if provider else None,
+            primary_and_cleanup_evidence_failure_not_waived=phase=='receipt_io',
+            GPU_observer_stub=True,container_identity_and_cleanup_stubbed=False,
+            provider_model_GPU_math_or_benchmark_qualified=False)
+        write(directory/'control-validation.json',result)
+        return result
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run-control-only',action='store_true')
+    parser.add_argument('--signal-freeze',required=True)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args = parser.parse_args()
+    require(args.run_control_only, 'ROOT explicitly authorizes harmless actual owners after source review')
+    source = verify_preparation(args.signal_freeze,MAPPING)
+    namespace = private_namespace()
+    pins = read(HERE/'source-pins.json')
+    deps = dependencies()
+    root = args.new_output.resolve()
+    require(root.parent == HERE and not root.exists(), 'fresh exclusive harmless output')
+    root.mkdir(mode=0o700)
+    report = dict(status='UNQUALIFIED',signal_freeze_sha256=source,cases=[],namespace=namespace,
+                  original_V8_Plan_Provider_53ee_cleanup=True,GPU_observer_stub=True,
+                  container_identity_or_cleanup_stubbed=False,provider_model_GPU_math_or_benchmark_qualified=False)
+    try:
+        cases = [(phase,sig) for phase in ('before_popen','readiness','request','cleanup')
+                 for sig in (signal.SIGTERM,signal.SIGINT)] + [('receipt_io',None)]
+        for index,(phase,sig) in enumerate(cases):
+            report['cases'].append(case(root,index,phase,sig,pins['harmless_image'],deps))
+        require(len(report['cases']) == 9 and verify_preparation(args.signal_freeze,MAPPING) == source,
+                'all nine actual harmless cases and unchanged sources')
+        report['status'] = 'PASS_NINE_HARMLESS_OWNER_CASES_NOT_PROVIDER_OR_GPU_QUALIFICATION'
+    except BaseException as error:
+        report['failure'] = repr(error)[:4096]
+        raise
+    finally:
+        write(root/'control-report.json',report)
+        tree_bound(root)
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/signal_contract_v4.py
@@ -0,0 +1,38 @@
+"""Additive V4 source binding, with immutable V3 mapping and original owners."""
+import hashlib
+import json
+from pathlib import Path
+import sys
+
+HERE = Path(__file__).resolve().parent
+V3 = HERE.parent/'fd-mapping-v3'
+if str(V3) not in sys.path:
+    sys.path.insert(0, str(V3))
+from mapping_contract_v3 import pinned, unique, require, verify_preparation as verify_mapping
+
+MAPPING = '14bdbe64496a25cecac4f8bd7144ec61c8e160c127c9c3273c0d066e14946a71'
+GUARD = '00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128'
+
+
+def verify_preparation(expected, mapping):
+    require(mapping == MAPPING, 'exact original reviewed V3 mapping source, no alteration')
+    path = HERE/'freeze-manifest.json'
+    raw = pinned(path, dict(bytes=path.stat().st_size, sha256=expected))
+    value = json.loads(raw, object_pairs_hook=unique)
+    require(value['mapping_freeze_sha256'] == MAPPING and value['guard_source_sha256'] == GUARD and
+            value['math_queue_provider_or_GPU_qualified'] is False, 'signal-only source scope')
+    require(len(value['files']) == 10 and len({row['file'] for row in value['files']}) == 10,
+            'complete ten-file source closure')
+    for row in value['files']:
+        require(row['file'] == Path(row['file']).name, 'finite V4 local filename')
+        pinned(HERE/row['file'], row)
+        pinned(HERE/'frozen'/row['file'], row)
+    guard = next(row for row in value['files'] if row['file'] == 'sigterm_cancellation.py')
+    require(guard['sha256'] == GUARD, 'exact accepted record-only TERM/INT guard bytes')
+    pins_row = next(row for row in value['files'] if row['file'] == 'source-pins.json')
+    pins = json.loads(pinned(HERE/'source-pins.json', pins_row), object_pairs_hook=unique)
+    for row in pins['original_source_metadata']:
+        pinned(row['path'], row)
+    pinned(HERE/'REVIEW.diff', value['review_artifact'])
+    verify_mapping(MAPPING)
+    return expected
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/signal_owner_v4.py
@@ -0,0 +1,38 @@
+"""Synchronous cancellation checkpoints; original Plan/Provider owns retirement."""
+from contextlib import contextmanager
+
+
+def checkpoint(cancellation):
+    try:
+        cancellation.check()
+    except InterruptedError as error:
+        # Provider.start catches Exception for retryable readiness failures.
+        # This synchronous BaseException must escape to Plan.sample's finally.
+        raise KeyboardInterrupt(str(error)) from error
+
+
+@contextmanager
+def checked_sample(plan, label, preflight, cancellation):
+    checkpoint(cancellation)  # No owner/Popen has been constructed yet.
+
+    def checked_preflight():
+        # Original Provider.start already retained Popen/monitor/log here.
+        checkpoint(cancellation)
+        result = preflight()
+        checkpoint(cancellation)
+        return result
+
+    with plan.sample(label, checked_preflight) as owner:
+        checkpoint(cancellation)
+        yield owner
+        checkpoint(cancellation)
+    # No checkpoint runs inside original cleanup, even under repeated signals.
+    # A preexisting primary/cleanup exception propagates without replacement.
+    checkpoint(cancellation)
+
+
+def signal_metadata(cancellation):
+    return dict(recorded=bool(cancellation.received), TERM_INT_record_only=True,
+                asynchronous_raise=False, safe_checkpoint_exception='KeyboardInterrupt',
+                original_Plan_Provider_and_53ee_cleanup_unchanged=True,
+                guard_source_sha256='00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128')
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/sigterm_cancellation.py
@@ -0,0 +1,47 @@
+"""SIGTERM/SIGINT checkpoints around the original bounded log drain, not new ownership."""
+import signal
+
+
+class SigtermCancellation:
+    def __init__(self):
+        self.received = False
+        self.previous = {}
+
+    def _record(self, signum, frame):
+        # Never asynchronously raise across Popen assignment or retirement.
+        self.received = True
+
+    def __enter__(self):
+        for signum in (signal.SIGTERM, signal.SIGINT):
+            self.previous[signum] = signal.signal(signum, self._record)
+        return self
+
+    def __exit__(self, kind, value, traceback):
+        for signum, previous in self.previous.items():
+            signal.signal(signum, previous)
+        return False
+
+    def exception(self):
+        return InterruptedError("interrupted: retire original owned serving regression before failure")
+
+    def check(self):
+        if self.received:
+            raise self.exception()
+
+    def observe(self, log, process, seconds):
+        original_drain = log.drain
+
+        def checked_drain(timeout):
+            self.check()
+            original_drain(timeout)
+            self.check()
+
+        self.check()
+        log.drain = checked_drain
+        try:
+            result = log.observe(process, seconds)
+            self.check()
+            return result
+        finally:
+            # Original cleanup/finish drain proceeds even after repeated SIGTERM.
+            log.drain = original_drain
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/source-pins.json
@@ -0,0 +1,130 @@
+{
+  "actual_model_DSO_provider_payload_read_or_hash_by_author": false,
+  "harmless_GPU_observer_explicit_stub_only": true,
+  "harmless_container_discovery_or_cleanup_stubbed": false,
+  "harmless_image": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "harmless_image_source": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/source-pins-v3.json",
+  "original_source_metadata": [
+    {
+      "bytes": 6258,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/README.md",
+      "sha256": "05b5a2b700fc6190e2a4e9b0c7623cd27ed788d00aa99700a35c38404f421669"
+    },
+    {
+      "bytes": 11156,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/REVIEW.diff",
+      "sha256": "1198260b9f23d5e5309aeff8f152e4a6a3552351555b479e11492c495d45034b"
+    },
+    {
+      "bytes": 2729,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/freeze-manifest.json",
+      "sha256": "14bdbe64496a25cecac4f8bd7144ec61c8e160c127c9c3273c0d066e14946a71"
+    },
+    {
+      "bytes": 3654,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/mapping_contract_v3.py",
+      "sha256": "53317e5465c007a9345e7a86846f40b24b20ded45269b2b8fa935aac41a1f70f"
+    },
+    {
+      "bytes": 6100,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/namespace_control_v3.py",
+      "sha256": "3cef212940bc6c57824f23346d3ad734ee6f9fd1b8ed11386bbd92cad7384f79"
+    },
+    {
+      "bytes": 14638,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/run_container_v3.py",
+      "sha256": "e2c78f169b143db1bb7cd0b5545782e8dc98f8a1467e9479ba69c4d971a3811f"
+    },
+    {
+      "bytes": 8250,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/source-pins.json",
+      "sha256": "69871be05bf7a095f6480c99618ea7e83b3e99e6657de78e6101967abda70cab"
+    },
+    {
+      "bytes": 10949,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/test_fd_mapping_v3.py",
+      "sha256": "3396148c8555d71cca325d69b5137e6d98d41332a8c2a351d201f931104e70ae"
+    },
+    {
+      "bytes": 16819,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/validate_container_v3.py",
+      "sha256": "45741dc644139831af7f0f6cab7a210633e74bddb4c7bb411966d41b4f796f04"
+    },
+    {
+      "bytes": 14554,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/verified_fd_mapping_v3.py",
+      "sha256": "24aca7d4297a8697b153d15fbb59c4a07b8bf9e0f33fce7a35caf5d41fe4311e"
+    },
+    {
+      "bytes": 1703,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/parent-bindings.json",
+      "sha256": "2a6204cca537c7d76d1b6536e3c357149bae21f8eae14ac83d83832c42999a66"
+    },
+    {
+      "bytes": 2072,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-native-before-after-20261020/harmless_http_v2.py",
+      "sha256": "99f3c2f41502c16b28153d0a5352dadf4297b76419796bea9921967c56d70c23"
+    },
+    {
+      "bytes": 13543,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-native-before-after-20261020/run_native_ab_v2.py",
+      "sha256": "5290daeab4d40e27cc2a9da4357ecf3e816fcf1aa391484714f0e2d76c78a954"
+    },
+    {
+      "bytes": 1428,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-native-before-after-20261020/sigterm_cancellation.py",
+      "sha256": "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128"
+    },
+    {
+      "bytes": 8934,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/bench-source-pins.json",
+      "sha256": "217ec76d82af88e2da4e05aad46ec42b771a04b0edaaab47f298b1c047757ce4"
+    },
+    {
+      "bytes": 5708,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/observations_v8.py",
+      "sha256": "23a783abbead3da819f9d0b046643010b669a6f14f2abb1e0752bd4ae35e4a9d"
+    },
+    {
+      "bytes": 11840,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_container_v4.py",
+      "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+    },
+    {
+      "bytes": 9685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_plan_v8.py",
+      "sha256": "346f8f98753018e6ddacaff291dec48def8ce024be6e9c4670c154fc028bedcb"
+    },
+    {
+      "bytes": 10387,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/protocol_v8.py",
+      "sha256": "61de8d581e9353c685bec518eab5f974df10de8baa57353e867606887b9ff98e"
+    },
+    {
+      "bytes": 9968,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/run_compare_v8.py",
+      "sha256": "fa08b592357217d58dd438014f6986976cb591e36fad8fd601c3fb83bf388adb"
+    },
+    {
+      "bytes": 58765,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/source-pins-v3.json",
+      "sha256": "09f74c3f792ed8a1d4e00ad727db3d8d39e3c0a16f3c7711d169ed466e98e238"
+    },
+    {
+      "bytes": 7068,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/support_v3.py",
+      "sha256": "1bde9fbe0137b5aa1b4f49c619d22003699985642092377294252bb80099885c"
+    },
+    {
+      "bytes": 23813,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/tools/serve-compare.py",
+      "sha256": "c2ab2c10d4a6c8506f8f07d47b7a509d4313211e9cd93465bf65cc56080fa878"
+    },
+    {
+      "bytes": 40831,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/tools/serve-lifecycle.py",
+      "sha256": "b3f9c76f872711edbe8eea5731f6f85c3ba119e86386b8dd50798b507fae9cd3"
+    }
+  ],
+  "schema": "zg.cupti.record-only-signals.sources.v4"
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/test_signal_v4.py
@@ -0,0 +1,81 @@
+"""ROOT-only mock protocol fixtures; no process/container/GPU proof."""
+from contextlib import contextmanager
+import signal
+import unittest
+
+from sigterm_cancellation import SigtermCancellation
+from signal_owner_v4 import checked_sample
+
+
+class Plan:
+    def __init__(self, cancellation, stage=None, cleanup_failure=False):
+        self.cancellation, self.stage = cancellation, stage
+        self.cleanup_failure = cleanup_failure
+        self.events = []
+        self.closed = False
+
+    @contextmanager
+    def sample(self, label, preflight):
+        self.events.append('retained owner/Popen handoff')
+        try:
+            if self.stage == 'handoff':
+                self.cancellation._record(signal.SIGTERM, None)
+            try:
+                preflight()
+            except Exception:
+                self.events.append('retryable Exception swallowed')
+            self.events.append('ready')
+            yield self
+        finally:
+            self.events.append('cleanup begin')
+            if self.stage == 'cleanup':
+                self.cancellation._record(signal.SIGINT, None)
+            self.events.extend(['TERM', 'KILL', 'reap', 'EOF'])
+            self.closed = True
+            if self.cleanup_failure:
+                raise RuntimeError('fixture original cleanup failure')
+
+
+class SignalProtocol(unittest.TestCase):
+    def test_prelaunch_and_post_handoff_pending_signal_never_retry_or_lose_owner(self):
+        cancellation = SigtermCancellation()
+        cancellation._record(signal.SIGTERM, None)
+        plan = Plan(cancellation)
+        with self.assertRaises(KeyboardInterrupt):
+            with checked_sample(plan, 'fixture', lambda: None, cancellation):
+                self.fail('prelaunch signal must not enter')
+        self.assertEqual(plan.events, [])
+        cancellation = SigtermCancellation()
+        plan = Plan(cancellation, stage='handoff')
+        with self.assertRaises(KeyboardInterrupt):
+            with checked_sample(plan, 'fixture', lambda: None, cancellation):
+                self.fail('handoff cancellation must escape except Exception')
+        self.assertTrue(plan.closed)
+        self.assertNotIn('retryable Exception swallowed', plan.events)
+        self.assertNotIn('ready', plan.events)
+        self.assertEqual(plan.events[-4:], ['TERM', 'KILL', 'reap', 'EOF'])
+
+    def test_body_or_cleanup_signal_raises_only_after_complete_original_cleanup(self):
+        for stage in ('body', 'cleanup'):
+            cancellation = SigtermCancellation()
+            plan = Plan(cancellation, stage=stage)
+            with self.subTest(stage=stage), self.assertRaises(KeyboardInterrupt):
+                with checked_sample(plan, 'fixture', lambda: None, cancellation):
+                    if stage == 'body':
+                        cancellation._record(signal.SIGTERM, None)
+            self.assertTrue(plan.closed)
+            self.assertEqual(plan.events[-4:], ['TERM', 'KILL', 'reap', 'EOF'])
+
+    def test_primary_and_cleanup_failures_keep_original_precedence(self):
+        cancellation = SigtermCancellation()
+        plan = Plan(cancellation, stage='cleanup', cleanup_failure=True)
+        with self.assertRaisesRegex(RuntimeError, 'original cleanup failure') as caught:
+            with checked_sample(plan, 'fixture', lambda: None, cancellation):
+                raise ValueError('fixture primary failure')
+        self.assertTrue(plan.closed)
+        self.assertIsInstance(caught.exception.__context__, ValueError)
+        self.assertEqual(str(caught.exception.__context__), 'fixture primary failure')
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/validate_container_v4.py
@@ -0,0 +1,247 @@
+"""ROOT-only independent complete-workload/namespace/activity/owner validator.
+
+Stock public Torch profiler loss completeness remains explicitly UNQUALIFIED;
+recorded kernel counts cannot establish source-correct goodput or timing speed.
+"""
+import argparse
+import hashlib
+from pathlib import Path
+import sys
+
+from signal_contract_v4 import verify_preparation as verify_signal_preparation
+from mapping_contract_v3 import verify_preparation as verify_mapping_preparation
+from capacity_v1_loader import activity
+from container_activity import summarize
+from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
+from launch_derivative import derive
+from verified_fd_mapping_v3 import validate_mapped_file
+from completion_v2 import verify_preparation as verify_finalization, write_completion
+
+
+def successful_owner(receipt,parent):
+    require(receipt['status']=='OWNERS_CLOSED' and receipt['cleanup_errors']==[]
+            and receipt['primary_failure'] is None and receipt['ready'] is True
+            and receipt['owned_container_absent_after_reap'] is True,'actual complete successful owned retirement')
+    cleanup=receipt['foreground_cleanup']
+    require(cleanup['pgid']==receipt['owned_cli_pid']>0 and cleanup['leader_reaped'] is True
+            and cleanup['group_absent'] is True and cleanup['all_nonzero_signals_before_reap'] is True
+            and cleanup['kill_attempts_before_reap']==2 and cleanup['leader_reap_attempts'] in (1,2)
+            and cleanup['stage_errors']==[],'all independent group signals before reap and final absence')
+    log=receipt['log_cleanup']
+    require(log['reader_complete'] is True and log['reader_error'] is None,'actual complete original log/pipe EOF')
+    observations=receipt['physical_observations']
+    require(observations['failures']==[] and observations['samples']
+            and observations['exact_peak'] is False
+            and observations['complete_physical_allocator_accounting'] is False,
+            'actual bounded original physical observations, no invented allocator peak')
+    for row in observations['samples']:
+        if row['compute_apps']:
+            require(all(app['pid'] in row['authenticated_container_host_pids'] for app in row['compute_apps']),
+                    'every actual GPU PID bound to original public container numeric HPIDs')
+    require(receipt['GPU_quiet_after'] and all(not row['compute_apps'] for row in receipt['GPU_quiet_after'][-3:]),
+            'actual final card quiet after original owner cleanup')
+
+
+def workload(result,protocol,concurrency):
+    require(result['offered_requests']==len(protocol['requests']) and result['concurrency']==concurrency
+            and result['max_pending']==protocol['max_pending']
+            and result['launch_deadline_seconds']==protocol['launch_deadline_seconds']
+            and result['run_id']=='matched-cuda-activity','exact canonical workload denominator and scheduling policy')
+    fingerprint(result)
+    require(len(result['requests'])==len(protocol['requests']),'all original canonical offers')
+    for index,(actual,expected) in enumerate(zip(result['requests'],protocol['requests'])):
+        require(actual['input_index']==index and actual['request_id']==expected['request_id']
+                and actual['prompt_sha256']==hashlib.sha256(expected['prompt'].encode()).hexdigest()
+                and actual['max_tokens']==expected['max_tokens'], 'actual full original prompt/max-token/request identity')
+
+
+def driver_identity(rows,host_pid):
+    ids=set()
+    for row in rows:
+        if row['kind']=='driver':
+            require(type(row['process']) is int,'actual raw driver process ID')
+            ids.add(row['process'])
+            require(len(ids)==1,'single native process driver identity; never merge namespaces')
+    require(len(ids)==1 and ids<={1,host_pid},'raw driver ID equals one authenticated process representation')
+    return next(iter(ids))
+
+
+def native_activity(directory,receipt,parent,report):
+    control=read(directory/'namespace-controller.json')
+    actual=read(directory/'native-exec-attestation.json')
+    require(control['schema']=='zg.cupti.container.controller.v3' and
+            control['fd_mapping_followup_sha256']==report['fd_mapping_followup_sha256'],
+            'actual reviewed additive same-FD mapping controller')
+    require(control['pidfd_closed'] is True and control['reported_namespace_pid']==1
+            and control['exec_attestation']==actual and actual['nonce']==receipt['owned_nonce']
+            and control['binding']['owned_nonce']==receipt['owned_nonce']
+            and control['binding']['owned_name']==receipt['owned_name'], 'original owner/exec/controller binding')
+    hp=control['binding']['host_init_pid']
+    require(hp>0 and actual['pid_namespace_pid']==1 and actual['namespace_inodes']==control['host_before']['namespace_inodes']
+            and control['host_before']['NSpid'][0]==hp and control['host_before']['NSpid'][-1]==1,
+            'host init PID and actual namespace PID preserved independently')
+    require(actual['file_size_limits']==[FILE_LIMIT,FILE_LIMIT], 'actual finite diagnostic file-size readback')
+    for key,field in (('native','native_sha256'),('collector','collector_sha256'),('cupti','cupti_sha256')):
+        require(control['mapped_files'][key]['sha256']==actual[field] and control['mapped_files'][key]['mappings'],
+                'actual complete mapped artifact identity retained')
+        path={'native':'/runtime/zerogravity','collector':'/collector-artifact/collector.so',
+              'cupti':parent['cupti_library']['container_path']}[key]
+        validate_mapped_file(control['mapped_files'][key],actual[field],hp,path)
+    require(actual['collector_on_outer_python'] is False and actual['collector_requested_only_at_this_exec'] is True
+            and actual['native_sha256']==report['protocol']['zerogravity_binary_sha256']
+            and actual['collector_sha256']==report['collector_sha256']
+            and actual['collector_capacity_freeze_sha256']==report['capacity_freeze_sha256']
+            and actual['cupti_sha256']==parent['cupti_library']['sha256'],'only final-exec exact artifact/provider closure')
+    require(len(control['events'])==2,'one complete original native activity window')
+    start,stop=read(directory/'window-start.json'),read(directory/'window-stop.json')
+    for event,verb,value in zip(control['events'],('START','STOP'),(start,stop)):
+        require(event['verb']==verb and event['reply']==value and event['host_peer_pid']==hp
+                and event['host_peer_uid']==control['host_controller_uid'] and event['reported_namespace_pid']==1,
+                'actual host SO_PEERCRED authenticated independently from CUPTI namespace identity')
+    require(start['nonce']==stop['nonce']==receipt['owned_nonce']
+            and start['begin']==stop['begin'] and start['end']==0
+            and start['start_monotonic']==stop['start_monotonic']<stop['stop_monotonic'], 'one actual timestamped window')
+    path=directory/'observed-activity/activity.jsonl'
+    require(read(directory/'observed-activity/collector-stop.json')==stop,'persisted stop exactly acknowledged')
+    digest=sha(path)
+    driver_pid=driver_identity(activity.rows(path,stop['bytes']),hp)
+    driver_ids={driver_pid}
+    result=summarize(activity.rows(path,stop['bytes']),stop,1,driver_pid)
+    require(sha(path)==digest,'immutable entire raw trace across independent complete passes')
+    before=read(directory/'health-before.json')['value']['admission']
+    after=read(directory/'health-after.json')['value']['admission']
+    for value in (before,after):
+        require(value['active_sessions']==value['queued_requests']==value['output_credit_parked_requests']==0
+                and value['failed_gpu_quanta']==0,'actual terminal original GPU work at window boundaries')
+    require(after['accepted_output_tokens']>before['accepted_output_tokens'],'actual canonical output within window')
+    result.update(raw_activity_sha256=digest,raw_bytes=path.stat().st_size,namespace_pid=1,
+        authenticated_init_host_pid=hp,actual_raw_driver_process_ids=sorted(driver_ids),
+        driver_process_representation='namespace' if driver_pid==1 else 'host',
+        source_dispatch_delta=after['completed_forward_kernel_dispatches']-before['completed_forward_kernel_dispatches'],
+        source_dispatches_are_not_driver_measurements=True,all_retained_records_lossless=True)
+    return result
+
+
+def main():
+    parser=argparse.ArgumentParser()
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--mapping-freeze',required=True)
+    parser.add_argument('--signal-freeze',required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+    verify_mapping_preparation(args.mapping_freeze)
+    verify_finalization()
+    parent=frozen(args.own_freeze); host_parents(parent)
+    output=args.output.resolve(strict=True)
+    require(output.parent==HERE,'owned diagnostic output')
+    report=read(output/'report.json')
+    require(report['signal_followup_sha256']==verify_signal_preparation(args.signal_freeze,args.mapping_freeze),
+            'actual exact V4 record-only safe-signal source closure')
+    require(report['signal_cancellation']==dict(recorded=False,TERM_INT_record_only=True,
+            asynchronous_raise=False,safe_checkpoint_exception='KeyboardInterrupt',
+            original_Plan_Provider_and_53ee_cleanup_unchanged=True,
+            guard_source_sha256='00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128'),
+            'no recorded cancellation or asynchronous signal handoff accepted')
+    require(report['fd_mapping_followup_sha256']==verify_mapping_preparation(args.mapping_freeze),
+            'actual report uses exact reviewed same-FD mapping source closure')
+    require(report.get('artifact_finalization_followup_sha256')==verify_finalization(),
+            'actual report used reviewed final receipt reservation')
+    require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+            and report['own_freeze_sha256']==args.own_freeze
+            and report['benchmark_v8_sha256']==parent['benchmark_v8_manifest']['sha256']
+            and report['capacity_freeze_sha256']==parent['capacity_manifest']['sha256']
+            and report['output_transparency_exact'] is True and report['finalization_errors']==[]
+            and report['performance_timing_usable'] is False
+            and report['native_or_stock_math_changes'] is False,'executed bounded diagnostic-only scope')
+    benchmark=Path(parent['parents']['benchmark']); sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from cgroup_readback_v3 import validate_caps
+    from run_container_v4 import stock_traces
+    pins=verify_preparation(report['benchmark_v8_sha256']); verify_small_sources()
+    protocol=validate_protocol(report['protocol'])
+    protocol_path=Path(report['protocol_path'])
+    require(protocol_path.parent.parent==benchmark and protocol_path.name=='protocol.json'
+            and sha(protocol_path)==report['protocol_sha256'] and read(protocol_path)==protocol,
+            'actual immutable original V8 launch protocol file and complete value')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    require(sha(report['lifecycle_manifest'])==report['lifecycle_manifest_sha256'],'complete original lifecycle manifest')
+    lifecycle=read(report['lifecycle_manifest'])
+    require(lifecycle['owned_cleanup_source_sha256']==parent['owned_helper_sha256']
+            and lifecycle['adapter_source_sha256']==sha(benchmark/'owned_plan_v8.py')
+            and len(lifecycle['receipts'])==len(report['samples'])==4,'all four original owner receipts')
+    order=(('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed'))
+    native_summary=stock_summary=None
+    for sample,receipt_path,expected in zip(report['samples'],lifecycle['receipts'],order):
+        label,mode=expected
+        require((sample['provider'],sample['mode'])==expected and sample['owned_receipt_path']==receipt_path,
+                'all four actual distinct ordered samples')
+        receipt=read(receipt_path); directory=Path(receipt_path).parent
+        require(output in directory.parents,'original owned sample stays in this exclusive output')
+        tree_bound(directory,768*1024**2)
+        successful_owner(receipt,parent)
+        bootstrap=read(directory/'bootstrap.json')
+        validate_caps(bootstrap['cgroup_readback'],protocol['cpu_affinity'])
+        require(bootstrap['provider']==label and bootstrap['preparation_sha256']==report['benchmark_v8_sha256']
+                and bootstrap['stock_math_modified'] is False
+                and bootstrap['actual_cpu_affinity']==sorted(protocol['cpu_affinity']), 'original runtime caps/source bootstrap')
+        base=launch(protocol,pins,report['benchmark_v8_sha256'],label,Path(project['conversion_directory']),
+            profiler=label=='vllm' and mode=='observed')
+        # Receipt's mounted artifact path is authenticated by actual exec SHA/build
+        # below; all other launch bytes must be exact derivation of immutable V8.
+        artifact=Path(next(receipt['argv'][i+1].removesuffix(':/collector-artifact:ro')
+            for i,v in enumerate(receipt['argv'][:-1]) if v=='-v' and receipt['argv'][i+1].endswith(':/collector-artifact:ro'))) if label=='zerogravity' else HERE
+        if label=='zerogravity':
+            build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+            require(sha(artifact/'build.json')==report['collector_build_sha256']
+                    and build['collector_sha256']==report['collector_sha256'], 'independent actual ROOT build binding')
+        command=derive(base,label,mode,args.own_freeze,artifact)
+        replacements={'{owned_name}':receipt['owned_name'],'{owned_nonce}':receipt['owned_nonce'],
+            '{cidfile}':str(directory/'cid'),'{sample_directory}':str(directory)}
+        for key,value in replacements.items(): command=[part.replace(key,value) for part in command]
+        require(receipt['argv']==command,'every actual launch argument equals reviewed exact canonical derivative')
+        workload(sample['observation'],protocol,report['concurrency'])
+        require(read(directory/'measurement.json')==sample['observation'],'actual original measured HTTP observations')
+        if protocol['policy']=='cold_process':
+            require(sample['warmup'] is None and receipt['warmup'] is None,'no collector-only warmup in cold_process')
+        else:
+            require(sample['warmup']==receipt['warmup'] and sample['warmup']['completed'] is True
+                    and sample['warmup']['tokens']>0,'actual exact canonical warmed_process request')
+        if label=='zerogravity':
+            actual=read(directory/'native-exec-attestation.json')
+            require(actual['mode']==mode and actual['argv']==['/runtime/zerogravity']+bootstrap['engine_argv']
+                    and actual['own_freeze_sha256']==args.own_freeze,'same original final exec and own closure')
+            if mode=='observed': native_summary=native_activity(directory,receipt,parent,report)
+        elif mode=='observed':
+            for key,route in (('profile_start','/start_profile'),('profile_stop','/stop_profile')):
+                require(sample[key]['route']==route and sample[key]['HTTP_status']==200,'actual original public profiler controls')
+            # Reparse complete original trace after owner is gone, not stored report count.
+            class Owner: pass
+            owner=Owner(); owner.directory=directory
+            stock_summary=stock_traces(owner,audit)
+            require(stock_summary==sample['actual_recorded_CUDA_activity'],'independent entire actual stock trace replay')
+    for label in ('zerogravity','vllm'):
+        pair=[item for item in report['samples'] if item['provider']==label]
+        require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),'exact all wire token/text/finish/usage transparency')
+    require(native_summary is not None and stock_summary,'both actual canonical workload activity scopes present')
+    result=dict(status='PASS_CONTAINER_RECORDED_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY',
+        own_freeze_sha256=args.own_freeze,report_sha256=sha(output/'report.json'),
+        native=native_summary,stock_recorded=stock_summary,artifact_bounds_before_validation_receipt=tree_bound(output),
+        same_canonical_workload_policy_and_caps_checked=True,plain_observed_wire_outputs_exact=True,
+        source_profiles_numerically_equal=False,source_correct_goodput_qualified=False,
+        matched_complete_hardware_kernel_counts_qualified=False,stock_collector_losslessness_verified=False,
+        timing_usable=False,device_profiler_allocation_admission_qualified=False)
+    frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
+    result['artifact_finalization_followup_sha256']=verify_finalization()
+    result['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
+    result['signal_followup_sha256']=verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+    result['signal_cancellation']=report['signal_cancellation']
+    write_completion(output,'validation.json',result)
+    verify_mapping_preparation(args.mapping_freeze)
+    verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+
+
+if __name__=='__main__': main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/freeze-manifest.json
@@ -0,0 +1,82 @@
+{
+  "actual_beforelaunch_nochild_cases": 2,
+  "actual_harmless_nonprelaunch_container_cases_required": 7,
+  "author_imports_tests_process_container_model_provider_GPU_build_or_payload_hash": false,
+  "author_source_checks": [
+    "AST8 only, no proposed imports/tests",
+    "guard/payload byte-identical accepted originals",
+    "private namespace function AST-identical",
+    "all old V3 functions unchanged except main wrappers",
+    "no container discovery/cleanup patches in harmless actual cases; only observation quiet/Monitor stubs",
+    "all24 original small source/metadata pins exact; V3eightlive/frozen unchanged"
+  ],
+  "checkpoint_pending_exception": "KeyboardInterrupt",
+  "files": [
+    {
+      "bytes": 5444,
+      "file": "README.md",
+      "sha256": "e16d5115fc1908f2113bf433f1da07c835a14a7644fc22ba4502fa8d05ca7196"
+    },
+    {
+      "bytes": 2072,
+      "file": "harmless_http_v2.py",
+      "sha256": "99f3c2f41502c16b28153d0a5352dadf4297b76419796bea9921967c56d70c23"
+    },
+    {
+      "bytes": 15533,
+      "file": "run_container_v4.py",
+      "sha256": "29d0fec6efb65c4192f5dc523afb5fd4cd53f39a1a72bfd7c170ffda36741ada"
+    },
+    {
+      "bytes": 14359,
+      "file": "run_harmless_v4.py",
+      "sha256": "4330e73eaf3ae0e7ba48ef5db320cc9a2324891a2996ef413d43b318ca6beaff"
+    },
+    {
+      "bytes": 1911,
+      "file": "signal_contract_v4.py",
+      "sha256": "ae525c6ad958828e83f5209e583e167d085ea29a6b5abacdb28ae012484d56fb"
+    },
+    {
+      "bytes": 1525,
+      "file": "signal_owner_v4.py",
+      "sha256": "ea5d42c1cfe831bd91dcb1e4d67f462ae44ac90d46f10957dff44dcc53934a2d"
+    },
+    {
+      "bytes": 1428,
+      "file": "sigterm_cancellation.py",
+      "sha256": "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128"
+    },
+    {
+      "bytes": 6885,
+      "file": "source-pins.json",
+      "sha256": "9e6f2d4a164c32d36bc38f81d3c66dca8b5ea4d777c85e18948be03ad97974ec"
+    },
+    {
+      "bytes": 3613,
+      "file": "test_signal_v4.py",
+      "sha256": "4251eb10c217918efce667277576bcbdc23cc3635a8adb41ac812ec696cf3548"
+    },
+    {
+      "bytes": 17868,
+      "file": "validate_container_v4.py",
+      "sha256": "06d2bd980dc7be48017fab2e5ec315eac928323a95e9c9e2cb89f7fa352f9459"
+    }
+  ],
+  "guard_source_sha256": "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128",
+  "harmless_container_identity_or_cleanup_stubbed": false,
+  "harmless_observer_stub": true,
+  "mapping_freeze_sha256": "14bdbe64496a25cecac4f8bd7144ec61c8e160c127c9c3273c0d066e14946a71",
+  "math_queue_provider_or_GPU_qualified": false,
+  "no_asynchronous_signal_raise_across_Popen_handoff_or_cleanup": true,
+  "pure_mock_fixture_methods_prepared": 3,
+  "real_harmless_cases_prepared": 9,
+  "review_artifact": {
+    "bytes": 10245,
+    "file": "REVIEW.diff",
+    "sha256": "fcc96e0d6cfe94416a653a21ec4d7899db81c148a67178984805e95103247a2d"
+  },
+  "schema": "zg.cupti.record-only-safe-owner-signals.preparation.v4",
+  "status": "SOURCE_ONLY_UNEXECUTED_REQUIRES_ROOT_AND_PEER_REVIEW",
+  "unchanged_original_source_metadata_pins": 24
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/REVIEW.diff
@@ -0,0 +1,178 @@
+--- immutable-v3/run_container_v3.py
++++ additive-v4/run_container_v4.py
+@@ -15,6 +15,9 @@
+ import time
+ import urllib.request
+ 
++from signal_contract_v4 import verify_preparation as verify_signal_preparation
++from signal_owner_v4 import checked_sample, checkpoint, signal_metadata
++from sigterm_cancellation import SigtermCancellation
+ from mapping_contract_v3 import (verify_preparation as verify_mapping_preparation,
+     bounded_controller_metadata)
+ from container_contracts import (HERE,TREE_LIMIT,build_binding,fingerprint,frozen,host_parents,
+@@ -64,17 +67,19 @@
+     return output
+ 
+ 
+-def main():
++def main(cancellation):
+     parser=argparse.ArgumentParser(description=__doc__)
+     parser.add_argument('--run',action='store_true')
+     parser.add_argument('--own-freeze',required=True)
+     parser.add_argument('--mapping-freeze',required=True)
++    parser.add_argument('--signal-freeze',required=True)
+     parser.add_argument('--launch-directory',type=Path,required=True)
+     parser.add_argument('--collector-artifact-directory',type=Path,required=True)
+     parser.add_argument('--concurrency',type=int,choices=(1,2,4,8),required=True)
+     parser.add_argument('--output',type=Path,required=True)
+     args=parser.parse_args()
+     require(args.run,'ROOT reviewed explicit actual execution only')
++    verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+     verify_mapping_preparation(args.mapping_freeze)
+     verify_finalization()
+     parent=frozen(args.own_freeze); host_parents(parent)
+@@ -132,6 +137,7 @@
+         native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+     report['artifact_finalization_followup_sha256']=verify_finalization()
+     report['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
++    report['signal_followup_sha256']=verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+     allowed=os.sched_getaffinity(0)
+     require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
+     urllib.request.install_opener(opener())
+@@ -142,12 +148,13 @@
+             spec_command=derive(base,label,mode,args.own_freeze,artifact)
+             plan.spec['providers'][label]['argv']=spec_command
+             sample=dict(provider=label,mode=mode,policy=protocol['policy'],warmup=None)
+-            with plan.sample(label,lambda:compare.preflight(urls[label],alias,None)) as owner:
++            with checked_sample(plan,label,lambda:compare.preflight(urls[label],alias,None),cancellation) as owner:
+                 if protocol['policy']=='warmed_process':
+                     warm=bench.stream_one(urls[label],alias,protocol['requests'][0]['prompt'],
+                         protocol['max_tokens'],180,None,request_id='matched-diag-warmup',return_token_ids=True)
+                     owner.record_warmup(asdict(warm)); sample['warmup']=asdict(warm)
+                     require(warm.completed,'exact canonical unmeasured warmed_process request')
++                checkpoint(cancellation)
+                 before=idle(bench,urls[label]) if label=='zerogravity' else None
+                 if before is not None: write(owner.directory/'health-before.json',before)
+                 controller=None; started=False; primary=None; cleanup=[]
+@@ -155,18 +162,21 @@
+                     if mode=='observed':
+                         if label=='zerogravity':
+                             controller=Controller(owner,owned,parent,args.mapping_freeze)
++                            checkpoint(cancellation)
+                             start=controller.command('START')
+                             started=True
+                             write(owner.directory/'window-start.json',start)
+                         else:
+                             sample['profile_start']=post(urls[label],'/start_profile')
+                             started=True
++                    checkpoint(cancellation)
+                     measured=bench.bench(urls[label],alias,protocol['requests'][0]['prompt'],len(protocol['requests']),
+                         args.concurrency,protocol['max_tokens'],180,None,workload=protocol['requests'],
+                         return_token_ids=True,max_pending=protocol['max_pending'],
+                         launch_deadline_seconds=protocol['launch_deadline_seconds'],run_id='matched-cuda-activity')
+                     fingerprint(measured); owner.record_measurement(measured)
+                     sample['observation']=measured
++                    checkpoint(cancellation)
+                     after=idle(bench,urls[label]) if label=='zerogravity' else None
+                     if after is not None: write(owner.directory/'health-after.json',after)
+                 except BaseException as error:
+@@ -193,10 +203,12 @@
+             report['samples'].append(sample)
+             tree_bound(owner.directory,768*1024**2)
+             tree_bound(output)
++            checkpoint(cancellation)
+         for label in ('zerogravity','vllm'):
+             pair=[item for item in report['samples'] if item['provider']==label]
+             require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),
+                     'complete plain/observed wire IDs/text/finish/usage exact for '+label)
++        checkpoint(cancellation)
+         report['output_transparency_exact']=True
+         report['status']='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+     except BaseException as error:
+@@ -215,6 +227,7 @@
+         try:
+             frozen(args.own_freeze); host_parents(parent); verify_preparation(v8); verify_small_sources()
+             verify_mapping_preparation(args.mapping_freeze)
++            verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+         except BaseException as error: failures.append('post-run immutable source closure: '+repr(error))
+         try:
+             source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+@@ -225,16 +238,18 @@
+                     'unchanged actual model/projector/binary/collector/build after retirement')
+         except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
+         report['finalization_errors']=failures
+-        if failures: report['status']='UNQUALIFIED'
++        report['signal_cancellation']=signal_metadata(cancellation)
++        if failures or cancellation.received: report['status']='UNQUALIFIED'
+         write_completion(output,'report.json',report)
+         verify_mapping_preparation(args.mapping_freeze)
++        verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+         if failures:
+             raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
++        if 'failure' not in report:
++            checkpoint(cancellation)
+ 
+ 
+ 
+ if __name__=='__main__':
+-    def interrupted(_signum,_frame): raise KeyboardInterrupt('ROOT container diagnostic interrupted')
+-    saved=signal.signal(signal.SIGTERM,interrupted)
+-    try: main()
+-    finally: signal.signal(signal.SIGTERM,saved)
++    with SigtermCancellation() as cancellation:
++        main(cancellation)
+--- immutable-v3/validate_container_v3.py
++++ additive-v4/validate_container_v4.py
+@@ -8,6 +8,7 @@
+ from pathlib import Path
+ import sys
+ 
++from signal_contract_v4 import verify_preparation as verify_signal_preparation
+ from mapping_contract_v3 import verify_preparation as verify_mapping_preparation
+ from capacity_v1_loader import activity
+ from container_activity import summarize
+@@ -125,14 +126,23 @@
+     parser=argparse.ArgumentParser()
+     parser.add_argument('--own-freeze',required=True)
+     parser.add_argument('--mapping-freeze',required=True)
++    parser.add_argument('--signal-freeze',required=True)
+     parser.add_argument('--output',type=Path,required=True)
+     args=parser.parse_args()
++    verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+     verify_mapping_preparation(args.mapping_freeze)
+     verify_finalization()
+     parent=frozen(args.own_freeze); host_parents(parent)
+     output=args.output.resolve(strict=True)
+     require(output.parent==HERE,'owned diagnostic output')
+     report=read(output/'report.json')
++    require(report['signal_followup_sha256']==verify_signal_preparation(args.signal_freeze,args.mapping_freeze),
++            'actual exact V4 record-only safe-signal source closure')
++    require(report['signal_cancellation']==dict(recorded=False,TERM_INT_record_only=True,
++            asynchronous_raise=False,safe_checkpoint_exception='KeyboardInterrupt',
++            original_Plan_Provider_and_53ee_cleanup_unchanged=True,
++            guard_source_sha256='00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128'),
++            'no recorded cancellation or asynchronous signal handoff accepted')
+     require(report['fd_mapping_followup_sha256']==verify_mapping_preparation(args.mapping_freeze),
+             'actual report uses exact reviewed same-FD mapping source closure')
+     require(report.get('artifact_finalization_followup_sha256')==verify_finalization(),
+@@ -150,7 +160,7 @@
+     from run_diagnostic_v8 import post
+     from audit_cuda_trace_v8 import audit
+     from cgroup_readback_v3 import validate_caps
+-    from run_container_v3 import stock_traces
++    from run_container_v4 import stock_traces
+     pins=verify_preparation(report['benchmark_v8_sha256']); verify_small_sources()
+     protocol=validate_protocol(report['protocol'])
+     protocol_path=Path(report['protocol_path'])
+@@ -227,8 +237,11 @@
+     frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
+     result['artifact_finalization_followup_sha256']=verify_finalization()
+     result['fd_mapping_followup_sha256']=verify_mapping_preparation(args.mapping_freeze)
++    result['signal_followup_sha256']=verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
++    result['signal_cancellation']=report['signal_cancellation']
+     write_completion(output,'validation.json',result)
+     verify_mapping_preparation(args.mapping_freeze)
++    verify_signal_preparation(args.signal_freeze,args.mapping_freeze)
+ 
+ 
+ if __name__=='__main__': main()

````

