//! P07-E: backend completion bridge for one asynchronous GPU region.
//!
//! A region owns its device-event admissions before the actual CubeCL launch.
//! On successful physical readback, Ready is reported, ephemeral buffer leases
//! retire, and only then can the model consume the completed output. On a
//! device error, host future cancellation/drop or panic, Drop poisons the
//! resident source cache and fences the unfinished event/leases. No queued
//! GPU memory may be reused until trusted device teardown.
//!
//! This is a real async-readback *bridge* but NOT concurrent multi-request
//! scheduling: the current provider holds its residency Mutex across await.
//! It intentionally does NOT call TaskCoordinator::publish: the existing
//! native Rust model remains the sole committed state owner.

use crate::physical_quarantine;
use cubecl::{client::Client, server::Handle};
use std::sync::Mutex;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::device_events::{
    BufferLeaseId, BufferRole, DeviceAccess, DeviceEventId, DeviceEventLedger,
};

use crate::residency::ResidentCache;

/// A privately owned device submission. The type cannot be cloned; on early
/// drop an unobserved GPU result is never reinterpreted as completed.
#[must_use]
pub struct InFlightRegion<'a, T> {
    residency: &'a mut ResidentCache<T>,
    events: &'a Mutex<DeviceEventLedger>,
    id: DeviceEventId,
    dynamic: BufferLeaseId,
    output: BufferLeaseId,
    dynamic_bytes: usize,
    output_bytes: usize,
    /// Clones of every submitted source/dynamic/output GPU Handle. These
    /// survive a caller dropping an awaiting GPU future, rather than
    /// relying on backend-specific detached kernel binding lifetimes.
    gpu_handles: Vec<Handle>,
    gpu_client: Option<Client>,
    armed: bool,
}

impl<'a, T> InFlightRegion<'a, T> {
    /// Reserve scratch bytes and register device submission **before**
    /// launching. This operation is atomic on admission failures.
    pub fn submit(
        residency: &'a mut ResidentCache<T>,
        events: &'a Mutex<DeviceEventLedger>,
        dynamic_bytes: usize,
        output_bytes: usize,
    ) -> Result<Self> {
        if residency.is_poisoned() {
            return Err(Error::State("source residency already fenced".into()));
        }
        let mut ledger = events
            .lock()
            .map_err(|_| Error::State("device event ledger mutex poisoned".into()))?;
        let dynamic = ledger.admit(dynamic_bytes, BufferRole::Activation)?;
        let output = match ledger.admit(output_bytes, BufferRole::Output) {
            Ok(output) => output,
            Err(e) => {
                ledger.release_lease(dynamic)?;
                return Err(e);
            }
        };
        let event = ledger.submit(
            0, // The logical root is independently owned by QwenDenseCpu.
            &[(dynamic, DeviceAccess::Read), (output, DeviceAccess::Write)],
        );
        let id = match event {
            Ok(id) => id,
            Err(e) => {
                ledger.release_lease(output)?;
                ledger.release_lease(dynamic)?;
                return Err(e);
            }
        };
        Ok(Self {
            residency,
            events,
            id,
            dynamic,
            output,
            dynamic_bytes,
            output_bytes,
            gpu_handles: Vec::new(),
            gpu_client: None,
            armed: true,
        })
    }

    /// Retain the physical memory handles for every kernel submission.
    /// Must be called BEFORE using them in a CubeCL launch, so all buffers
    /// remain owned after caller cancellation, future drop or unwind.
    ///
    /// Retained handles are normally freed after verified GPU readback.
    /// On an uncertain completion, Drop moves them into the process-lifetime
    /// service quarantine. No subsequent source-GPU provider may reuse
    /// the same service without trusted recovery.
    pub fn retain_gpu_buffers(&mut self, client: &Client, buffers: &[Handle]) -> Result<()> {
        if buffers.is_empty() {
            return Err(Error::State("cannot retain empty GPU buffer set".into()));
        }
        physical_quarantine::ensure_service_safe(client)?;
        if buffers
            .iter()
            .any(|handle| handle.service != client.service_id())
        {
            return Err(Error::State(
                "GPU buffer belongs to another device service".into(),
            ));
        }
        if let Some(previous) = &self.gpu_client {
            if previous.service_id() != client.service_id() {
                return Err(Error::State(
                    "GPU region tried to use multiple device services".into(),
                ));
            }
        } else {
            self.gpu_client = Some(client.clone());
        }
        self.gpu_handles.extend(buffers.iter().cloned());
        Ok(())
    }

    /// This method may only be called once, and only AFTER successful GPU
    /// async readback AND exact output geometry/finite-value validation.
    ///
    /// A failure in either accounting step leaves the guard armed; its Drop
    /// path poisons source residency even if the event already retired.
    pub fn complete(self, output_bytes: usize) -> Result<()> {
        let uploaded_bytes = self.dynamic_bytes;
        self.complete_with_upload(output_bytes, uploaded_bytes)
    }

    /// Device-created work buffers consume admission but are not H2D uploads.
    pub fn complete_with_upload(
        mut self,
        output_bytes: usize,
        uploaded_bytes: usize,
    ) -> Result<()> {
        if uploaded_bytes > self.dynamic_bytes {
            return Err(Error::State(
                "physical upload exceeds admitted dynamic bytes".into(),
            ));
        }
        if output_bytes != self.output_bytes {
            return Err(Error::State(
                "device completion does not match admitted output byte count".into(),
            ));
        }
        {
            let mut ledger = self
                .events
                .lock()
                .map_err(|_| Error::State("device event ledger mutex poisoned".into()))?;
            ledger.report_physical_ready(self.id)?;
            // The event receipt is consumed by the GPU region, not used to
            // publish a StateRoot. TaskCoordinator still owns that action.
            ledger.request_cancel(self.id)?;
            ledger.release_lease(self.output)?;
            ledger.release_lease(self.dynamic)?;
        }
        self.residency.complete(uploaded_bytes, output_bytes)?;
        self.armed = false;
        Ok(())
    }
}

impl<T> Drop for InFlightRegion<'_, T> {
    fn drop(&mut self) {
        if self.armed {
            self.residency.poison();
            // Move the physical buffer handles into the process-wide GPU
            // quarantine BEFORE releasing the model's cache lock. Provider
            // destruction cannot retire memory for an unsignaled kernel.
            let buffers = std::mem::take(&mut self.gpu_handles);
            if let Some(client) = self.gpu_client.take() {
                physical_quarantine::quarantine_uncertain(client, buffers);
            } else if !buffers.is_empty() {
                // Fail closed even if the service identity is unavailable.
                std::mem::forget(buffers);
            }
            if let Ok(mut ledger) = self.events.lock() {
                // Failure may be post-readback accounting; if that event
                // already retired, the cache poison remains authoritative.
                let _ = ledger.report_completion_unknown(self.id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn device_created_scratch_is_admitted_but_not_reported_uploaded() {
        let events = std::sync::Mutex::new(
            zerogravity_core::device_events::DeviceEventLedger::new(1, 128).unwrap(),
        );
        let mut cache = crate::residency::ResidentCache::<u8>::new(256).unwrap();
        super::InFlightRegion::submit(&mut cache, &events, 80, 16)
            .unwrap()
            .complete_with_upload(16, 12)
            .unwrap();
        assert_eq!(cache.stats().dynamic_uploaded_bytes, 12);
        assert_eq!(cache.stats().completed_readback_bytes, 16);
        assert_eq!(events.lock().unwrap().resident_bytes(), 0);
        assert!(!cache.is_poisoned());
    }
    #[test]
    fn impossible_upload_receipt_remains_unknown_and_fenced() {
        let events = std::sync::Mutex::new(
            zerogravity_core::device_events::DeviceEventLedger::new(1, 128).unwrap(),
        );
        let mut cache = crate::residency::ResidentCache::<u8>::new(256).unwrap();
        assert!(
            super::InFlightRegion::submit(&mut cache, &events, 80, 16)
                .unwrap()
                .complete_with_upload(16, 81)
                .is_err()
        );
        assert_eq!(cache.stats().completed_regions, 0);
        assert_eq!(events.lock().unwrap().resident_bytes(), 96);
        assert!(cache.is_poisoned());
    }

    use super::*;
    use zerogravity_core::device_events::EventPhase;

    #[test]
    fn mock_future_completion_requires_real_receipt_and_retires_buffers() {
        let events = Mutex::new(DeviceEventLedger::new(1, 128).unwrap());
        let mut cache = ResidentCache::<u8>::new(256).unwrap();
        let ticket = InFlightRegion::submit(&mut cache, &events, 32, 16).unwrap();
        assert_eq!(
            events.lock().unwrap().phase(ticket.id).unwrap(),
            EventPhase::Submitted
        );
        assert_eq!(events.lock().unwrap().resident_bytes(), 48);
        ticket.complete(16).unwrap();
        assert!(!cache.is_poisoned());
        assert_eq!(cache.stats().completed_regions, 1);
        assert_eq!(cache.stats().dynamic_uploaded_bytes, 32);
        assert_eq!(events.lock().unwrap().resident_bytes(), 0);
        assert_eq!(events.lock().unwrap().pending_events(), 0);
    }

    #[test]
    fn pending_future_drop_fences_event_and_source_buffers() {
        let events = Mutex::new(DeviceEventLedger::new(1, 128).unwrap());
        let mut cache = ResidentCache::<u8>::new(256).unwrap();
        let id = {
            let ticket = InFlightRegion::submit(&mut cache, &events, 24, 16).unwrap();
            ticket.id
            // A caller discards the future without physical GPU completion.
            // The source cache must become permanently fenced.
        };
        assert!(cache.is_poisoned());
        let ledger = events.lock().unwrap();
        assert_eq!(ledger.phase(id).unwrap(), EventPhase::CompletionUnknown);
        assert_eq!(ledger.resident_bytes(), 40);
        drop(ledger);
        assert!(InFlightRegion::submit(&mut cache, &events, 16, 16).is_err());
    }

    /// A Rust future that was actually polled and then cancelled must
    /// poison any source allocation whose physical result was not signaled.
    #[test]
    fn cancellation_of_pending_async_future_fences_source_and_buffers() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};

        let events = Mutex::new(DeviceEventLedger::new(1, 128).unwrap());
        let mut cache = ResidentCache::<u8>::new(256).unwrap();
        let future = async {
            let pending = InFlightRegion::submit(&mut cache, &events, 28, 20).unwrap();
            // In the real provider this would be CubeCL read_async.await.
            std::future::pending::<()>().await;
            pending.complete(20).unwrap();
        };
        let mut future = Box::pin(future);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(events.lock().unwrap().resident_bytes(), 48);
        drop(future);
        assert!(cache.is_poisoned());
        let state = events.lock().unwrap();
        assert_eq!(state.pending_events(), 1);
        assert_eq!(state.resident_bytes(), 48);
    }

    #[test]
    fn async_cache_serialization_and_cancel_fence_are_fail_closed() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};

        let cache = async_lock::Mutex::new(ResidentCache::<u8>::new(128).unwrap());
        let events = Mutex::new(DeviceEventLedger::new(1, 128).unwrap());

        let first = async {
            let mut resident = cache.lock().await;
            let pending = InFlightRegion::submit(&mut resident, &events, 32, 32).unwrap();
            std::future::pending::<()>().await;
            pending.complete(32).unwrap();
        };
        let mut first = Box::pin(first);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(matches!(first.as_mut().poll(&mut context), Poll::Pending));

        let second = async {
            let mut resident = cache.lock().await;
            InFlightRegion::submit(&mut resident, &events, 16, 16)
                .map(|ticket| ticket.complete(16).unwrap())
        };
        let mut second = Box::pin(second);
        assert!(matches!(second.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(events.lock().unwrap().resident_bytes(), 64);

        // Dropping the first future simulates host cancellation while GPU
        // readback is pending; the async lock is then released, but the
        // source lease and uncertain event remain fenced.
        drop(first);
        assert!(cache.lock_blocking().is_poisoned());
        assert!(matches!(
            second.as_mut().poll(&mut context),
            Poll::Ready(Err(_))
        ));
        assert_eq!(events.lock().unwrap().resident_bytes(), 64);
        assert_eq!(events.lock().unwrap().pending_events(), 1);
    }

    /// Test-only backend implements the new event bridge inside the real
    /// Qwen3.5 MatVecKernel integration, without claiming physical GPU work.
    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires local SHA-pinned Qwen3.5 0.8B and 2B GGUF models"]
    fn real_qwen_event_bridge_completed_vs_unknown_preserves_state() {
        use crate::residency::ResidencyKey;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
        use zerogravity_model_qwen35::QwenDenseCpu;

        struct TestEventBackend {
            cache: Mutex<ResidentCache<(crate::host_descriptor::StaticSource, Vec<u8>)>>,
            ledger: Mutex<DeviceEventLedger>,
            completed: AtomicUsize,
            inject_device_loss: bool,
        }
        impl TestEventBackend {
            fn new(inject_device_loss: bool) -> Self {
                Self {
                    cache: Mutex::new(ResidentCache::new(16 * 1024 * 1024).unwrap()),
                    ledger: Mutex::new(DeviceEventLedger::new(1, 16 * 1024 * 1024).unwrap()),
                    completed: AtomicUsize::new(0),
                    inject_device_loss,
                }
            }
        }
        impl MatVecKernel for TestEventBackend {
            fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
                if region.tensor_name != "blk.0.attn_gate.weight" {
                    return Ok(RegionExecution::Unsupported);
                }
                let key = ResidencyKey::from_region(region)?;
                let bytes = crate::host_descriptor::StaticSource::gpu_bytes_for(region)?;
                let mut cache = self
                    .cache
                    .lock()
                    .map_err(|_| Error::State("test GPU cache mutex poisoned".into()))?;
                let source = cache.get_or_try_upload(key, bytes, || {
                    crate::host_descriptor::StaticSource::from_region(region)
                })?;
                let prepared = match source.0.dynamic(region.activation)? {
                    crate::host_descriptor::DynamicStep::Q4(q) => q,
                    crate::host_descriptor::DynamicStep::Mixed(_) => {
                        return Err(Error::State("expected Q4_0".into()));
                    }
                };
                // This is a MOCK mathematical output, not a GPU kernel.
                // Copy the borrowed test tensor to allow the separate mutable
                // admission, as a real GPU provider would clone its Handle.
                let packed = source.1.clone();
                let dynamic_bytes = (prepared.scales.len() + prepared.activations.len()) * 4;
                let output_bytes = prepared.rows * 4;
                let inflight =
                    InFlightRegion::submit(&mut cache, &self.ledger, dynamic_bytes, output_bytes)?;
                if self.inject_device_loss {
                    return Err(Error::State("injected GPU completion failure".into()));
                }
                let mut result = Vec::with_capacity(prepared.rows);
                for row in 0..prepared.rows {
                    let mut acc = 0f32;
                    for block in 0..prepared.blocks_per_row {
                        let id = row * prepared.blocks_per_row + block;
                        let mut dot = 0i32;
                        for lane in 0..16 {
                            let b = packed[id * 16 + lane];
                            let offset = block * 32 + lane;
                            dot += (i32::from(b & 15) - 8) * prepared.activations[offset];
                            dot += (i32::from(b >> 4) - 8) * prepared.activations[offset + 16];
                        }
                        acc = (dot as f32).mul_add(prepared.scales[id], acc);
                    }
                    result.push(acc);
                }
                // Pretend the independent test backend has physically
                // signaled; the event bridge enforces all admission/retire.
                inflight.complete(output_bytes)?;
                self.completed.fetch_add(1, Ordering::Relaxed);
                Ok(RegionExecution::Completed(result))
            }
        }

        for model in ["qwen35-0.8b", "qwen35-2b"] {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target/models")
                .join(model);
            let gguf = dir.join("model.gguf");
            let config = dir.join("config.json");
            let tokens = [11u32, 353u32];
            let mut cpu = QwenDenseCpu::open(&gguf, &config).unwrap();
            let reference = cpu.prefill_transaction(&tokens).unwrap();

            let backend = Arc::new(TestEventBackend::new(false));
            let mut accelerated =
                QwenDenseCpu::open_with_kernel(&gguf, &config, backend.clone()).unwrap();
            let generated = accelerated.prefill_transaction(&tokens).unwrap();
            assert_eq!(
                generated, reference,
                "{model}: event path changed full-model logits"
            );
            assert_eq!(accelerated.state_fingerprint(), cpu.state_fingerprint());
            assert_eq!(backend.completed.load(Ordering::Relaxed), tokens.len());
            let stats = backend.cache.lock().unwrap().stats();
            assert_eq!(stats.source_uploads, 1);
            assert_eq!(stats.source_cache_hits, 1);
            assert_eq!(backend.ledger.lock().unwrap().resident_bytes(), 0);
            assert_eq!(backend.ledger.lock().unwrap().pending_events(), 0);

            let failed = Arc::new(TestEventBackend::new(true));
            let mut stopped =
                QwenDenseCpu::open_with_kernel(&gguf, &config, failed.clone()).unwrap();
            let original = stopped.state_fingerprint();
            assert!(stopped.prefill_transaction(&tokens).is_err());
            assert_eq!(stopped.position(), 0);
            assert_eq!(stopped.root_id(), 0);
            assert_eq!(stopped.state_fingerprint(), original);
            assert!(failed.cache.lock().unwrap().is_poisoned());
            assert_eq!(failed.ledger.lock().unwrap().pending_events(), 1);
            assert!(stopped.prefill_transaction(&tokens).is_err());
            assert_eq!(stopped.root_id(), 0);
            println!(
                "P07_E_REAL_QWEN_HOST_EVENT_BRIDGE_PASS model={model} tokens={} event_ready={} failed_root=0",
                tokens.len(),
                backend.completed.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn no_partial_admission_and_malformed_ready_poison() {
        let events = Mutex::new(DeviceEventLedger::new(1, 64).unwrap());
        let mut cache = ResidentCache::<u8>::new(256).unwrap();
        assert!(InFlightRegion::submit(&mut cache, &events, 48, 32).is_err());
        assert!(!cache.is_poisoned());
        assert_eq!(events.lock().unwrap().resident_bytes(), 0);
        {
            let pending = InFlightRegion::submit(&mut cache, &events, 20, 20).unwrap();
            assert!(pending.complete(19).is_err());
        }
        assert!(cache.is_poisoned());
        assert_eq!(cache.stats().completed_regions, 0);
        assert_eq!(events.lock().unwrap().resident_bytes(), 40);
    }
}
