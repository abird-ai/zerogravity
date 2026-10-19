//! P07-C: real immutable CubeCL weight residency with source-keyed leases.
//!
//! One provider owns one device, a bounded source weight cache and a
//! synchronous completion boundary. Its cache mutex stays locked while
//! source buffers are in use by queued GPU work; successful read_one is
//! the physical Ready fence and unlocks any later LRU eviction.
//!
//! Static original GGUF weight data is uploaded only on a cache miss.
//! Each step still transfers dynamic source-Q8 activations and the
//! per-source-block FP32 composite scales. We DO NOT claim a fully resident
//! model, KV residency, async TaskCoordinator or high GPU throughput.
//! The owner of StateRoot remains the Rust model executor.
//!
//! Reopening/replacing the original model gets a distinct SourceIdentity.
//! An invalid/unbound identity never aliases a previously cached buffer.

#[cfg(feature = "qualification")]
mod activation_qualification;
mod binary_execution;
mod device_execution;
mod embedding_execution;

use async_lock::Mutex as AsyncMutex;
use cubecl::{Device, client::Client, prelude::*, server::Handle};
use std::collections::BTreeSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::{
    AttentionQkCompleted, AttentionQkExecution, AttentionQkRegion, AttentionSoftmaxCompleted,
    AttentionSoftmaxExecution, AttentionSoftmaxRegion, AttentionValueRegion,
    GdnCoefficientCompleted, GdnCoefficientExecution, GdnCoefficientRegion, GdnCoefficients,
    GdnConvCompleted, GdnConvExecution, GdnConvRegion, GdnRecurrentCompleted,
    GdnRecurrentExecution, GdnRecurrentRegion, GroupedRmsNormRegion, MatVecKernel, MatVecRegion,
    RegionExecution, RmsNormRegion, RopeKvCompleted, RopeKvExecution, RopeKvRegion, SiluRegion,
};
use zerogravity_core::device_events::DeviceEventLedger;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};

use crate::async_bridge::InFlightRegion;
use crate::gdn_conv;
use crate::host_descriptor::{DeviceStep, DynamicStep, StaticSource};
use crate::mixed_ops;
use crate::physical_queue::{PhysicalQueue, QueueStats, queue_for};
use crate::physical_service::PhysicalServiceBudget;
use crate::q4_q8;
use crate::residency::{ResidencyKey, ResidencyStats, ResidentCache};
use crate::rms_norm;
use crate::source_accumulation::{self, SourceAccumulation};

/// Proof that device work and host-visible output passed a real readback.
#[must_use]
struct ReadbackReady {
    values: Vec<f32>,
    dynamic_admitted_bytes: usize,
    dynamic_h2d_bytes: usize,
    completed_d2h_bytes: usize,
}

struct ResidentHandle {
    handle: Handle,
    bytes: usize,
    elements: usize,
    rows: usize,
    kind: QuantKind,
}

struct ResidentWeightLease {
    client: Client,
    fence_flag: Arc<AtomicBool>,
    /// Managed CubeCL physical memory belongs exclusively to this provider
    /// and never crosses to another Device service or stream.
    handle: Handle,
    /// Number of bytes retained on device for this original source tensor.
    bytes: usize,
    elements: usize,
    host: StaticSource,
}

impl Drop for ResidentWeightLease {
    fn drop(&mut self) {
        if self.fence_flag.load(Ordering::Acquire) {
            crate::physical_quarantine::quarantine_uncertain(
                self.client.clone(),
                vec![self.handle.clone()],
            );
        }
    }
}

enum PreparedStep {
    Q4(q4_q8::PreparedQ4Q8),
    Mixed(mixed_ops::Prepared),
    Device(DeviceStep),
}

impl PreparedStep {
    fn rows(&self) -> usize {
        match self {
            Self::Q4(p) => p.rows,
            Self::Mixed(p) => p.rows,
            Self::Device(p) => p.layout.rows,
        }
    }
    fn dynamic_admission_bytes(&self) -> Result<usize> {
        if let Self::Device(p) = self {
            return crate::device_activation::DeviceActivation::admission_bytes(&p.layout)?
                .checked_sub(crate::device_activation::DeviceActivation::status_bytes(
                    &p.layout,
                )?)
                .ok_or_else(|| Error::Limit("device activation admission underflow".into()));
        }
        let elements = match self {
            Self::Q4(p) => p.scales.len().checked_add(p.activations.len()),
            Self::Device(_) => unreachable!("device admission handled above"),
            Self::Mixed(p) => {
                if p.kind == QuantKind::F32 {
                    Some(p.input_f32.len())
                } else {
                    p.scales
                        .len()
                        .checked_add(p.activation.len())
                        .and_then(|n| n.checked_add(p.offsets.len()))
                }
            }
        }
        .ok_or_else(|| Error::Limit("GPU activation/scale footprint overflow".into()))?;
        elements
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("GPU dynamic buffer byte size overflow".into()))
    }

    fn dynamic_upload_bytes(&self) -> Result<usize> {
        match self {
            Self::Device(p) => crate::device_activation::DeviceActivation::upload_bytes(&p.layout),
            _ => self.dynamic_admission_bytes(),
        }
    }
    fn readback_bytes(&self) -> Result<usize> {
        let status = match self {
            Self::Device(p) => crate::device_activation::DeviceActivation::status_bytes(&p.layout)?,
            _ => 0,
        };
        self.rows()
            .checked_mul(4)
            .and_then(|n| n.checked_add(status))
            .ok_or_else(|| Error::Limit("source output cohort overflow".into()))
    }

    fn kind(&self) -> QuantKind {
        match self {
            Self::Q4(_) => QuantKind::Q4_0,
            Self::Mixed(p) => p.kind,
            Self::Device(p) => p.kind,
        }
    }

    /// Dispatch exactly one validated source region on the GPU. The retained
    /// source weight handle is reused, while each request gets new dynamic
    /// activation, scale, offset and output scratch leases. The caller
    /// holds exclusive cache ownership through the final readback.
    async fn execute(
        &self,
        client: &Client,
        source: &ResidentHandle,
        in_flight: &mut InFlightRegion<'_, ResidentWeightLease>,
        numeric_mode: SourceAccumulation,
    ) -> Result<ReadbackReady> {
        if source.bytes == 0
            || source.elements == 0
            || source.rows != self.rows()
            || source.kind != self.kind()
            || source.handle.service != client.service_id()
        {
            return Err(Error::State(
                "GPU resident lease codec, weight or output geometry changed".into(),
            ));
        }
        if let Self::Device(p) = self {
            return Self::execute_device(p, client, source, in_flight, numeric_mode).await;
        }
        let rows = self.rows();
        let count = u32::try_from(rows)
            .map_err(|_| Error::Limit("CubeCL output row grid overflow".into()))?;
        let grid = CubeCount::Static(count.div_ceil(32), 1, 1);
        let threads = CubeDim::new_1d(32);
        let output = client.empty(
            rows.checked_mul(4)
                .ok_or_else(|| Error::Limit("GPU output bytes overflow".into()))?,
        );
        // Keep the original checked source Handle and output allocation
        // alive until after physical completion, including cancelled futures.
        in_flight.retain_gpu_buffers(client, &[source.handle.clone(), output.clone()])?;
        let mut dynamic_bytes;
        match self {
            Self::Device(_) => unreachable!("device execution handled above"),
            Self::Q4(p) => {
                let scales = client.create_from_slice(f32::as_bytes(&p.scales));
                let q8 = client.create_from_slice(i32::as_bytes(&p.activations));
                in_flight.retain_gpu_buffers(client, &[scales.clone(), q8.clone()])?;
                dynamic_bytes = p.scales.len() * 4 + p.activations.len() * 4;
                // SAFETY: rows, blocks and every physical output index were
                // verified against the source GGUF before dispatch.
                unsafe {
                    q4_q8::source_q4_q8_gemv::launch_unchecked(
                        client,
                        grid,
                        threads,
                        BufferArg::from_raw_parts(source.handle.clone(), source.elements),
                        BufferArg::from_raw_parts(scales, p.scales.len()),
                        BufferArg::from_raw_parts(q8, p.activations.len()),
                        BufferArg::from_raw_parts(output.clone(), rows),
                        p.blocks_per_row,
                        rows,
                        numeric_mode == SourceAccumulation::WideF64,
                    );
                }
            }
            Self::Mixed(p) => {
                if p.kind == QuantKind::F32 {
                    let input = client.create_from_slice(f32::as_bytes(&p.input_f32));
                    in_flight.retain_gpu_buffers(client, std::slice::from_ref(&input))?;
                    dynamic_bytes = p.input_f32.len() * 4;
                    // SAFETY: F32 row geometry is validated in prepare().
                    unsafe {
                        mixed_ops::f32_ggml_dot_gemv::launch_unchecked(
                            client,
                            grid,
                            threads,
                            BufferArg::from_raw_parts(source.handle.clone(), source.elements),
                            BufferArg::from_raw_parts(input, p.input_f32.len()),
                            BufferArg::from_raw_parts(output.clone(), rows),
                            p.cols,
                            rows,
                            numeric_mode == SourceAccumulation::WideF64,
                        );
                    }
                } else {
                    let scales = client.create_from_slice(f32::as_bytes(&p.scales));
                    let q8 = client.create_from_slice(i32::as_bytes(&p.activation));
                    in_flight.retain_gpu_buffers(client, &[scales.clone(), q8.clone()])?;
                    dynamic_bytes = p.scales.len() * 4 + p.activation.len() * 4;
                    match p.kind {
                        QuantKind::Q4_1 => {
                            let offsets = client.create_from_slice(f32::as_bytes(&p.offsets));
                            in_flight.retain_gpu_buffers(client, std::slice::from_ref(&offsets))?;
                            dynamic_bytes += p.offsets.len() * 4;
                            // SAFETY: source Q4_1, dynamic Q8_1 and offsets
                            // were validated against the matrix geometry.
                            unsafe {
                                mixed_ops::q4_1_q8_1_gemv::launch_unchecked(
                                    client,
                                    grid,
                                    threads,
                                    BufferArg::from_raw_parts(
                                        source.handle.clone(),
                                        source.elements,
                                    ),
                                    BufferArg::from_raw_parts(scales, p.scales.len()),
                                    BufferArg::from_raw_parts(offsets, p.offsets.len()),
                                    BufferArg::from_raw_parts(q8, p.activation.len()),
                                    BufferArg::from_raw_parts(output.clone(), rows),
                                    p.blocks,
                                    rows,
                                    numeric_mode == SourceAccumulation::WideF64,
                                );
                            }
                        }
                        QuantKind::Q8_0 => {
                            // SAFETY: original source block/row geometry is
                            // checked before this Q8_0 source kernel launch.
                            unsafe {
                                mixed_ops::q8_0_q8_0_gemv::launch_unchecked(
                                    client,
                                    grid,
                                    threads,
                                    BufferArg::from_raw_parts(
                                        source.handle.clone(),
                                        source.elements,
                                    ),
                                    BufferArg::from_raw_parts(scales, p.scales.len()),
                                    BufferArg::from_raw_parts(q8, p.activation.len()),
                                    BufferArg::from_raw_parts(output.clone(), rows),
                                    p.blocks,
                                    rows,
                                    numeric_mode == SourceAccumulation::WideF64,
                                );
                            }
                        }
                        QuantKind::Q6K => {
                            // SAFETY: original 210-byte Q6_K source blocks
                            // and 256-code signed Q8_K inputs were checked.
                            unsafe {
                                mixed_ops::q6_k_q8_k_gemv::launch_unchecked(
                                    client,
                                    grid,
                                    threads,
                                    BufferArg::from_raw_parts(
                                        source.handle.clone(),
                                        source.elements,
                                    ),
                                    BufferArg::from_raw_parts(scales, p.scales.len()),
                                    BufferArg::from_raw_parts(q8, p.activation.len()),
                                    BufferArg::from_raw_parts(output.clone(), rows),
                                    p.blocks,
                                    rows,
                                    numeric_mode == SourceAccumulation::WideF64,
                                );
                            }
                        }
                        _ => {
                            return Err(Error::Unsupported(
                                "P07-C source codec not implemented".into(),
                            ));
                        }
                    }
                }
            }
        }
        // Test-only opt-in physical submission-drop checkpoint, deliberately
        // after real CubeCL command launch but before any readback/Ready.
        // Excluded entirely from production CubeCL/CPU builds.
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;

        // Real synchronous CubeCL readback confirms submitted work and
        // readback are physically completed and host-visible. No unready
        // event or outstanding kernel is reported as a Completed region.
        // The async read actually waits for completion of the submitted
        // CubeCL device work; merely launching a kernel is not a Ready event.
        let mut completed = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("P07-E physical GPU readback failed: {e}")))?;
        if completed.len() != 1 {
            return Err(Error::State(
                "GPU async readback returned incorrect handle count".into(),
            ));
        }
        let bytes = completed.remove(0);
        if bytes.len() != rows * 4 {
            return Err(Error::State("GPU output readback byte size invalid".into()));
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("verified output stride")))
            .collect::<Vec<_>>();
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error::State(
                "GPU device readback contained nonfinite source output".into(),
            ));
        }
        Ok(ReadbackReady {
            dynamic_admitted_bytes: dynamic_bytes,
            completed_d2h_bytes: values.len() * 4,
            dynamic_h2d_bytes: dynamic_bytes,
            values,
        })
    }
}

/// Physical source-resident backend. Supports multiple source tensors under
/// one bounded device weight budget. It is Send+Sync for the model executor,
/// but each actual GPU operation serializes while a device lease is in use.
/// No LRU eviction occurs until the earlier GPU readback returned Ready.
pub struct CubeResidentProvider {
    device: Device,
    selected: BTreeSet<String>,
    max_rows: usize,
    cache: AsyncMutex<ResidentCache<ResidentWeightLease>>,
    /// One local physical service event namespace. Tracks dynamic scratch
    /// and output buffers; resident static weights remain pinned in cache.
    events: Mutex<DeviceEventLedger>,
    /// Shared physical provider envelope. Cache/event budgets subdivide this
    /// envelope instead of making duplicate global reservations.
    service: PhysicalServiceBudget,
    queue: Arc<PhysicalQueue>,
    owner: ResourceOwner,
    numeric_mode: SourceAccumulation,
    device_activation: bool,
    device_quantized_regions: AtomicUsize,
}

impl CubeResidentProvider {
    pub fn new(
        device: Device,
        selected: impl IntoIterator<Item = String>,
        max_rows: usize,
        max_resident_bytes: usize,
        physical_device_cap: u64,
        owner: ResourceOwner,
    ) -> Result<Self> {
        crate::backend::validate_provider(&device)?;
        if max_rows == 0 || max_rows > 1_000_000 {
            return Err(Error::Limit("invalid max GPU output rows".into()));
        }
        let list = selected.into_iter().collect::<Vec<_>>();
        let names = list.iter().cloned().collect::<BTreeSet<_>>();
        if names.is_empty() || names.len() != list.len() || names.iter().any(|x| x.is_empty()) {
            return Err(Error::Malformed(
                "selected GPU model tensor names invalid/duplicate".into(),
            ));
        }
        let cap = u64::try_from(max_resident_bytes)
            .map_err(|_| Error::Limit("GPU provider cap overflows u64".into()))?;
        let service = PhysicalServiceBudget::attach(
            &device.client(),
            physical_device_cap,
            cap,
            owner.clone(),
        )?;
        let card = service.physical_key();
        let queue = queue_for(card.clone())?;
        if queue.key() != &card {
            return Err(Error::State(
                "physical queue attached to a different GPU".into(),
            ));
        }
        Ok(Self {
            device,
            selected: names,
            max_rows,
            cache: AsyncMutex::new(ResidentCache::new(max_resident_bytes)?),
            events: Mutex::new(DeviceEventLedger::new(1, max_resident_bytes)?),
            service,
            queue,
            owner,
            numeric_mode: SourceAccumulation::WideF64,
            device_activation: false,
            device_quantized_regions: AtomicUsize::new(0),
        })
    }

    #[cfg(feature = "qualification")]
    pub(crate) fn with_device_activation_quantization(mut self, enabled: bool) -> Self {
        self.device_activation = enabled;
        self
    }

    /// ONLY for a physically checked, source-bit-comparing diagnostic.
    /// Normal source-model providers default to the FP64 wide kernel; the
    /// native shader F32 FMA is known to differ on the AMD Renoir RADV card.
    #[cfg(feature = "qualification")]
    pub(crate) fn with_source_diagnostic_accumulation(
        mut self,
        accumulation: SourceAccumulation,
    ) -> Self {
        self.numeric_mode = accumulation;
        self
    }

    pub fn quarantined_physical_handles(&self) -> Result<usize> {
        let client = self.device.client();
        crate::physical_quarantine::quarantined_handles_for(&client)
    }

    pub fn in_flight_device_bytes(&self) -> Result<usize> {
        self.events
            .lock()
            .map(|ledger| ledger.resident_bytes())
            .map_err(|_| Error::State("CubeCL event ledger mutex poisoned".into()))
    }

    pub fn worker_stats(&self) -> Result<QueueStats> {
        self.queue.stats()
    }

    pub fn physical_key(&self) -> zerogravity_core::resource_arbiter::PhysicalDeviceKey {
        self.service.physical_key()
    }

    /// Device-wide generation reset: invalidates ALL queued sessions sharing
    /// this physical card. NEVER use for individual request cancellation;
    /// dropping that request's WaitingTicket preserves other sessions. An
    /// active command must still reach genuine readback/quarantine and a
    /// revoked queue generation cannot publish a stale Qwen StateRoot.
    pub fn advance_physical_card_epoch(&self, next: u64) -> Result<usize> {
        self.queue.advance_epoch(next)
    }

    pub fn unreserved_physical_bytes(&self) -> Result<u64> {
        self.service.remaining()
    }

    pub fn stats(&self) -> Result<ResidencyStats> {
        let resident = self.cache.lock_blocking();
        let mut stats = resident.stats();
        stats.device_quantized_regions = self.device_quantized_regions.load(Ordering::Relaxed);
        Ok(stats)
    }

    pub fn is_fenced(&self) -> Result<bool> {
        Ok(self.cache.lock_blocking().is_poisoned())
    }
}

impl CubeResidentProvider {
    /// Asynchronous CubeCL physical readback integration. The current
    /// source cache is held exclusively across await using an async-aware
    /// mutex, preserving source weight lease validity without blocking the
    /// executor. This is not yet concurrent multi-request
    /// stream scheduling: MatVecKernel remains a synchronous adapter.
    pub async fn try_execute_async(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
        self.try_execute_async_for(region, self.owner.clone(), None)
            .await
    }

    /// The source tensor allocation remains borrowed from the caller across
    /// the queue and readback: no second model engine, deep-copy or detached
    /// GPU future is created. The physical card has ONE bounded fair queue.
    pub async fn try_execute_async_for(
        &self,
        region: MatVecRegion<'_>,
        session: ResourceOwner,
        deadline_tick: Option<u64>,
    ) -> Result<RegionExecution> {
        if !self.selected.contains(region.tensor_name)
            || region.rows > self.max_rows
            || !matches!(
                region.kind,
                QuantKind::Q4_0
                    | QuantKind::Q4_1
                    | QuantKind::Q8_0
                    | QuantKind::Q6K
                    | QuantKind::F32
            )
        {
            return Ok(RegionExecution::Unsupported);
        }
        source_accumulation::ensure_capability(&self.device.client(), self.numeric_mode)?;
        if session.model != self.owner.model {
            return Err(Error::State(
                "physical source provider cannot execute another model identity".into(),
            ));
        }
        // Input validation belongs BEFORE the global Submitted boundary.
        // A malformed caller must not poison a physical card that never
        // received any source upload or kernel dispatch.
        if region.activation.len() != region.columns
            || !region.activation.iter().all(|value| value.is_finite())
        {
            return Err(Error::Malformed(
                "GPU activation shape/nonfinite input".into(),
            ));
        }
        let key = ResidencyKey::from_region(region)?;
        let bytes = StaticSource::gpu_bytes_for(region)?;
        // Source physical weight + dynamic activation/scales/offsets + output
        // share one ALREADY charged provider envelope. This is preflight
        // geometry, not a second GPU memory reservation or a source decode.
        let blocks = region.columns
            / if region.kind == QuantKind::Q6K {
                256
            } else {
                32
            };
        let scale_blocks = if region.kind == QuantKind::F32 {
            0
        } else {
            region
                .rows
                .checked_mul(blocks)
                .ok_or_else(|| Error::Limit("queued source block count overflow".into()))?
        };
        let scale_fields = if region.kind == QuantKind::Q4_1 { 2 } else { 1 };
        let dynamic_count = region
            .columns
            .checked_add(
                scale_blocks
                    .checked_mul(scale_fields)
                    .ok_or_else(|| Error::Limit("queued source scale count overflow".into()))?,
            )
            .ok_or_else(|| Error::Limit("queued activation geometry overflow".into()))?;
        let device_layout = if self.device_activation && region.kind != QuantKind::F32 {
            if !self
                .device
                .client()
                .properties()
                .features
                .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
            {
                return Err(Error::Unsupported(
                    "device source quantization requires shader UInt64".into(),
                ));
            }
            Some(crate::activation_quant::ActivationLayout::new(
                region.kind,
                region.columns,
                region.rows,
            )?)
        } else {
            None
        };
        let status_bytes = match &device_layout {
            Some(l) => crate::device_activation::DeviceActivation::status_bytes(l)?,
            None => 0,
        };
        let dynamic_bytes = match &device_layout {
            Some(l) => crate::device_activation::DeviceActivation::admission_bytes(l)?
                .checked_sub(status_bytes),
            None => dynamic_count.checked_mul(4),
        }
        .ok_or_else(|| Error::Limit("queued source scratch bytes overflow".into()))?;
        let output_bytes = region
            .rows
            .checked_mul(4)
            .and_then(|n| n.checked_add(status_bytes))
            .ok_or_else(|| Error::Limit("queued source output cohort overflow".into()))?;
        let ticket = self.queue.submit(
            &self.service,
            session,
            &[
                AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: u64::try_from(bytes)
                        .map_err(|_| Error::Limit("GPU weight bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("GPU transient bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes)
                        .map_err(|_| Error::Limit("GPU output bytes overflow".into()))?,
                },
            ],
            deadline_tick,
        )?;
        // Do not hold source cache or global resource mutex during fairness
        // queueing. Dropping an unselected Future cancels without GPU fencing.
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        // Exclusively own the resident memory while validating, submitting,
        // waiting, and recording completion, including LRU eviction.
        // Unlike std::sync::Mutex, this lock suspends the caller without
        // blocking an async executor while the GPU readback is pending.
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        // All fallible HOST validation happens before the retained physical
        // lease changes to Submitted. A malformed source, scale, activation or
        // locally satisfiable scratch request must not permanently fence the
        // entire physical card before it has received a GPU command.
        //
        // On a cache hit reuse immutable decoded host metadata. A miss decodes
        // the original GGUF bytes ONCE here, before its physical H2D upload.
        let prepare = |host: &StaticSource| -> Result<PreparedStep> {
            if device_layout.is_some() {
                return Ok(PreparedStep::Device(
                    host.dynamic_device(region.activation)?,
                ));
            }
            match host.dynamic(region.activation)? {
                DynamicStep::Q4(p) => Ok(PreparedStep::Q4(p)),
                DynamicStep::Mixed(p) => Ok(PreparedStep::Mixed(p)),
            }
        };
        let (prepared, mut staged_source) = match resident.cached(&key) {
            Some(existing) => {
                if existing.bytes != bytes {
                    return Err(Error::State("GPU source byte geometry changed".into()));
                }
                (prepare(&existing.host)?, None)
            }
            None => {
                let source = StaticSource::from_region(region)?;
                let prepared = prepare(&source.0)?;
                (prepared, Some(source))
            }
        };
        if prepared.readback_bytes()? != output_bytes
            || prepared.dynamic_admission_bytes()? != dynamic_bytes
        {
            return Err(Error::State(
                "prepared GPU buffers differ from queued admission".into(),
            ));
        }
        // LRU eviction of other, idle source tensors is complete before H2D.
        // Reserve the selected source AND scratch/output as one local unit.
        resident.ensure_source_and_scratch_room(&key, bytes, dynamic_bytes, output_bytes)?;

        // Only now may actual GPU commands be launched. Any error/future drop
        // from here conservatively fences the physical card and retains the
        // real source/dynamic/output buffers in P07-F quarantine.
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let fence_flag = self.service.fence_flag();
        let weight = resident.get_or_try_upload_with_scratch(
            key,
            bytes,
            dynamic_bytes,
            output_bytes,
            || {
                let (host, gpu_source) = staged_source.take().ok_or_else(|| {
                    Error::State("missing prepared host source for GPU cache miss".into())
                })?;
                let handle = client.create_from_slice(&gpu_source);
                Ok(ResidentWeightLease {
                    client: client.clone(),
                    fence_flag,
                    handle,
                    bytes: host.gpu_bytes,
                    elements: host.gpu_elements,
                    host,
                })
            },
        )?;
        // Clone the already cached immutable source handle before checking
        // the combined quota. The cache borrow may then end, while the lease
        // and source metadata remain valid under the owning async lock.
        let pinned = ResidentHandle {
            handle: weight.handle.clone(),
            bytes: weight.bytes,
            elements: weight.elements,
            rows: weight.host.rows,
            kind: weight.host.kind,
        };
        resident.ensure_weight_and_scratch_budget(dynamic_bytes, output_bytes)?;
        // Admission occurs before kernel launch and binds this readback to
        // precisely the admitted dynamic and output buffer footprint.
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        // A cancelled future drops the guard and permanently fences the
        // cached source weights. Actual backend completion must be observed.
        let ready = prepared
            .execute(&client, &pinned, &mut in_flight, self.numeric_mode)
            .await?;
        if ready.dynamic_admitted_bytes != dynamic_bytes
            || ready.dynamic_h2d_bytes != prepared.dynamic_upload_bytes()?
            || ready.completed_d2h_bytes != output_bytes
        {
            return Err(Error::State(
                "GPU physical byte receipt differs from admitted leases".into(),
            ));
        }
        in_flight.complete_with_upload(ready.completed_d2h_bytes, ready.dynamic_h2d_bytes)?;
        // Only genuine GPU readback + P07-E event ledger clearance may
        // return this shared physical memory envelope to reusable Ready.
        physical_work.complete()?;
        // A cancelled/stale queue epoch may NEVER publish a completed but
        // causally obsolete result into the one StateRoot coordinator.
        worker_slot.finish()?;
        if matches!(prepared, PreparedStep::Device(_)) {
            self.device_quantized_regions
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(RegionExecution::Completed(ready.values))
    }
}

impl CubeResidentProvider {
    /// True asynchronous physical RMSNorm region. The CPU/source normalization
    /// math stays authoritative, but both inverse-scale reduction and the
    /// pointwise application execute on ONE physical GPU stream, with no CPU
    /// readback between them. A real full-output readback gates StateRoot.
    pub async fn try_rms_norm_async(&self, region: RmsNormRegion<'_>) -> Result<RegionExecution> {
        self.try_grouped_rms_norm_async(GroupedRmsNormRegion {
            source_identity: region.source_identity,
            tensor_name: region.tensor_name,
            input: region.input,
            weights: region.weights,
            group_width: region.input.len(),
            groups: 1,
            eps: region.eps,
        })
        .await
    }

    /// One real physical dispatch/readback for multiple independent source
    /// RMS heads. Same buffer/queue/event/lease machinery as single RMSNorm;
    /// no second execution engine, hidden CPU norm or per-head GPU roundtrip.
    pub async fn try_grouped_rms_norm_async(
        &self,
        region: GroupedRmsNormRegion<'_>,
    ) -> Result<RegionExecution> {
        if !self.selected.contains(region.tensor_name) {
            return Ok(RegionExecution::Unsupported);
        }
        if !region.source_identity.is_bound()
            || region.tensor_name.is_empty()
            || region.input.is_empty()
            || region.group_width == 0
            || region.groups == 0
            || region.group_width > 16_384
            || region.groups > 4096
            || region.input.len() > 1_048_576
            || region.group_width.checked_mul(region.groups) != Some(region.input.len())
            || region.weights.len() != region.group_width
            || !region.eps.is_finite()
            || region.eps <= 0.0
            || !region
                .input
                .iter()
                .chain(region.weights)
                .all(|v| v.is_finite())
        {
            return Err(Error::Malformed(
                "invalid physical source RMS normalization region".into(),
            ));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;
        let rows = region.input.len();
        let width = region.group_width;
        let groups = region.groups;
        // Inputs + gamma are transient H2D, intermediate inverse is device
        // scratch, final output is separately charged D2H. All are nested
        // under the ALREADY reserved per-provider global physical envelope.
        let output_bytes = rows
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("GPU RMS output bytes overflow".into()))?;
        // One shared width-element gamma, but independently stored input,
        // FP32 squared values and two FP32 inverse values per head.
        let dynamic_bytes = rows
            .checked_mul(8)
            .and_then(|n| width.checked_mul(4).and_then(|w| n.checked_add(w)))
            .and_then(|n| groups.checked_mul(8).and_then(|g| n.checked_add(g)))
            .ok_or_else(|| Error::Limit("GPU RMS source/scratch bytes overflow".into()))?;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("GPU RMS transient bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes)
                        .map_err(|_| Error::Limit("GPU RMS result bytes overflow".into()))?,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        // Same serialization discipline as source-GEMV: no LRU victim can
        // be freed while a previous GPU source kernel still uses that slot.
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let source = client.create_from_slice(f32::as_bytes(region.input));
        let weights = client.create_from_slice(f32::as_bytes(region.weights));
        let squares = client.empty(output_bytes);
        let inverse = client.empty(groups * 8);
        let output = client.empty(output_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                source.clone(),
                weights.clone(),
                squares.clone(),
                inverse.clone(),
                output.clone(),
            ],
        )?;
        let count = u32::try_from(rows)
            .map_err(|_| Error::Limit("physical RMS grid rows overflow".into()))?;
        let head_count = u32::try_from(groups)
            .map_err(|_| Error::Limit("physical RMS GPU heads overflow".into()))?;
        // SAFETY: source/weight/inverse/output memory bounds were checked;
        // exactly one thread writes inverse[0] before the next queued kernel
        // reads it. Same CubeCL service guarantees ordered commands.
        unsafe {
            rms_norm::rms_squares::launch_unchecked(
                &client,
                CubeCount::Static(count.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(source.clone(), rows),
                BufferArg::from_raw_parts(squares.clone(), rows),
                rows,
            );
            rms_norm::rms_inverse::launch_unchecked(
                &client,
                CubeCount::Static(head_count.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(squares, rows),
                BufferArg::from_raw_parts(inverse.clone(), groups * 2),
                region.eps,
                width,
                groups,
            );
            rms_norm::rms_round_inverse::launch_unchecked(
                &client,
                CubeCount::Static(head_count.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(inverse.clone(), groups * 2),
                groups,
            );
            rms_norm::rms_apply::launch_unchecked(
                &client,
                CubeCount::Static(count.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(source, rows),
                BufferArg::from_raw_parts(weights, width),
                BufferArg::from_raw_parts(inverse.clone(), groups * 2),
                BufferArg::from_raw_parts(output.clone(), rows),
                width,
                groups,
            );
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;

        let mut completed = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("physical GPU RMS readback failed: {e}")))?;
        if completed.len() != 1 {
            return Err(Error::State(
                "physical GPU RMS output handle count wrong".into(),
            ));
        }
        let bytes = completed.remove(0);
        if bytes.len() != output_bytes {
            return Err(Error::State(
                "physical GPU RMS returned incomplete output".into(),
            ));
        }
        let actual = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("GPU RMS physical readback f32 stride checked"),
                )
            })
            .collect::<Vec<_>>();
        if !actual.iter().all(|value| value.is_finite()) {
            return Err(Error::State("physical GPU RMS output nonfinite".into()));
        }
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(RegionExecution::Completed(actual))
    }
}

impl CubeResidentProvider {
    /// GDN causal convolution with PRIVATE proposed 3-frame history.
    /// The CPU-owned candidate StateRoot is never changed until BOTH the raw
    /// pre-SiLU output and next history were physically read back and checked.
    /// Uses the same per-PCI queue, logical admission, RAII event/handle
    /// keepalives and fail-closed completion as source GEMV and grouped RMS.
    pub async fn try_gdn_conv_async(&self, region: GdnConvRegion<'_>) -> Result<GdnConvExecution> {
        if !self.selected.contains(region.tensor_name) {
            return Ok(GdnConvExecution::Unsupported);
        }
        let channels = region.activation.len();
        if !region.source_identity.is_bound()
            || region.tensor_name.is_empty()
            || channels == 0
            || channels > 16_384
            || channels.checked_mul(4) != Some(region.weights.len())
            || channels.checked_mul(3) != Some(region.history.len())
            || !region
                .activation
                .iter()
                .chain(region.weights.iter())
                .chain(region.history.iter())
                .all(|value| value.is_finite())
        {
            return Err(Error::Malformed(
                "physical Qwen GDN convolution invalid source identity/shape/f32 value".into(),
            ));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;

        // One activation, four immutable F32 conv coefficients and three
        // previous history values per source channel; the GPU returns one
        // scalar conv raw and three proposed next-history values.
        let raw_bytes = channels
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("GDN convolution output bytes overflow".into()))?;
        let history_bytes = channels
            .checked_mul(12)
            .ok_or_else(|| Error::Limit("GDN convolution history bytes overflow".into()))?;
        let dynamic_bytes = raw_bytes
            .checked_mul(12) // activation+weights+history+four rounded FP32 products
            .ok_or_else(|| Error::Limit("GDN convolution input/products overflow".into()))?;
        let output_bytes = raw_bytes
            .checked_add(history_bytes)
            .ok_or_else(|| Error::Limit("GDN convolution output footprint overflow".into()))?;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("GDN convolution input bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes).map_err(|_| {
                        Error::Limit("GDN convolution output bytes overflow".into())
                    })?,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;

        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let activation = client.create_from_slice(f32::as_bytes(region.activation));
        let weights = client.create_from_slice(f32::as_bytes(region.weights));
        let previous = client.create_from_slice(f32::as_bytes(region.history));
        let products = client.empty(channels * 16);
        let raw = client.empty(raw_bytes);
        let proposed = client.empty(history_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                activation.clone(),
                weights.clone(),
                previous.clone(),
                products.clone(),
                raw.clone(),
                proposed.clone(),
            ],
        )?;
        let grid = u32::try_from(channels)
            .map_err(|_| Error::Limit("GDN convolution GPU grid overflow".into()))?;
        // SAFETY: bounded original 1:4:3 source buffers and complete 1:3
        // output buffers are explicitly size checked; each GPU lane reads a
        // unique source channel and writes only its disjoint private result.
        unsafe {
            gdn_conv::gdn_conv_products::launch_unchecked(
                &client,
                CubeCount::Static(grid.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(activation, channels),
                BufferArg::from_raw_parts(weights, channels * 4),
                BufferArg::from_raw_parts(previous, channels * 3),
                BufferArg::from_raw_parts(products.clone(), channels * 4),
                BufferArg::from_raw_parts(proposed.clone(), channels * 3),
                channels,
            );
            for level in 0..3 {
                gdn_conv::gdn_conv_reduce::launch_unchecked(
                    &client,
                    CubeCount::Static(grid.div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(products.clone(), channels * 4),
                    BufferArg::from_raw_parts(raw.clone(), channels),
                    channels,
                    level,
                );
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![raw, proposed])
            .await
            .map_err(|e| Error::State(format!("GDN convolution physical readback failed: {e}")))?;
        if results.len() != 2 || results[0].len() != raw_bytes || results[1].len() != history_bytes
        {
            return Err(Error::State(
                "GDN convolution incomplete device output or next history".into(),
            ));
        }
        let decode = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .map(|part| f32::from_le_bytes(part.try_into().expect("validated GDN FP32 bytes")))
                .collect()
        };
        let ready = GdnConvCompleted {
            raw: decode(&results[0]),
            history: decode(&results[1]),
        };
        if !ready
            .raw
            .iter()
            .chain(ready.history.iter())
            .all(|v| v.is_finite())
        {
            return Err(Error::State("GDN physical output/history nonfinite".into()));
        }
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(GdnConvExecution::Completed(ready))
    }
}

impl CubeResidentProvider {
    pub async fn try_silu_async(&self, region: SiluRegion<'_>) -> Result<RegionExecution> {
        if !self.selected.contains(region.operation) {
            return Ok(RegionExecution::Unsupported);
        }
        region.validate()?;
        let channels = region.input.len();
        if !region.source_identity.is_bound()
            || region.operation.is_empty()
            || channels > 65_536
            || region.input.iter().any(|x| x.abs() > 256.0)
        {
            return Err(Error::Unsupported("physical source SiLU requires bound identity and <=65536 finite inputs in [-256,256]".into()));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;

        let raw_bytes = channels
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("SiLU byte overflow".into()))?;
        // Input, optional multiplier, ten physically stored source intermediates.
        let dynamic_bytes = raw_bytes
            .checked_mul(1 + usize::from(region.multiplier.is_some()) + crate::silu::SCRATCH)
            .ok_or_else(|| Error::Limit("SiLU scratch overflow".into()))?;
        let output_bytes = raw_bytes;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("SiLU input bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes)
                        .map_err(|_| Error::Limit("SiLU output bytes overflow".into()))?,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;

        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let input = client.create_from_slice(f32::as_bytes(region.input));
        let scratch = client.empty(channels * crate::silu::SCRATCH * 4);
        let output = client.empty(output_bytes);
        let multiplier = region
            .multiplier
            .map(|m| client.create_from_slice(f32::as_bytes(m)));
        let mut buffers = vec![input.clone(), scratch.clone(), output.clone()];
        buffers.extend(multiplier.iter().cloned());
        in_flight.retain_gpu_buffers(&client, &buffers)?;
        let grid =
            u32::try_from(channels).map_err(|_| Error::Limit("SiLU grid overflow".into()))?;
        // SAFETY: validated complete input/multiplier and ten FP32 scratch
        // slots per element; each lane owns its disjoint scratch/output.
        unsafe {
            for stage in 0..12 {
                crate::silu::silu_step::launch_unchecked(
                    &client,
                    CubeCount::Static(grid.div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(input.clone(), channels),
                    BufferArg::from_raw_parts(scratch.clone(), channels * crate::silu::SCRATCH),
                    BufferArg::from_raw_parts(output.clone(), channels),
                    channels,
                    stage,
                );
            }
            if let Some(m) = multiplier {
                crate::silu::silu_gate::launch_unchecked(
                    &client,
                    CubeCount::Static(grid.div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(m, channels),
                    BufferArg::from_raw_parts(output.clone(), channels),
                    channels,
                );
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("SiLU physical readback failed: {e}")))?;
        if results.len() != 1 || results[0].len() != output_bytes {
            return Err(Error::State("SiLU incomplete physical output".into()));
        }
        let ready: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().expect("checked SiLU FP32 bytes")))
            .collect();
        if !ready.iter().all(|x| x.is_finite()) {
            return Err(Error::State("nonfinite SiLU physical output".into()));
        }
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(RegionExecution::Completed(ready))
    }
}

impl CubeResidentProvider {
    pub async fn try_gdn_coefficients_async(
        &self,
        region: GdnCoefficientRegion<'_>,
    ) -> Result<GdnCoefficientExecution> {
        if !self.selected.contains(region.operation) {
            return Ok(GdnCoefficientExecution::Unsupported);
        }
        region.validate()?;
        let channels = region.alpha.len();
        if !region.source_identity.is_bound()
            || region.operation.is_empty()
            || channels > 65_536
            || region
                .alpha
                .iter()
                .chain(region.beta)
                .chain(region.dt)
                .any(|x| x.abs() > 128.0)
            || region.a.iter().any(|x| *x > 0.0 || x.abs() > 1_000_000.0)
        {
            return Err(Error::Unsupported("source GDN coefficients require bound identity, <=65536 heads, alpha/beta/dt in [-128,128], A in [-1e6,0]".into()));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;

        if !self
            .device
            .client()
            .properties()
            .features
            .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
        {
            return Err(Error::Unsupported(
                "source GDN exp scale requires shader Int64".into(),
            ));
        }
        let input_bytes = channels * 16;
        let table_bytes = crate::source_transcendentals::TABLES.len() * 8;
        let dynamic_bytes = input_bytes + table_bytes + channels * 24 + channels * 24;
        let output_bytes = channels * 16;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("GDN coefficient input bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes).map_err(|_| {
                        Error::Limit("GDN coefficient output bytes overflow".into())
                    })?,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;

        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let packed: Vec<f32> = (0..channels)
            .flat_map(|h| [region.alpha[h], region.beta[h], region.dt[h], region.a[h]])
            .collect();
        let input = client.create_from_slice(f32::as_bytes(&packed));
        let tables =
            client.create_from_slice(f64::as_bytes(&crate::source_transcendentals::TABLES));
        let scratch = client.empty(channels * 24);
        let shifted = client.empty(channels * 24);
        let output = client.empty(output_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                input.clone(),
                tables.clone(),
                scratch.clone(),
                shifted.clone(),
                output.clone(),
            ],
        )?;
        let grid = u32::try_from(channels)
            .map_err(|_| Error::Limit("GDN coefficient grid overflow".into()))?;
        // SAFETY: bounded complete 4-float inputs/output, 6-float scratch,
        // 3-double shifted values per head; every lane owns disjoint values.
        unsafe {
            for stage in 0..8 {
                crate::gdn_coefficients::coefficient_step::launch_unchecked(
                    &client,
                    CubeCount::Static(grid.div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(input.clone(), channels * 4),
                    BufferArg::from_raw_parts(
                        tables.clone(),
                        crate::source_transcendentals::TABLES.len(),
                    ),
                    BufferArg::from_raw_parts(scratch.clone(), channels * 6),
                    BufferArg::from_raw_parts(shifted.clone(), channels * 3),
                    BufferArg::from_raw_parts(output.clone(), channels * 4),
                    channels,
                    stage,
                );
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("GDN coefficient physical readback failed: {e}")))?;
        if results.len() != 1 || results[0].len() != output_bytes {
            return Err(Error::State(
                "GDN coefficient incomplete physical output".into(),
            ));
        }
        let values: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().expect("checked coefficient bytes")))
            .collect();
        let mut ready = GdnCoefficientCompleted {
            coefficients: Vec::with_capacity(channels),
            softplus: Vec::with_capacity(channels),
            gate_raw: Vec::with_capacity(channels),
        };
        for row in values.chunks_exact(4) {
            ready.softplus.push(row[0]);
            ready.gate_raw.push(row[1]);
            ready.coefficients.push(GdnCoefficients {
                gain: row[2],
                beta: row[3],
                scale: region.scale,
            });
        }
        ready.validate(channels)?;
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(GdnCoefficientExecution::Completed(ready))
    }
}

impl CubeResidentProvider {
    /// Diagnostic-only scalar source-input fixture. Never supplies model output.
    /// Uses the same physical queue/admission/retention/readback as model regions.
    #[cfg(feature = "qualification")]
    pub(crate) async fn qualify_softmax_reciprocals(&self) -> Result<()> {
        let mut inputs = vec![f64::from_bits(0x3ff0000018000024)];
        for e in 0..=12u64 {
            let bits = (1023 + e) << 52;
            for delta in [-1i64, 0, 1] {
                let value = f64::from_bits(bits.wrapping_add_signed(delta));
                if (1.0..=4096.0).contains(&value) {
                    inputs.push(value);
                }
            }
        }
        let count = inputs.len();
        let dynamic_bytes = count * 8;
        let output_bytes = count * 12;
        let client = self.device.client();
        source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
        if !client
            .properties()
            .features
            .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
        {
            return Err(Error::Unsupported(
                "reciprocal fixture requires UInt64".into(),
            ));
        }
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let input = client.create_from_slice(f64::as_bytes(&inputs));
        let reciprocal = client.empty(count * 8);
        let inverse = client.empty(count * 4);
        in_flight.retain_gpu_buffers(
            &client,
            &[input.clone(), reciprocal.clone(), inverse.clone()],
        )?;
        let grid = CubeCount::Static((count as u32).div_ceil(32), 1, 1);
        // SAFETY: all source sums are finite in1..4096; complete bounded buffers.
        unsafe {
            crate::attention_softmax::reciprocal::launch_unchecked(
                &client,
                grid.clone(),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input, count),
                BufferArg::from_raw_parts(reciprocal.clone(), count),
                count,
            );
            crate::attention_softmax::inverses::launch_unchecked(
                &client,
                grid,
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(reciprocal.clone(), count),
                BufferArg::from_raw_parts(inverse.clone(), count),
                count,
            );
        }
        let bytes = client
            .read_async(vec![reciprocal, inverse])
            .await
            .map_err(|e| Error::State(format!("reciprocal fixture readback failed: {e}")))?;
        if bytes.len() != 2 || bytes[0].len() != count * 8 || bytes[1].len() != count * 4 {
            return Err(Error::State("reciprocal fixture incomplete cohort".into()));
        }
        for (i, &sum) in inputs.iter().enumerate() {
            let q = f64::from_le_bytes(
                bytes[0][i * 8..i * 8 + 8]
                    .try_into()
                    .expect("checked double"),
            );
            let inv = f32::from_le_bytes(
                bytes[1][i * 4..i * 4 + 4]
                    .try_into()
                    .expect("checked single"),
            );
            let expected = 1.0f64 / sum;
            if q.to_bits() != expected.to_bits() || inv.to_bits() != (expected as f32).to_bits() {
                return Err(Error::State(format!(
                    "source reciprocal physical bits mismatch fixture{i}"
                )));
            }
        }
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        println!(
            "P07_H11_PHYSICAL_TWO_ROUND_RECIPROCAL_PASS fixtures={count} counter_double=3fefffffd0000000 counter_single=3f7ffffe exact_bits=1 scalar_fixture_not_model_context=1"
        );
        Ok(())
    }

    pub async fn try_attention_softmax_async(
        &self,
        r: AttentionSoftmaxRegion<'_>,
    ) -> Result<AttentionSoftmaxExecution> {
        if !self.selected.contains(r.operation) {
            return Ok(AttentionSoftmaxExecution::Unsupported);
        }
        r.validate()?;
        if !r.source_identity.is_bound() || r.operation.is_empty() || r.heads > 64 || r.width > 4096
        {
            return Err(Error::Unsupported(
                "source softmax requires bound identity, <=64 heads and <=4096 valid scores".into(),
            ));
        }
        let client = self.device.client();
        source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
        if !client
            .properties()
            .features
            .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
        {
            return Err(Error::Unsupported(
                "exact source softmax reciprocal requires UInt64".into(),
            ));
        }
        let n = r.scores.len();
        let h = r.heads;
        let groups = r.width.div_ceil(8);
        let dynamic_bytes = n * 64 + h * 4 + h * groups * 24;
        let output_bytes = n * 8 + h * 20;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let input = client.create_from_slice(f32::as_bytes(r.scores));
        let scratch = client.empty(n * crate::attention_softmax::SCRATCH * 4);
        let candidate = client.empty(n * 4);
        let sums = client.empty(n * 8);
        let subtracted = client.empty(n * 8);
        let maximum = client.empty(h * 4);
        let a = client.empty(h * groups * 16);
        let b = client.empty(h * groups * 8);
        let exp = client.empty(n * 4);
        let prob = client.empty(n * 4);
        let total = client.empty(h * 8);
        let reciprocal = client.empty(h * 8);
        let inverse = client.empty(h * 4);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                input.clone(),
                scratch.clone(),
                candidate.clone(),
                sums.clone(),
                subtracted.clone(),
                maximum.clone(),
                a.clone(),
                b.clone(),
                exp.clone(),
                prob.clone(),
                total.clone(),
                reciprocal.clone(),
                inverse.clone(),
            ],
        )?;
        let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        use crate::attention_softmax as sm;
        // SAFETY: validated shape caps bound every byte/offset/grid. Every
        // rounded FMA candidate, sum and subtraction crosses original buffers.
        unsafe {
            sm::maxima::launch_unchecked(
                &client,
                grid(h),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input.clone(), n),
                BufferArg::from_raw_parts(maximum.clone(), h),
                h,
                r.width,
            );
            for stage in 0..2 {
                sm::prepare::launch_unchecked(
                    &client,
                    grid(n),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(input.clone(), n),
                    BufferArg::from_raw_parts(maximum.clone(), h),
                    BufferArg::from_raw_parts(scratch.clone(), n * sm::SCRATCH),
                    n,
                    r.width,
                    stage,
                );
            }
            for step in 0..8 {
                if step == 1 || step == 3 {
                    sm::prepare::launch_unchecked(
                        &client,
                        grid(n),
                        CubeDim::new_1d(32),
                        BufferArg::from_raw_parts(input.clone(), n),
                        BufferArg::from_raw_parts(maximum.clone(), h),
                        BufferArg::from_raw_parts(scratch.clone(), n * sm::SCRATCH),
                        n,
                        r.width,
                        if step == 1 { 2 } else { 3 },
                    );
                }
                for stage in 0..3 {
                    sm::polynomial::launch_unchecked(
                        &client,
                        grid(n),
                        CubeDim::new_1d(32),
                        BufferArg::from_raw_parts(scratch.clone(), n * sm::SCRATCH),
                        BufferArg::from_raw_parts(candidate.clone(), n),
                        BufferArg::from_raw_parts(sums.clone(), n),
                        BufferArg::from_raw_parts(subtracted.clone(), n),
                        n,
                        step,
                        stage,
                    );
                }
            }
            sm::exponentials::launch_unchecked(
                &client,
                grid(n),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(scratch.clone(), n * sm::SCRATCH),
                BufferArg::from_raw_parts(exp.clone(), n),
                n,
            );
            for level in 0..3 {
                let (source, dest, source_len, dest_len) = if level == 0 {
                    (exp.clone(), a.clone(), n, h * groups * 4)
                } else if level == 1 {
                    (a.clone(), b.clone(), h * groups * 4, h * groups * 2)
                } else {
                    (b.clone(), a.clone(), h * groups * 2, h * groups * 4)
                };
                sm::reduce::launch_unchecked(
                    &client,
                    grid(h * groups * (4 >> level)),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(source, source_len),
                    BufferArg::from_raw_parts(dest, dest_len),
                    h,
                    r.width,
                    level,
                );
            }
            sm::totals::launch_unchecked(
                &client,
                grid(h),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(a.clone(), h * groups * 4),
                BufferArg::from_raw_parts(total.clone(), h),
                h,
                groups,
            );
            sm::reciprocal::launch_unchecked(
                &client,
                grid(h),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(total.clone(), h),
                BufferArg::from_raw_parts(reciprocal.clone(), h),
                h,
            );
            sm::inverses::launch_unchecked(
                &client,
                grid(h),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(reciprocal.clone(), h),
                BufferArg::from_raw_parts(inverse.clone(), h),
                h,
            );
            sm::probabilities::launch_unchecked(
                &client,
                grid(n),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(exp.clone(), n),
                BufferArg::from_raw_parts(inverse.clone(), h),
                BufferArg::from_raw_parts(prob.clone(), n),
                n,
                r.width,
            );
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let bytes = client
            .read_async(vec![exp, prob, total, reciprocal, inverse])
            .await
            .map_err(|e| Error::State(format!("softmax physical readback failed: {e}")))?;
        if bytes.len() != 5
            || bytes
                .iter()
                .map(|b| b.len())
                .ne([n * 4, n * 4, h * 8, h * 8, h * 4])
        {
            return Err(Error::State("incomplete softmax readback cohort".into()));
        }
        let floats = |i: usize| {
            bytes[i]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("checked softmax bytes")))
                .collect()
        };
        let doubles = |i: usize| {
            bytes[i]
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().expect("checked softmax double bytes")))
                .collect()
        };
        let ready = AttentionSoftmaxCompleted {
            exponentials: floats(0),
            probabilities: floats(1),
            sums: doubles(2),
            reciprocals: doubles(3),
            inverse: floats(4),
        };
        ready.validate(r)?;
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(AttentionSoftmaxExecution::Completed(ready))
    }

    pub async fn try_attention_value_async(
        &self,
        region: AttentionValueRegion<'_>,
    ) -> Result<RegionExecution> {
        if !self.selected.contains(region.operation) {
            return Ok(RegionExecution::Unsupported);
        }
        region.validate()?;
        if !region.source_identity.is_bound()
            || region.operation.is_empty()
            || !region.head_width.is_multiple_of(32)
            || region.head_width > 512
            || region.query_heads > 64
            || region.kv_heads > 64
            || region.values.len() > 4096
            || region.padded_tokens != region.values.len().div_ceil(256) * 256
        {
            return Err(Error::Unsupported("source value requires bound identity, width 32..512 multiple32, <=64 heads, <=4096 prefix and canonical 256-slot padding".into()));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;
        let tokens = region.values.len();
        let columns = region.query_heads * region.head_width;
        let lane_count = columns * 32;
        let channels = region.probabilities.len() + tokens * region.kv_heads * region.head_width;
        let dynamic_bytes = channels * 4 + lane_count * 24;
        let output_bytes = columns * 4;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        let packed: Vec<f32> = region
            .probabilities
            .iter()
            .copied()
            .chain(region.values.iter().flat_map(|key| key.iter().copied()))
            .collect();
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let input = client.create_from_slice(f32::as_bytes(&packed));
        let lanes = [client.empty(lane_count * 4), client.empty(lane_count * 4)];
        let sums = client.empty(lane_count * 8);
        let subtracted = client.empty(lane_count * 8);
        let output = client.empty(output_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                input.clone(),
                lanes[0].clone(),
                lanes[1].clone(),
                sums.clone(),
                subtracted.clone(),
                output.clone(),
            ],
        )?;
        let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        // SAFETY: selected shape caps bound all offsets/bytes and dispatches.
        // Old accumulator and proposed FP32 lanes never alias within a block.
        unsafe {
            for block in 0..region.padded_tokens / 32 {
                for stage in 0..3 {
                    crate::attention_value::accumulate::launch_unchecked(
                        &client,
                        grid(lane_count),
                        CubeDim::new_1d(32),
                        BufferArg::from_raw_parts(input.clone(), channels),
                        BufferArg::from_raw_parts(lanes[(block + 1) % 2].clone(), lane_count),
                        BufferArg::from_raw_parts(lanes[block % 2].clone(), lane_count),
                        BufferArg::from_raw_parts(sums.clone(), lane_count),
                        BufferArg::from_raw_parts(subtracted.clone(), lane_count),
                        block as u32,
                        region.head_width,
                        tokens,
                        region.query_heads,
                        region.kv_heads,
                        stage,
                    );
                }
            }
            let last = (region.padded_tokens / 32 - 1) % 2;
            for level in 0..5 {
                let src = lanes[(last + level) % 2].clone();
                let dest = if level == 4 {
                    output.clone()
                } else {
                    lanes[(last + level + 1) % 2].clone()
                };
                crate::attention_qk::reduce::launch_unchecked(
                    &client,
                    grid(columns * (16 >> level)),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(src, lane_count),
                    BufferArg::from_raw_parts(dest, if level == 4 { columns } else { lane_count }),
                    columns,
                    level,
                );
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("value physical readback failed: {e}")))?;
        if results.len() != 1 || results[0].len() != output_bytes {
            return Err(Error::State("incomplete physical value readback".into()));
        }
        let values: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().expect("checked value bytes")))
            .collect();
        region.validate_completed(&values)?;
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(RegionExecution::Completed(values))
    }
    pub async fn try_attention_qk_async(
        &self,
        region: AttentionQkRegion<'_>,
    ) -> Result<AttentionQkExecution> {
        if !self.selected.contains(region.operation) {
            return Ok(AttentionQkExecution::Unsupported);
        }
        region.validate()?;
        if !region.source_identity.is_bound()
            || region.operation.is_empty()
            || !region.head_width.is_multiple_of(32)
            || region.head_width > 512
            || region.query_heads > 64
            || region.kv_heads > 64
            || region.keys.len() > 4096
        {
            return Err(Error::Unsupported("source QK requires bound identity, width 32..512 multiple32, <=64 heads and <=4096 prefix".into()));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;
        let tokens = region.keys.len();
        let columns = region.query_heads * tokens;
        let lane_count = columns * 32;
        let channels = region.query.len() + tokens * region.kv_heads * region.head_width;
        let dynamic_bytes = channels * 4 + lane_count * 24;
        let output_bytes = columns * 8;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        let packed: Vec<f32> = region
            .query
            .iter()
            .copied()
            .chain(region.keys.iter().flat_map(|key| key.iter().copied()))
            .collect();
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let input = client.create_from_slice(f32::as_bytes(&packed));
        let lanes = [client.empty(lane_count * 4), client.empty(lane_count * 4)];
        let sums = client.empty(lane_count * 8);
        let subtracted = client.empty(lane_count * 8);
        let output = client.empty(output_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                input.clone(),
                lanes[0].clone(),
                lanes[1].clone(),
                sums.clone(),
                subtracted.clone(),
                output.clone(),
            ],
        )?;
        let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        // SAFETY: selected shape caps bound all offsets/bytes and dispatches.
        // Old accumulator and proposed FP32 lanes never alias within a block.
        unsafe {
            for block in 0..region.head_width / 32 {
                for stage in 0..3 {
                    crate::attention_qk::accumulate::launch_unchecked(
                        &client,
                        grid(lane_count),
                        CubeDim::new_1d(32),
                        BufferArg::from_raw_parts(input.clone(), channels),
                        BufferArg::from_raw_parts(lanes[(block + 1) % 2].clone(), lane_count),
                        BufferArg::from_raw_parts(lanes[block % 2].clone(), lane_count),
                        BufferArg::from_raw_parts(sums.clone(), lane_count),
                        BufferArg::from_raw_parts(subtracted.clone(), lane_count),
                        region.head_width,
                        tokens,
                        region.query_heads,
                        region.kv_heads,
                        block,
                        stage,
                    );
                }
            }
            let last = (region.head_width / 32 - 1) % 2;
            for level in 0..5 {
                let src = lanes[(last + level) % 2].clone();
                let dest = if level == 4 {
                    output.clone()
                } else {
                    lanes[(last + level + 1) % 2].clone()
                };
                crate::attention_qk::reduce::launch_unchecked(
                    &client,
                    grid(columns * (16 >> level)),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(src, lane_count),
                    BufferArg::from_raw_parts(
                        dest,
                        if level == 4 { columns * 2 } else { lane_count },
                    ),
                    columns,
                    level,
                );
            }
            crate::attention_qk::scale::launch_unchecked(
                &client,
                grid(columns),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(output.clone(), columns * 2),
                region.scale,
                columns,
            );
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("QK physical readback failed: {e}")))?;
        if results.len() != 1 || results[0].len() != output_bytes {
            return Err(Error::State("incomplete physical QK readback".into()));
        }
        let mut values: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().expect("checked QK bytes")))
            .collect();
        let scores = values.split_off(columns);
        let ready = AttentionQkCompleted {
            dots: values,
            scores,
        };
        ready.validate(region)?;
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(AttentionQkExecution::Completed(ready))
    }

    pub async fn try_rope_kv_async(&self, region: RopeKvRegion<'_>) -> Result<RopeKvExecution> {
        if !self.selected.contains(region.operation) {
            return Ok(RopeKvExecution::Unsupported);
        }
        region.validate()?;
        if !region.source_identity.is_bound()
            || region.operation.is_empty()
            || region.position > 32_768
            || region.head_width > 512
            || region.rotate > 64
            || region.query_heads > 64
            || region.kv_heads > 64
        {
            return Err(Error::Unsupported("source RoPE/KV requires bound identity, <=32K position, <=512 head width, <=64 rotated values and <=64 Q/KV heads".into()));
        }
        let heads = region.query_heads + region.kv_heads;
        let half = region.rotate / 2;
        let channels = region.query.len() + region.key.len() + region.value.len();
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;

        for kind in [
            cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64),
            cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I64),
        ] {
            if !self
                .device
                .client()
                .properties()
                .features
                .supports_type(kind)
            {
                return Err(Error::Unsupported(
                    "source RoPE trig reduction requires shader Int64".into(),
                ));
            }
        }
        let dynamic_bytes = channels * 4
            + half * 4
            + half * 8
            + heads * half * 8
            + crate::source_trig::TRIG_TABLES.len() * 8;
        let dynamic_bytes = dynamic_bytes + heads * half * 32;
        let output_bytes = channels * 4;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("RoPE/KV input bytes overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes)
                        .map_err(|_| Error::Limit("RoPE/KV output bytes overflow".into()))?,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;

        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let packed: Vec<f32> = region
            .query
            .iter()
            .chain(region.key)
            .chain(region.value)
            .copied()
            .collect();
        let input = client.create_from_slice(f32::as_bytes(&packed));
        let tables = client.create_from_slice(f64::as_bytes(&crate::source_trig::TRIG_TABLES));
        let angles = client.empty(half * 4);
        let trig = client.empty(half * 8);
        let products = client.empty(heads * half * 8);
        let sums = client.empty(heads * half * 16);
        let subtracted = client.empty(heads * half * 16);
        let output = client.empty(output_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                input.clone(),
                tables.clone(),
                angles.clone(),
                trig.clone(),
                products.clone(),
                sums.clone(),
                subtracted.clone(),
                output.clone(),
            ],
        )?;
        let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        // SAFETY: bounded complete Q/K/V inputs/output and disjoint angle,
        // trig/product buffers; the current-token slab never aliases a prefix.
        unsafe {
            for stage in 0..half {
                crate::rope_kv::angle_step::launch_unchecked(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(angles.clone(), half),
                    region.position as f32,
                    region.theta_scale,
                    stage,
                );
            }
            crate::source_trig::sincos_angles::launch_unchecked(
                &client,
                grid(half),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(angles, half),
                BufferArg::from_raw_parts(tables, crate::source_trig::TRIG_TABLES.len()),
                BufferArg::from_raw_parts(trig.clone(), half * 2),
                half,
            );
            crate::rope_kv::rope_products::launch_unchecked(
                &client,
                grid(heads * half),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input.clone(), channels),
                BufferArg::from_raw_parts(trig.clone(), half * 2),
                BufferArg::from_raw_parts(products.clone(), heads * half * 2),
                region.head_width,
                half,
                heads,
                region.query_heads,
                1,
            );
            crate::rope_kv::rope_sums::launch_unchecked(
                &client,
                grid(channels),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input.clone(), channels),
                BufferArg::from_raw_parts(trig.clone(), half * 2),
                BufferArg::from_raw_parts(products.clone(), heads * half * 2),
                BufferArg::from_raw_parts(sums.clone(), heads * half * 2),
                BufferArg::from_raw_parts(output.clone(), channels),
                region.head_width,
                half,
                heads,
                channels,
                region.query_heads,
                1,
            );
            crate::rope_kv::rope_subtract::launch_unchecked(
                &client,
                grid(heads * half * 2),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input.clone(), channels),
                BufferArg::from_raw_parts(trig.clone(), half * 2),
                BufferArg::from_raw_parts(products.clone(), heads * half * 2),
                BufferArg::from_raw_parts(sums.clone(), heads * half * 2),
                BufferArg::from_raw_parts(subtracted.clone(), heads * half * 2),
                region.head_width,
                half,
                heads,
                region.query_heads,
                1,
            );
            crate::rope_kv::rope_apply::launch_unchecked(
                &client,
                grid(heads * half * 2),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input, channels),
                BufferArg::from_raw_parts(trig, half * 2),
                BufferArg::from_raw_parts(products, heads * half * 2),
                BufferArg::from_raw_parts(sums, heads * half * 2),
                BufferArg::from_raw_parts(subtracted, heads * half * 2),
                BufferArg::from_raw_parts(output.clone(), channels),
                region.head_width,
                half,
                heads,
                region.query_heads,
                1,
            );
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("RoPE/KV physical readback failed: {e}")))?;
        if results.len() != 1 || results[0].len() != output_bytes {
            return Err(Error::State("RoPE/KV incomplete physical output".into()));
        }
        let values: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().expect("checked RoPE/KV bytes")))
            .collect();
        let qend = region.query.len();
        let kend = qend + region.key.len();
        let ready = RopeKvCompleted {
            query: values[..qend].to_vec(),
            key: values[qend..kend].to_vec(),
            value: values[kend..].to_vec(),
        };
        ready.validate(region)?;
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(RopeKvExecution::Completed(ready))
    }
}

/// The two GDN dots use the same source schedule and transient FP32 tree.
struct RecurrentDot<'a> {
    state: &'a Handle,
    qkv: &'a Handle,
    lanes: &'a Handle,
    reduced: &'a Handle,
    output: &'a Handle,
    width: usize,
    values: usize,
    vector_offset: usize,
}

fn launch_recurrent_dot(client: &Client, dot: RecurrentDot<'_>) {
    use crate::gdn_recurrent as gdn;
    let RecurrentDot {
        state,
        qkv,
        lanes,
        reduced,
        output,
        width,
        values,
        vector_offset,
    } = dot;
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    // SAFETY: bounded caller-validated source state/QKV, 32/16 lane scratch,
    // disjoint output columns. Each level reads the previous physical FP32 buffer.
    unsafe {
        for block in 0..width / 32 {
            gdn::dot_lanes::launch_unchecked(
                client,
                grid(values * 32),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(state.clone(), values * width),
                BufferArg::from_raw_parts(qkv.clone(), values * 3),
                BufferArg::from_raw_parts(lanes.clone(), values * 32),
                width,
                values,
                vector_offset,
                block,
            );
        }
        for level in 0..5 {
            let input = if level % 2 == 0 { lanes } else { reduced };
            let dest = if level == 4 {
                output
            } else if level % 2 == 0 {
                reduced
            } else {
                lanes
            };
            let in_len = if level % 2 == 0 {
                values * 32
            } else {
                values * 16
            };
            let out_len = if level == 4 {
                values
            } else if level % 2 == 0 {
                values * 16
            } else {
                values * 32
            };
            gdn::dot_reduce::launch_unchecked(
                client,
                grid(values * (16 >> level)),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input.clone(), in_len),
                BufferArg::from_raw_parts(dest.clone(), out_len),
                values,
                level,
            );
        }
    }
}

impl CubeResidentProvider {
    pub async fn try_gdn_recurrent_async(
        &self,
        region: GdnRecurrentRegion<'_>,
    ) -> Result<GdnRecurrentExecution> {
        if !self.selected.contains(region.operation) {
            return Ok(GdnRecurrentExecution::Unsupported);
        }
        region.validate()?;
        if !region.source_identity.is_bound() || region.operation.is_empty() {
            return Err(Error::Malformed("unbound GDN recurrent source".into()));
        }
        let d = region.width;
        let heads = region.coefficients.len();
        if !d.is_multiple_of(32) || d > 256 || heads > 64 {
            return Ok(GdnRecurrentExecution::Unsupported);
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;
        let n = region.q.len();
        let m = region.state.len();
        let state_bytes = m
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("GDN state bytes overflow".into()))?;
        let value_bytes = n
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("GDN output bytes overflow".into()))?;
        // Previous + decayed state, packed Q/K/V, three coefficients/head,
        // delta and 32/16-lane FP32 reduction scratch. The proposal and raw
        // output form one readback cohort. The shape bounds above make all
        // subsequent byte/grid arithmetic fit usize/u32 even on 32-bit hosts.
        let dynamic_bytes = state_bytes * 2 + value_bytes * (4 + 32 + 16) + heads * 12;
        let output_bytes = state_bytes + value_bytes;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        let qkv_values: Vec<f32> = region
            .q
            .iter()
            .chain(region.k)
            .chain(region.v)
            .copied()
            .collect();
        let coefficient_values: Vec<f32> = region
            .coefficients
            .iter()
            .flat_map(|c| [c.gain, c.beta, c.scale])
            .collect();
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let qkv = client.create_from_slice(f32::as_bytes(&qkv_values));
        let coefficients = client.create_from_slice(f32::as_bytes(&coefficient_values));
        let previous = client.create_from_slice(f32::as_bytes(region.state));
        let decayed = client.empty(state_bytes);
        let delta = client.empty(value_bytes);
        let proposed = client.empty(state_bytes);
        let output = client.empty(value_bytes);
        let lanes = client.empty(value_bytes * 32);
        let reduced = client.empty(value_bytes * 16);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                qkv.clone(),
                coefficients.clone(),
                previous.clone(),
                decayed.clone(),
                delta.clone(),
                proposed.clone(),
                output.clone(),
                lanes.clone(),
                reduced.clone(),
            ],
        )?;
        let state_grid = CubeCount::Static((m as u32).div_ceil(32), 1, 1);
        let value_grid = CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        use crate::gdn_recurrent as gdn;
        // SAFETY: validate checked all head/vector/state products. Device
        // guards bound each disjoint lane. Width<=256 and heads<=64 bound
        // all grids and buffer lengths; all nine handles live to readback.
        unsafe {
            gdn::decay::launch_unchecked(
                &client,
                state_grid.clone(),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(previous, m),
                BufferArg::from_raw_parts(coefficients.clone(), heads * 3),
                BufferArg::from_raw_parts(decayed.clone(), m),
                d,
                m,
            );
            launch_recurrent_dot(
                &client,
                RecurrentDot {
                    state: &decayed,
                    qkv: &qkv,
                    lanes: &lanes,
                    reduced: &reduced,
                    output: &delta,
                    width: d,
                    values: n,
                    vector_offset: n,
                },
            );
            gdn::residual::launch_unchecked(
                &client,
                value_grid.clone(),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(qkv.clone(), n * 3),
                BufferArg::from_raw_parts(delta.clone(), n),
                n,
            );
            gdn::beta::launch_unchecked(
                &client,
                value_grid.clone(),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(coefficients.clone(), heads * 3),
                BufferArg::from_raw_parts(delta.clone(), n),
                d,
                n,
            );
            gdn::update::launch_unchecked(
                &client,
                state_grid,
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(decayed, m),
                BufferArg::from_raw_parts(qkv.clone(), n * 3),
                BufferArg::from_raw_parts(delta, n),
                BufferArg::from_raw_parts(proposed.clone(), m),
                d,
                n,
                m,
            );
            launch_recurrent_dot(
                &client,
                RecurrentDot {
                    state: &proposed,
                    qkv: &qkv,
                    lanes: &lanes,
                    reduced: &reduced,
                    output: &output,
                    width: d,
                    values: n,
                    vector_offset: 0,
                },
            );
            gdn::scale::launch_unchecked(
                &client,
                value_grid,
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(coefficients, heads * 3),
                BufferArg::from_raw_parts(output.clone(), n),
                d,
                n,
            );
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output, proposed])
            .await
            .map_err(|e| Error::State(format!("GDN recurrent physical readback failed: {e}")))?;
        if results.len() != 2 || results[0].len() != value_bytes || results[1].len() != state_bytes
        {
            return Err(Error::State(
                "incomplete GDN recurrent physical readback cohort".into(),
            ));
        }
        let decode = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("checked FP32 stride")))
                .collect()
        };
        let ready = GdnRecurrentCompleted {
            output: decode(&results[0]),
            state: decode(&results[1]),
        };
        if !ready
            .output
            .iter()
            .chain(&ready.state)
            .all(|v| v.is_finite())
        {
            return Err(Error::State(
                "nonfinite GDN recurrent device proposal".into(),
            ));
        }
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(GdnRecurrentExecution::Completed(ready))
    }
}

impl MatVecKernel for CubeResidentProvider {
    fn try_binary(
        &self,
        r: zerogravity_core::accelerator::BinaryRegion<'_>,
    ) -> Result<RegionExecution> {
        futures_lite::future::block_on(self.try_binary_async(r))
    }
    fn try_embedding(
        &self,
        r: zerogravity_core::accelerator::EmbeddingRegion<'_>,
    ) -> Result<RegionExecution> {
        futures_lite::future::block_on(self.try_embedding_async(r))
    }

    fn try_attention_value(&self, r: AttentionValueRegion<'_>) -> Result<RegionExecution> {
        futures_lite::future::block_on(self.try_attention_value_async(r))
    }

    fn try_attention_softmax(
        &self,
        r: AttentionSoftmaxRegion<'_>,
    ) -> Result<AttentionSoftmaxExecution> {
        futures_lite::future::block_on(self.try_attention_softmax_async(r))
    }

    fn try_attention_qk(&self, region: AttentionQkRegion<'_>) -> Result<AttentionQkExecution> {
        futures_lite::future::block_on(self.try_attention_qk_async(region))
    }
    fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
        // Legacy task-facing API stays source-compatible; the same actual
        // CubeCL read_async/device-fence path is driven to completion here.
        futures_lite::future::block_on(self.try_execute_async(region))
    }

    fn try_rms_norm(&self, region: RmsNormRegion<'_>) -> Result<RegionExecution> {
        futures_lite::future::block_on(self.try_rms_norm_async(region))
    }

    fn try_grouped_rms_norm(&self, region: GroupedRmsNormRegion<'_>) -> Result<RegionExecution> {
        futures_lite::future::block_on(self.try_grouped_rms_norm_async(region))
    }

    fn try_gdn_recurrent(&self, region: GdnRecurrentRegion<'_>) -> Result<GdnRecurrentExecution> {
        futures_lite::future::block_on(self.try_gdn_recurrent_async(region))
    }

    fn try_rope_kv(&self, region: RopeKvRegion<'_>) -> Result<RopeKvExecution> {
        futures_lite::future::block_on(self.try_rope_kv_async(region))
    }

    fn try_gdn_coefficients(
        &self,
        region: GdnCoefficientRegion<'_>,
    ) -> Result<GdnCoefficientExecution> {
        futures_lite::future::block_on(self.try_gdn_coefficients_async(region))
    }

    fn try_silu(&self, region: SiluRegion<'_>) -> Result<RegionExecution> {
        futures_lite::future::block_on(self.try_silu_async(region))
    }

    fn try_gdn_conv(&self, region: GdnConvRegion<'_>) -> Result<GdnConvExecution> {
        futures_lite::future::block_on(self.try_gdn_conv_async(region))
    }
}
