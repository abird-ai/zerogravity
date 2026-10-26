//! Original full-model BF16 head consumer, separate from legacy F32 total_cmp.
//! No state/root publication, sampling policy, emission or Ready authority.
use super::BufferRange;
use crate::cuda::native::model_values::ModelValues;
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentReadback, ResidentService,
};
use crate::resident_program::source_bindings::SourceBindings;
use cubecl::client::Client;
use cubecl::{bytes::Bytes, prelude::*};
use cubecl_common::device::ServiceId;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    ActivationPrecision, ProgramNumericalProfile, SourceCall, SourceOperatorKind, ValueId,
    ValueShape,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

mod kernels;
#[cfg(test)]
mod tests;

const METADATA_BYTES: usize = 64 * 1024;
const LOGITS_BYTES: usize = kernels::VOCABULARY * 2;
const PARTIAL_BYTES: usize = kernels::PARTIAL_WORDS * 4;
const PACKET_BYTES: usize = 12;

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("BF16 greedy cancelled before enqueue".into()));
    }
    Ok(())
}

/// Only core's real full Head result can bind this output consumer.
fn require_head(record: &SourceBindings) -> Result<()> {
    let plan = record.plan();
    let call = record.call();
    let id = record.selection().output();
    if plan.source() != call.source()
        || plan.source_call() != Some(call)
        || plan.numerical_profile() != ProgramNumericalProfile::SourceCudaSm86Bf16V1
        || plan.is_operator_slice()
        || plan.frontier_delta() != 0
        || plan.logits().is_some()
        || plan.required_domains().len() != 42
        || plan.proposed_domains().len() != 42
        || record.selection().completed_layers() != 24
        || plan.values().get(id.0 as usize) != Some(&ValueShape::Vector(kernels::VOCABULARY))
        || plan.precisions().get(id.0 as usize) != Some(&ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::Unsupported(
            "BF16 greedy requires unpublished full24 BF16 head".into(),
        ));
    }
    let origin = plan
        .source_value_provenance(id)?
        .ok_or_else(|| Error::State("BF16 greedy actual Head SSA origin absent".into()))?;
    let node = plan
        .nodes()
        .get(origin.node_index())
        .ok_or_else(|| Error::State("BF16 greedy Head node absent".into()))?;
    let descriptor = record
        .operator(origin.node_index())?
        .ok_or_else(|| Error::State("BF16 greedy Head descriptor absent".into()))?;
    descriptor.validate()?;
    let selected = record
        .selection()
        .selected_row()
        .ok_or_else(|| Error::State("BF16 greedy selected head row absent".into()))?;
    if origin.call() != call
        || origin.output_index() != 0
        || descriptor.call() != call
        || descriptor.kind()
            != (SourceOperatorKind::Projection {
                selected_row: Some(selected),
            })
        || !matches!(descriptor.matrices(), [matrix] if matrix.rows == kernels::VOCABULARY)
        || !descriptor.parameters().is_empty()
        || node.state.is_some()
        || node.outputs.as_slice() != std::slice::from_ref(&id)
        || node.inputs.as_slice() != origin.inputs()
        || node.inputs.len() != 1
        || record.selection().normalized() != node.inputs.first().copied()
    {
        return Err(Error::State(
            "BF16 greedy Head descriptor/value/producer changed".into(),
        ));
    }
    Ok(())
}

struct HostOwner {
    values: Arc<ModelValues>,
    source: VerifiedTensorSource,
    // Actual vectors/descriptors in PreparedBf16Greedy drop before this charge.
    _metadata: WindowWorkspace,
}

/// Preallocated under the caller's genuine original initialization. The SAME
/// enclosing flight must retain this Arc before its first producer enqueue.
pub(crate) struct PreparedBf16Greedy {
    service: Arc<ResidentService>,
    logits: ResidentLease,
    partials: ResidentLease,
    packet: ResidentLease,
    status: ResidentLease,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
    call: SourceCall,
    logits_value: ValueId,
    owner: Arc<HostOwner>,
}
impl PreparedBf16Greedy {
    pub(crate) const fn managed_bytes() -> usize {
        PARTIAL_BYTES + PACKET_BYTES
    }

    pub(crate) fn prepare(
        source: VerifiedTensorSource,
        values: Arc<ModelValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let metadata = source.reserve_workspace(METADATA_BYTES)?;
        let record = values.record().as_ref();
        require_head(record)?;
        let service = values.service().clone();
        init.require_service(&service)?;
        let logits_value = record.selection().output();
        let call = record.call().clone();
        // ModelValues seals physical BF16 layout and original lease ownership
        // once at the real workspace allocator. No separate operand override.
        let logits = values.lease(logits_value)?.clone();
        let status = values.status().clone();
        if source.source_identity() != record.call().source()
            || source.source_identity() != service.source_identity()
            || !Arc::ptr_eq(logits.service(), &service)
            || logits.bytes() != LOGITS_BYTES
            || !matches!(logits.role(), BufferRole::Output | BufferRole::Activation)
            || !Arc::ptr_eq(status.service(), &service)
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "BF16 greedy original logits/status source/service/role".into(),
            ));
        }
        require_ranges(
            service.client().service_id(),
            &[logits.handle(), status.handle()],
            &[LOGITS_BYTES, 4],
        )?;
        if service.remaining_capacity()? < Self::managed_bytes() {
            return Err(Error::Limit(
                "BF16 greedy complete private buffer admission".into(),
            ));
        }
        let owner = Arc::new(HostOwner {
            values,
            source,
            _metadata: metadata,
        });
        init.retain_completion_owner(owner.clone())?;
        let partials = init.allocate_empty(PARTIAL_BYTES, BufferRole::Scratch)?;
        let packet = init.allocate_empty(PACKET_BYTES, BufferRole::Output)?;
        let mut result = Self {
            service,
            logits,
            partials,
            packet,
            status,
            accesses: Vec::with_capacity(4),
            outputs: Vec::with_capacity(2),
            call,
            logits_value,
            owner,
        };
        // The original initialization already owns the host admission and both
        // new leases. No producer command has been enqueued by this helper.
        result.validate_bindings()?;
        result.accesses.extend([
            (result.logits.clone(), DeviceAccess::Read),
            (result.partials.clone(), DeviceAccess::Write),
            (result.packet.clone(), DeviceAccess::Write),
            (result.status.clone(), DeviceAccess::Write),
        ]);
        result
            .outputs
            .extend([result.partials.clone(), result.packet.clone()]);
        let result = Arc::new(result);
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }

    fn validate_bindings(&self) -> Result<()> {
        self.service.ensure_usable()?;
        if self.owner.source.source_identity() != self.call.source()
            || self.service.source_identity() != self.call.source()
            || !Arc::ptr_eq(self.owner.values.service(), &self.service)
            || self.owner.values.record().call() != &self.call
            || self.owner.values.record().selection().output() != self.logits_value
        {
            return Err(Error::State(
                "BF16 greedy retained source/request changed".into(),
            ));
        }
        let original = self.owner.values.lease(self.logits_value)?;
        let original_status = self.owner.values.status();
        if original.id() != self.logits.id()
            || original.allocation_generation() != self.logits.allocation_generation()
            || original.service_generation() != self.logits.service_generation()
            || original_status.id() != self.status.id()
            || original_status.allocation_generation() != self.status.allocation_generation()
            || original_status.service_generation() != self.status.service_generation()
        {
            return Err(Error::State(
                "BF16 greedy original sealed Head/status lease changed".into(),
            ));
        }
        let leases = [&self.logits, &self.partials, &self.packet, &self.status];
        let sizes = [LOGITS_BYTES, PARTIAL_BYTES, PACKET_BYTES, 4];
        let mut handles = Vec::with_capacity(4);
        for (lease, bytes) in leases.into_iter().zip(sizes) {
            if !Arc::ptr_eq(lease.service(), &self.service) || lease.bytes() != bytes {
                return Err(Error::State(
                    "BF16 greedy original exact lease service/extent".into(),
                ));
            }
            handles.push(lease.handle());
        }
        require_ranges(self.service.client().service_id(), &handles, &sizes)
    }

    /// SAME body output postlude, after the actual Head node, before original
    /// completion. Shared status has already been cleared once by that body.
    pub(crate) fn enqueue(&self, cancelled: Option<&AtomicBool>) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.validate_bindings()?;
        // SAFETY: all exact original ranges/allocations checked above and
        // retained by caller's original armed flight. Packed loads are U32
        // transport only; logical/physical input remains dense BF16.
        unsafe {
            kernels::block_winners::launch_unchecked(
                self.service.client(),
                CubeCount::Static((kernels::BLOCKS as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(self.logits.handle(), kernels::VOCABULARY / 2),
                BufferArg::from_raw_parts(self.partials.handle(), kernels::PARTIAL_WORDS),
                BufferArg::from_raw_parts(self.status.handle(), 1),
            );
        }
        not_cancelled(cancelled)?;
        // SAFETY: same checked retained private ranges; preceding block stage
        // is ordered on this original actor/stream. No separate completion.
        unsafe {
            kernels::finish_winners::launch_unchecked(
                self.service.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(self.partials.handle(), kernels::PARTIAL_WORDS),
                BufferArg::from_raw_parts(self.packet.handle(), 3),
                BufferArg::from_raw_parts(self.status.handle(), 1),
            );
        }
        Ok(2)
    }

    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn readbacks(&self) -> Result<[ResidentReadback; 2]> {
        Ok([
            self.packet.readback(0, PACKET_BYTES)?,
            self.status.readback(0, 4)?,
        ])
    }
    /// Run only inside the original trusted physical completion callback.
    /// Packet validity is not a model root, complete-prompt emission or token
    /// feedback permit. Those remain with the original engine coordinator.
    pub(crate) fn validate_readback(&self, raw: &[Bytes]) -> Result<(u32, u32)> {
        if raw.len() != 2 {
            return Err(Error::State("BF16 greedy packet/status cohort".into()));
        }
        parse_packet(&raw[0], &raw[1])
    }
}

fn require_ranges(
    service: ServiceId,
    handles: &[cubecl::server::Handle],
    sizes: &[usize],
) -> Result<()> {
    if handles.len() != sizes.len() || handles.is_empty() || handles.len() > 4 {
        return Err(Error::State(
            "BF16 greedy complete fixed binding set".into(),
        ));
    }
    let mut ranges = Vec::with_capacity(handles.len());
    for (handle, &bytes) in handles.iter().zip(sizes) {
        let range = BufferRange::new(handle)?;
        if range.service != service
            || range.bytes.start != 0
            || range.bytes.end - range.bytes.start != bytes as u64
            || ranges
                .iter()
                .any(|old: &BufferRange| old.memory == range.memory)
        {
            return Err(Error::State(
                "BF16 greedy original exact distinct whole allocation ranges".into(),
            ));
        }
        ranges.push(range);
    }
    Ok(())
}

fn parse_packet(packet: &[u8], status: &[u8]) -> Result<(u32, u32)> {
    if packet.len() != 12 || status.len() != 4 {
        return Err(Error::State("BF16 greedy packet/status byte extent".into()));
    }
    let id = u32::from_le_bytes(packet[0..4].try_into().unwrap());
    let bits = u32::from_le_bytes(packet[4..8].try_into().unwrap());
    if packet[8..12] != [0; 4]
        || status != [0; 4]
        || id as usize >= kernels::VOCABULARY
        || bits & 0xffff != 0
        || bits & 0x7f800000 == 0x7f800000
    {
        return Err(Error::State(
            "BF16 greedy nonfinite/failed/foreign source-bit packet".into(),
        ));
    }
    Ok((id, bits))
}

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod natural;

mod reusable;

mod publication;

mod fixed_prompt_natural;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod text;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub(crate) mod cohort;

/// Same generated keys as actual independent N1/B4 greedy packet producers.
pub(crate) fn prepare_text_code(
    client: &Client,
    queue: &cubecl::cuda::OwnedCudaQueue,
    cohort: bool,
) -> Result<usize> {
    let settings = || {
        KernelSettings::new(
            *CubeDim::new_1d(32),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
    };
    let a = || BufferCompilationArg { inplace: None };
    let mut kernels: Vec<Box<dyn CubeKernel>> = vec![
        Box::new(kernels::block_winners::BlockWinners::new(
            settings(),
            client.properties_shared(),
            client.target_properties_shared(),
            a(),
            a(),
            a(),
        )),
        Box::new(kernels::finish_winners::FinishWinners::new(
            settings(),
            client.properties_shared(),
            client.target_properties_shared(),
            a(),
            a(),
            a(),
        )),
    ];
    if cohort {
        kernels.push(Box::new(
            kernels::cohort_block_winners::CohortBlockWinners::new(
                settings(),
                client.properties_shared(),
                client.target_properties_shared(),
                a(),
                a(),
                a(),
                4,
            ),
        ));
        kernels.push(Box::new(
            kernels::cohort_finish_winners::CohortFinishWinners::new(
                settings(),
                client.properties_shared(),
                client.target_properties_shared(),
                a(),
                a(),
                a(),
                4,
            ),
        ));
    }
    let count = kernels.len();
    for kernel in kernels {
        // SAFETY: same original admitted actor and exact launch factory/settings.
        unsafe { queue.prepare_kernel(kernel) }
            .map_err(|e| Error::State(format!("original greedy code readiness: {e}")))?;
    }
    Ok(count)
}
