//! Two fresh request pairs, one loaded model and the SAME original scratch.
use super::*;
use crate::Device;
use crate::cuda::native::source_rotary::RotaryOutcome;
use crate::resident_inference::native_fragment::natural_reuse::{
    PreparedReusableNatural, ReusableNaturalStorage, prepare_reusable_natural_cold,
};
use crate::resident_inference::native_fragment::{
    SourceModelInputs, SourceModelReadback, SourceModelScopes,
};
use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::{BufferLeaseId, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

mod faults;
pub use faults::{
    NaturalReuseFault, NaturalReuseFaultExecution, NaturalReuseFaultOutcome,
    qualify_natural_reuse_fault,
};

#[derive(Debug)]
pub struct NaturalReuseRound {
    pub roots: [RootBinding; 2],
    pub phase_roots: Vec<RootBinding>,
    pub short_tokens: [u32; 2],
    pub long_tokens: [u32; 2],
}
#[derive(Debug)]
pub struct NaturalReuseReceipt {
    pub rounds: Vec<NaturalReuseRound>,
    pub calls: usize,
    pub reused_geometries: usize,
    pub distinct_packets: usize,
    pub preserved_committed_packets: usize,
    pub prefix64_emissions: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
    pub device_copies: usize,
    pub dispatches: usize,
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}

pub(super) fn value_ids(
    body: &crate::resident_program::native_full::FullModelProgram,
) -> Result<Vec<BufferLeaseId>> {
    (0..327)
        .map(|i| body.values().lease(ValueId(i)).map(ResidentLease::id))
        .chain(std::iter::once(Ok(body.values().status().id())))
        .collect()
}

pub(super) async fn verify_packets(
    model: &Arc<ResidentModel>,
    storage: &Arc<ReusableNaturalStorage>,
    packets: &[Arc<CommittedBf16Feedback>],
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    let service = &model.service;
    for packet in packets {
        let packet_model = packet.model.upgrade().ok_or_else(|| {
            Error::State("reusable preserved packet original model retired".into())
        })?;
        if !Arc::ptr_eq(&packet_model, model)
            || !packet.source.same_issuer(storage.source())
            || packet.root.source() != service.source_identity()
            || packet.token.lease.source_identity() != service.source_identity()
            || packet.token.producer != Some(packet.root)
            || !Arc::ptr_eq(packet.token.lease.service(), service)
            || !packet.token.lease.is_initialized()
            || packet.token.lease.bytes() != 12
            || packet.token.lease.role() != BufferRole::Output
            || packet.token.lease.allocation_generation() != packet.allocation_generation
            || packet.token.lease.service_generation() != packet.service_generation
        {
            return Err(Error::State(
                "reusable preserved packet original source/root/generation differs".into(),
            ));
        }
    }
    let accesses = packets
        .iter()
        .map(|p| (p.token.lease.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = packets
        .iter()
        .map(|p| p.token.lease.readback(0, 12))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 96,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    let _workspace = model.workspace.lock().await;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(storage.completion_owner())?;
    for packet in packets {
        flight.retain_completion_owner(packet.clone())?;
    }
    drop(
        flight
            .complete_read_only_readback(&reads, |raw| {
                if raw.len() != packets.len() {
                    return Err(Error::State("reusable preserved packet cohort".into()));
                }
                for (raw, packet) in raw.iter().zip(packets) {
                    if raw[..] != packet.packet_bytes {
                        return Err(Error::State(
                            "reusable committed packet changed during scratch reuse".into(),
                        ));
                    }
                }
                Ok(())
            })
            .await?,
    );
    slot.finish()?;
    Ok(())
}
