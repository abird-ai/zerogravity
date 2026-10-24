//! One model-free original initialization/flight; no synthetic model Ready.
use super::*;
use cubecl::Device;
use zerogravity_core::{
    accelerator::SourceIdentity,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};

const CAP: usize = 16 * 1024 * 1024;
const LENGTHS: [usize; 8] = [1, 127, 129, 257, 2, 255, 1025, 130];
const FINITE: [u16; 6] = [0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f];
struct Plan {
    inputs: Vec<Vec<u8>>,
    lengths: Vec<u32>,
    initial: u32,
    expected: u32,
}
struct Case {
    // Dispatch bindings retire before their original physical owners.
    launch: crate::vllm_ptx_inline::PreparedPinnedLaunch,
    inputs: Vec<ResidentLease>,
    status: ResidentLease,
    expected_inputs: Vec<Vec<u8>>,
    expected_status: u32,
}
fn plan(width: usize, invalid: Option<(usize, usize, u16)>, initial: u32) -> Plan {
    let inputs = LENGTHS[..width]
        .iter()
        .enumerate()
        .map(|(slot, &n)| {
            let mut bits = vec![0x7fc1u16; n + 4]; // Poison outside exact logical view.
            for i in 0..n {
                bits[i + 2] = FINITE[(i + slot) % FINITE.len()];
            }
            if let Some((target, at, value)) = invalid
                && target == slot
            {
                bits[at + 2] = value;
            }
            bits.into_iter().flat_map(u16::to_ne_bytes).collect()
        })
        .collect();
    Plan {
        inputs,
        lengths: LENGTHS[..width].iter().map(|&n| n as u32).collect(),
        initial,
        expected: initial | u32::from(invalid.is_some()),
    }
}

pub(crate) async fn qualify(
    device: Device,
    host: &PhysicalDeviceKey,
    owner: ResourceOwner,
) -> Result<usize> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "finite cohort release CUDA qualification only".into(),
        ));
    }
    let arbiter = ResourceArbiter::process_shared();
    let host_ticket = arbiter.admit(
        host,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: CAP as u64,
        }],
    )?;
    let mut plans = Vec::new();
    for width in 2..=WIDTH {
        for initial in [0, 4, 5] {
            plans.push(plan(width, None, initial));
        }
        for (slot, &n) in LENGTHS.iter().enumerate().take(width) {
            for bits in [0x7f80, 0xff80, 0x7fc1, 0xff81] {
                plans.push(plan(width, Some((slot, 0, bits)), 0));
                plans.push(plan(width, Some((slot, n - 1, bits)), 4));
            }
        }
    }
    let payload = plans.iter().try_fold(0usize, |a, p| {
        p.inputs.iter().try_fold(a + 4, |n, b| {
            n.checked_add(b.len())
                .ok_or_else(|| Error::Limit("finite fixture extent overflow".into()))
        })
    })?;
    let count = plans.iter().map(|p| p.inputs.len() + 1).sum::<usize>();
    if plans.len() > 512
        || count > 4096
        || payload
            .checked_mul(8)
            .and_then(|b| b.checked_add(count * 1024))
            .is_none_or(|b| b > CAP)
    {
        return Err(Error::Limit(
            "finite fixture complete host/device bound".into(),
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        SourceIdentity::for_new_allocation()?,
        owner,
        CAP,
        5 * 1024 * 1024 * 1024,
    )?;
    let key = service.physical_key();
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: payload as u64,
        }],
        None,
    )?;
    let mut init_slot = ticket.wait().await?;
    init_slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let mut cases = Vec::with_capacity(plans.len());
    for p in plans {
        let mut inputs = Vec::with_capacity(p.inputs.len());
        let mut handles = Vec::with_capacity(p.inputs.len() + 1);
        for bytes in &p.inputs {
            let lease = init.upload_bytes(bytes, BufferRole::Activation)?;
            handles.push(lease.handle().offset_start(4).offset_end(4));
            inputs.push(lease);
        }
        let status = init.upload_bytes(&p.initial.to_ne_bytes(), BufferRole::Scratch)?;
        handles.push(status.handle());
        let (text, grid) = source(&p.lengths)?;
        let mut bindings = p
            .lengths
            .iter()
            .map(|&n| (n as usize * 2, BufferIOAttr::ReadOnly))
            .collect::<Vec<_>>();
        bindings.push((4, BufferIOAttr::ReadWrite));
        let kernel = PinnedKernel::new("zg_native_bf16_finite_cohort", text, 128, bindings.into())?;
        cases.push(Case {
            launch: kernel.prepare(service.client(), [grid, 1, 1], handles)?,
            inputs,
            status,
            expected_inputs: p.inputs,
            expected_status: p.expected,
        });
    }
    drop(init.complete_sync().await?);
    init_slot.finish()?;
    let cases = Arc::new(cases);
    let mut accesses = Vec::with_capacity(count);
    let mut reads = Vec::with_capacity(count);
    let mut outputs = Vec::with_capacity(cases.len());
    for c in cases.iter() {
        for input in &c.inputs {
            accesses.push((input.clone(), DeviceAccess::Read));
            reads.push(input.readback(0, input.bytes())?);
        }
        accesses.push((c.status.clone(), DeviceAccess::Write));
        reads.push(c.status.readback(0, 4)?);
        outputs.push(c.status.clone());
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: payload as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(cases.clone())?;
    for c in cases.iter() {
        c.launch.launch()?;
    }
    let (ready, raw) = flight
        .complete_with_readback(&outputs, &reads, |data| {
            if data.len() != count {
                return Err(Error::State("finite fixture whole readback".into()));
            }
            let mut at = 0;
            for c in cases.iter() {
                for bytes in &c.expected_inputs {
                    if data[at][..] != bytes[..] {
                        return Err(Error::State(
                            "finite fixture readonly range/canary changed".into(),
                        ));
                    }
                    at += 1;
                }
                if data[at].as_ref() != c.expected_status.to_ne_bytes() {
                    return Err(Error::State(
                        "finite fixture actual BF16/sticky predicate differs".into(),
                    ));
                }
                at += 1;
            }
            // This completion receipt covers synthetic status writes only; there is
            // no model, token, finite-model claim, root or publisher in this fixture.
            Ok(())
        })
        .await?;
    let cases_checked = cases.len();
    drop(raw);
    drop(ready);
    drop(reads);
    drop(outputs);
    drop(accesses);
    drop(cases);
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State("finite fixture retained lease/event".into()));
    }
    service
        .client()
        .memory_cleanup()
        .map_err(|e| Error::State(format!("finite fixture completed pool retirement: {e}")))?;
    slot.finish()?;
    drop(service);
    drop(host_ticket);
    let final_state = arbiter.snapshot(&key)?;
    if final_state.fenced || final_state.reserved != 0 {
        return Err(Error::State(
            "finite fixture final original physical reservation".into(),
        ));
    }
    Ok(cases_checked)
}
