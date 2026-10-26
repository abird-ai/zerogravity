//! Synthetic operator qualification only; no model/token/session/publisher.
//! Original physical12-step oracle stays unchanged; snapshots only copy U32.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{CubeElement, Device, bytes::Bytes};
use std::sync::Arc;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

const PAD: usize = 4;
const CANARY: u32 = 0xa5a55a5a;

fn view(lease: &ResidentLease, words: usize) -> Handle {
    lease
        .handle()
        .offset_start((PAD * 4) as u64)
        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
}

fn inputs(width: usize, rows: usize, recipe: usize) -> Vec<u32> {
    const SPECIAL: [u32; 26] = [
        0, 0x80000000, 1, 0x80000001, 2, 0x80000002, 3, 0x80000003, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x00800001, 0x80800001, 0x3f800000, 0xbf800000, 0x43800000,
        0xc3800000, 0x42af0000, 0xc2af0000, 0x42ce0000, 0xc2ce0000, 0x43050000, 0xc3050000,
        0x3f7fffff, 0xbf7fffff,
    ];
    (0..width * rows)
        .map(|at| {
            let random = (at as u32)
                .wrapping_mul(0x9e3779b9)
                .wrapping_add(0x5135791d);
            match recipe {
                0 => SPECIAL[at % SPECIAL.len()],
                1 => (random & 0x80000000) | ((random & 0x7fffffff) % 0x43800001),
                4 => {
                    // Analytic neighborhoods of the SOURCE range-reduction ties;
                    // these select inputs, never reconstruct expected outputs.
                    let n = [
                        0.0f32, 1.0, 125.0, 126.0, 127.0, 191.0, 192.0, 193.0, 367.0, 368.0,
                    ][(at / 257) % 10];
                    let center = ((n + 0.5) / f32::from_bits(0x3fb8aa3b)).to_bits();
                    let neighbor = center.wrapping_add((at % 257) as u32).wrapping_sub(128);
                    neighbor
                        | if (at / 2570).is_multiple_of(2) {
                            0
                        } else {
                            0x80000000
                        }
                }
                _ => (random & 0x80000000) | 0x3f000000 | (random & 0x007fffff),
            }
        })
        .collect()
}

/// Post-stage U32 snapshots add no arithmetic to original SiLU operators.
#[cube(launch_unchecked)]
fn snapshot(
    scratch: &[u32],
    output: &[u32],
    trace: &mut [u32],
    #[comptime] count: usize,
    #[comptime] stage: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let s = i * 10;
        let t = i * 17;
        if stage < 4usize {
            trace[t + stage] = scratch[s + stage];
        } else if stage == 4usize {
            trace[t + 4] = scratch[s + 4];
            trace[t + 5] = scratch[s + 5];
            trace[t + 6] = scratch[s + 6];
            trace[t + 7] = scratch[s + 7];
        } else if stage == 5usize {
            trace[t + 8] = scratch[s + 5];
        } else if stage == 6usize {
            trace[t + 9] = scratch[s + 5];
        } else if stage == 7usize {
            trace[t + 10] = scratch[s + 2];
            trace[t + 11] = scratch[s + 8];
        } else if stage == 8usize {
            trace[t + 12] = scratch[s + 2];
        } else if stage == 9usize {
            trace[t + 13] = scratch[s + 9];
        } else if stage == 10usize {
            trace[t + 14] = output[i];
        } else if stage == 11usize {
            trace[t + 15] = output[i];
        } else {
            trace[t + 16] = output[i];
        }
    }
}

fn arg(handle: &Handle, n: usize) -> BufferArg {
    // SAFETY: fixture creates full checked original allocations/views before any
    // launch and pins their leases in the single original physical flight.
    unsafe { BufferArg::from_raw_parts(handle.clone(), n) }
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}

fn certify(client: &Client, input: &Handle, output: &Handle, status: &Handle, n: usize) {
    // SAFETY: original buffers/count/status are preflighted; n<=524288<MAX_WORD_OPS.
    unsafe {
        crate::resident_bindings::certify_interval::launch_unchecked(
            client,
            grid(n),
            CubeDim::new_1d(32),
            arg(input, n),
            arg(status, 1),
            -256.0,
            256.0,
            n,
        );
        crate::program_kernels::certify_finite_strided::launch_unchecked(
            client,
            grid(n),
            CubeDim::new_1d(32),
            arg(output, n),
            arg(status, 1),
            0,
            n as u32,
            1,
            n,
        );
    }
}

fn original(
    client: &Client,
    input: &Handle,
    multiplier: Option<&Handle>,
    scratch: &Handle,
    output: &Handle,
    trace: &Handle,
    n: usize,
) -> usize {
    // SAFETY: all complete original leases are already in original submission.
    unsafe {
        for stage in 0..12 {
            crate::silu::silu_step::launch_unchecked(
                client,
                grid(n),
                CubeDim::new_1d(32),
                arg(input, n),
                arg(scratch, n * crate::silu::SCRATCH),
                arg(output, n),
                n,
                stage,
            );
            snapshot::launch_unchecked(
                client,
                grid(n),
                CubeDim::new_1d(32),
                arg(scratch, n * 10),
                arg(output, n),
                arg(trace, n * 17),
                n,
                stage,
            );
        }
        if let Some(m) = multiplier {
            crate::silu::silu_gate::launch_unchecked(
                client,
                grid(n),
                CubeDim::new_1d(32),
                arg(m, n),
                arg(output, n),
                n,
            );
        }
        snapshot::launch_unchecked(
            client,
            grid(n),
            CubeDim::new_1d(32),
            arg(scratch, n * 10),
            arg(output, n),
            arg(trace, n * 17),
            n,
            12,
        );
    }
    12 + usize::from(multiplier.is_some())
}

fn words(bytes: &Bytes, n: usize) -> Result<Vec<u32>> {
    if bytes.len() != (n + 2 * PAD) * 4 {
        return Err(Error::State("ordered SiLU complete canary readback".into()));
    }
    let data = bytes
        .chunks_exact(4)
        .map(|x| u32::from_ne_bytes(x.try_into().expect("word")))
        .collect::<Vec<_>>();
    if data[..PAD]
        .iter()
        .chain(&data[PAD + n..])
        .any(|&x| x != CANARY)
    {
        return Err(Error::State(
            "ordered SiLU prefix/suffix canary overwritten".into(),
        ));
    }
    Ok(data[PAD..PAD + n].to_vec())
}

#[cfg(target_arch = "x86_64")]
fn cpu_math_environment() -> Result<u32> {
    let mut value = 0u32;
    // SAFETY: writes one valid local MXCSR word; does not alter FP state.
    unsafe {
        std::arch::asm!("stmxcsr [{}]",in(reg) &mut value,options(nostack,preserves_flags));
    }
    if value & 0xe040 != 0 {
        return Err(Error::Unsupported(
            "original CPU SiLU oracle requires RN, FTZ/DAZ off".into(),
        ));
    }
    // Exception sticky flags may change during legitimate source arithmetic.
    Ok(value & 0xffc0)
}
#[cfg(not(target_arch = "x86_64"))]
fn cpu_math_environment() -> Result<u32> {
    Err(Error::Unsupported(
        "closed original Haswell CPU oracle".into(),
    ))
}

async fn case(
    service: &Arc<ResidentService>,
    rows: usize,
    width: usize,
    recipe: usize,
    multiplied: bool,
    fault: &str,
) -> Result<usize> {
    let n = geometry(rows, width)?;
    let mut source = inputs(width, rows, recipe);
    if fault == "nonfinite" {
        source[0] = 0x7fc12345;
    }
    if fault == "range" {
        source[0] = 0x43800001;
    }
    let multiplier = (0..n)
        .map(|i| {
            [
                0x3f800000, 0xbf800000, 0, 0x80000000, 1, 0x80000001, 0x40000000, 0xc0000000,
            ][i % 8]
        })
        .collect::<Vec<_>>();
    let multiplier = if recipe == 3 {
        source.clone()
    } else {
        multiplier
    };
    let sizes = [n, n, n * 10, n, n, n, n * 17, n * 17, 1, 1, 1];
    let bytes = sizes.iter().map(|n| (n + 2 * PAD) * 4).sum::<usize>();
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: bytes as u64,
        }],
        None,
    )?;
    // Initialization owns its original begun slot until physical Ready.
    let mut init_slot = ticket.wait().await?;
    init_slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    service
        .client()
        .memory_cleanup()
        .map_err(|e| Error::State(format!("ordered SiLU cold pool cleanup: {e}")))?;
    let mut leases = Vec::new();
    for (index, &size) in sizes.iter().enumerate() {
        let mut data = vec![CANARY; size + 2 * PAD];
        if index < 2 {
            data[PAD..PAD + size].copy_from_slice(if index == 0 { &source } else { &multiplier });
        }
        if index >= 8 {
            data[PAD] = if fault == "preexisting" { 5 } else { 0 };
        }
        leases.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
    }
    let initialized = init.complete_sync().await?;
    if initialized.outputs().len() != leases.len() {
        return Err(Error::State("ordered SiLU initialization coverage".into()));
    }
    drop(initialized);
    init_slot.finish()?;
    let handles = leases
        .iter()
        .zip(sizes)
        .map(|(l, n)| view(l, n))
        .collect::<Vec<_>>();
    let client = service.client();
    let multiplier_handle = multiplied.then_some(&handles[if recipe == 3 { 0 } else { 1 }]);
    let binding = |output: usize, trace: Option<usize>| Bindings {
        input: &handles[0],
        multiplier: multiplier_handle,
        output: &handles[output],
        trace: trace.map(|i| &handles[i]),
    };
    let traced = PreparedOrderedSilu::prepare(client, rows, width, binding(4, Some(7)))?;
    let compact = PreparedOrderedSilu::prepare(client, rows, width, binding(5, None))?;
    let stable = (
        service.resident_bytes()?,
        service.uploaded_value_bytes(),
        service.pending_events()?,
    );
    let invalid = PreparedOrderedSilu::prepare(
        client,
        rows,
        width,
        Bindings {
            input: &handles[0],
            multiplier: None,
            output: &handles[0],
            trace: None,
        },
    );
    if !matches!(invalid,Err(ref e) if e.to_string().contains("alias")) {
        return Err(Error::State("ordered SiLU prelaunch alias accepted".into()));
    }
    let mut foreign = handles[0].clone();
    foreign.service.service = std::any::TypeId::of::<()>();
    if PreparedOrderedSilu::prepare(
        client,
        rows,
        width,
        Bindings {
            input: &foreign,
            multiplier: None,
            output: &handles[4],
            trace: None,
        },
    )
    .is_ok()
        || stable
            != (
                service.resident_bytes()?,
                service.uploaded_value_bytes(),
                service.pending_events()?,
            )
    {
        return Err(Error::State(
            "ordered SiLU cold refusal changed original ledger/uploads".into(),
        ));
    }
    for issue in 0..3 {
        let mut bad = handles[0].clone();
        if issue == 0 {
            bad.offset_end = Some(bad.size() + 1);
        } else if issue == 1 {
            bad.offset_start = Some(bad.offset_start.unwrap_or(0) + 1);
        } else {
            bad.offset_end = Some(bad.offset_end.unwrap_or(0) + 4);
        }
        if PreparedOrderedSilu::prepare(
            client,
            rows,
            width,
            Bindings {
                input: &bad,
                multiplier: None,
                output: &handles[4],
                trace: None,
            },
        )
        .is_ok()
        {
            return Err(Error::State(
                "ordered SiLU malformed/short view accepted".into(),
            ));
        }
    }
    if PreparedOrderedSilu::prepare(
        client,
        rows,
        width,
        Bindings {
            input: &handles[0],
            multiplier: None,
            output: &handles[7],
            trace: Some(&handles[7]),
        },
    )
    .is_ok()
    {
        return Err(Error::State(
            "ordered SiLU trace/output alias accepted".into(),
        ));
    }
    // Cancelling an unsubmitted original queue ticket cannot launch this kernel.
    let queue_before = service.queue().stats()?;
    let cancelled = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 4,
        }],
        None,
    )?;
    drop(cancelled);
    let queue_after = service.queue().stats()?;
    if queue_after.cancelled_queued != queue_before.cancelled_queued + 1
        || stable
            != (
                service.resident_bytes()?,
                service.uploaded_value_bytes(),
                service.pending_events()?,
            )
    {
        return Err(Error::State(
            "ordered SiLU prelaunch cancellation touched physical data".into(),
        ));
    }
    let accesses = leases
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (
                l.clone(),
                if i < 2 {
                    DeviceAccess::Read
                } else {
                    DeviceAccess::Write
                },
            )
        })
        .collect::<Vec<_>>();
    let reads = leases
        .iter()
        .map(|l| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
    let uploads = service.uploaded_value_bytes();
    let mxcsr = cpu_math_environment()?;
    // All bindings/readbacks/refusals are checked before this separate math
    // flight. Same prepaid envelope, original fair queue and retained leases.
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let flight = service.submit(0, &accesses)?;
    let old = original(
        client,
        &handles[0],
        multiplier_handle,
        &handles[2],
        &handles[3],
        &handles[6],
        n,
    );
    let native = traced.launch(client)? + compact.launch(client)?;
    for (out, status) in [(3, 8), (4, 9), (5, 10)] {
        certify(client, &handles[0], &handles[out], &handles[status], n);
    }
    if fault == "unknown" {
        drop(flight);
        drop(slot);
        let budget = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
        if !budget.fenced
            || budget.reserved == 0
            || service.pending_events()? == 0
            || service.ensure_usable().is_ok()
            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
        {
            return Err(Error::State(
                "ordered SiLU unknown completion escaped original fence/quarantine".into(),
            ));
        }
        return Ok(0);
    }
    let mut checked = 0;
    let mut rejected_status = false;
    let result=flight.complete_with_readback(&leases[2..],&reads,|read| {
        if read.len()!=sizes.len() {return Err(Error::State("ordered SiLU full readback coverage".into()));}
        let data=read.iter().zip(sizes).map(|(b,n)|words(b,n)).collect::<Result<Vec<_>>>()?;
        if data[0]!=source || data[1]!=multiplier {return Err(Error::State("ordered SiLU read-only input changed".into()));}
        if data[8]!=data[9] || data[8]!=data[10] {return Err(Error::State("ordered SiLU original finite/status bits changed".into()));}
        if !fault.is_empty() && fault!="known-ready-abandon" {
            let expected=match fault {"preexisting"=>5,"nonfinite"=>18,"range"=>16,_=>0};
            if data[8][0]!=expected {return Err(Error::State("ordered SiLU invalid/preexisting status bits changed".into()));}
            rejected_status=true;
            return Err(Error::State("ordered SiLU original status rejected before Ready".into()));
        }
        if data[8][0]!=0 {return Err(Error::State("ordered SiLU unexpected finite/status refusal".into()));}
        for (at,&source_word) in source.iter().enumerate() {
            if data[3][at]!=data[4][at] || data[3][at]!=data[5][at] {
                return Err(Error::State(format!("ordered SiLU physical final bits at{at}: old={:08x} trace={:08x} compact={:08x}",data[3][at],data[4][at],data[5][at])));
            }
            for checkpoint in 0..17 {
                if data[6][at*17+checkpoint]!=data[7][at*17+checkpoint] {
                    return Err(Error::State(format!("ordered SiLU physical checkpoint{checkpoint} at{at}: old={:08x} native={:08x}",data[6][at*17+checkpoint],data[7][at*17+checkpoint])));
                }
            }
            let x=f32::from_bits(source_word);
            let cpu_exp=zerogravity_cpu::ops::ggml_source_exp(0.0-x);
            if data[6][at*17+12]!=cpu_exp.to_bits() || data[6][at*17+13]!=(1.0+cpu_exp).to_bits() {
                return Err(Error::State(format!("ordered SiLU original CPU exp/denominator bits at{at}")));
            }
            let cpu=zerogravity_cpu::ops::ggml_source_silu(x);
            let cpu=if multiplied {f32::from_bits(multiplier[at])*cpu} else {cpu};
            if data[3][at]!=cpu.to_bits() {
                return Err(Error::State(format!("ordered SiLU original CPU source bits at{at}: GPU={:08x} CPU={:08x}",data[3][at],cpu.to_bits())));
            }
            checked+=22;
        }
        if cpu_math_environment()?!=mxcsr {return Err(Error::State("ordered SiLU CPU oracle math state changed".into()));}
        Ok(())
    }).await;
    if rejected_status {
        drop(slot);
        if !matches!(result,Err(ref e) if e.to_string().contains("original status rejected"))
            || service.ensure_usable().is_ok()
            || service.pending_events()? == 0
            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
        {
            return Err(Error::State(
                "ordered SiLU invalid certificate escaped original quarantine".into(),
            ));
        }
        return Ok(0);
    }
    let (ready, data) = result?;
    if old != 12 + usize::from(multiplied)
        || native != 2
        || ready.outputs().len() != leases.len() - 2
        || data.len() != leases.len()
        || service.uploaded_value_bytes() != uploads
    {
        return Err(Error::State(
            "ordered SiLU actual source launch/Ready/read/upload coverage".into(),
        ));
    }
    // Known Ready private-candidate abandonment publishes no SessionAuthority/
    // StateRoot. Fullmodel per-request cancellation remains a separate gate.
    drop(ready);
    drop(data);
    slot.finish()?;
    Ok(checked)
}

#[test]
#[ignore = "ROOT release CUDA original queue/leases physical differential only"]
fn physical_ordered_silu() -> Result<()> {
    futures_lite::future::block_on(async {
        if cfg!(debug_assertions) {
            return Err(Error::Unsupported(
                "ordered SiLU qualification requires release".into(),
            ));
        }
        let fault = std::env::var("ZG_ORDERED_SILU_FAULT").unwrap_or_default();
        if !matches!(
            fault.as_str(),
            "" | "unknown" | "preexisting" | "nonfinite" | "range" | "known-ready-abandon"
        ) {
            return Err(Error::Malformed("ordered SiLU exact fault selector".into()));
        }
        let source = SourceIdentity::for_new_allocation()?;
        let service = ResidentService::new(
            Device::Cuda(cubecl::device::CudaDevice::new(0)),
            source,
            ResourceOwner::new(
                "ordered-silu-synthetic-operator",
                "original-physical-oracle",
            )?,
            256 * 1024 * 1024,
            640 * 1024 * 1024,
        )?;
        let key = service.physical_key();
        let mut compared = 0;
        if fault.is_empty() {
            for rows in [1, 2, 4, 8] {
                for width in [1, 31, 33, 128, 2048, 6144, 11008, 65536] {
                    for recipe in 0..3 {
                        for multiplied in [false, true] {
                            compared += case(&service, rows, width, recipe, multiplied, "").await?;
                            if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
                                return Err(Error::State(
                                    "ordered SiLU live private lease/event after case".into(),
                                ));
                            }
                        }
                    }
                }
            }
            for rows in [1, 2, 4, 8] {
                compared += case(&service, rows, 33, 3, true, "").await?;
                for multiplied in [false, true] {
                    compared += case(&service, rows, 5140, 4, multiplied, "").await?;
                }
            }
        } else {
            compared += case(&service, 8, 33, 0, true, &fault).await?;
        }
        let unsafe_terminal = matches!(
            fault.as_str(),
            "unknown" | "preexisting" | "nonfinite" | "range"
        );
        drop(service);
        let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
        if unsafe_terminal {
            if !final_state.fenced || final_state.reserved == 0 {
                return Err(Error::State(
                    "ordered SiLU uncertain envelope released".into(),
                ));
            }
        } else if final_state.fenced || final_state.reserved != 0 {
            return Err(Error::State(
                "ordered SiLU final original physical envelope".into(),
            ));
        }
        println!(
            "CUDA_ORDERED_SILU_OPERATOR_PASS B=1,2,4,8 numerical_bit_pairs={compared} fault={fault:?} old_math_launches=12_or13 native_math_launches=1 physical_trace_checkpoints=17 compact_vs_trace=exact CPU_source=exact status_policy=original canaries=exact queue_leases_Ready_quarantine=original fullmodel_HTTP_speed=UNQUALIFIED"
        );
        Ok(())
    })
}
