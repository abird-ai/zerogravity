# Draft Patches — volume 28

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 37563; SHA256 `e0de50e672adf727f9dae26de08a7d06a3ec756cd27bb2913c6e51f97cd080f7`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -0,0 +1,621 @@
+//! Synthetic operator qualification only; no model/token/session/publisher.
+//! Original physical12-step oracle stays unchanged; snapshots only copy U32.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentService};
+use cubecl::{CubeElement, Device, bytes::Bytes};
+use std::sync::Arc;
+use zerogravity_core::accelerator::SourceIdentity;
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+
+const PAD: usize = 4;
+const CANARY: u32 = 0xa5a55a5a;
+
+fn view(lease: &ResidentLease, words: usize) -> Handle {
+    lease
+        .handle()
+        .offset_start((PAD * 4) as u64)
+        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
+}
+
+fn inputs(width: usize, rows: usize, recipe: usize) -> Vec<u32> {
+    const SPECIAL: [u32; 26] = [
+        0, 0x80000000, 1, 0x80000001, 2, 0x80000002, 3, 0x80000003, 0x007fffff, 0x807fffff,
+        0x00800000, 0x80800000, 0x00800001, 0x80800001, 0x3f800000, 0xbf800000, 0x43800000,
+        0xc3800000, 0x42af0000, 0xc2af0000, 0x42ce0000, 0xc2ce0000, 0x43050000, 0xc3050000,
+        0x3f7fffff, 0xbf7fffff,
+    ];
+    (0..width * rows)
+        .map(|at| {
+            let random = (at as u32)
+                .wrapping_mul(0x9e3779b9)
+                .wrapping_add(0x5135791d);
+            match recipe {
+                0 => SPECIAL[at % SPECIAL.len()],
+                1 => (random & 0x80000000) | ((random & 0x7fffffff) % 0x43800001),
+                4 => {
+                    // Analytic neighborhoods of the SOURCE range-reduction ties;
+                    // these select inputs, never reconstruct expected outputs.
+                    let n = [
+                        0.0f32, 1.0, 125.0, 126.0, 127.0, 191.0, 192.0, 193.0, 367.0, 368.0,
+                    ][(at / 257) % 10];
+                    let center = ((n + 0.5) / f32::from_bits(0x3fb8aa3b)).to_bits();
+                    let neighbor = center.wrapping_add((at % 257) as u32).wrapping_sub(128);
+                    neighbor
+                        | if (at / 2570).is_multiple_of(2) {
+                            0
+                        } else {
+                            0x80000000
+                        }
+                }
+                _ => (random & 0x80000000) | 0x3f000000 | (random & 0x007fffff),
+            }
+        })
+        .collect()
+}
+
+/// Post-stage U32 snapshots add no arithmetic to original SiLU operators.
+#[cube(launch_unchecked)]
+fn snapshot(
+    scratch: &[u32],
+    output: &[u32],
+    trace: &mut [u32],
+    #[comptime] count: usize,
+    #[comptime] stage: usize,
+) {
+    let i = ABSOLUTE_POS;
+    if i < count {
+        let s = i * 10;
+        let t = i * 17;
+        if stage < 4usize {
+            trace[t + stage] = scratch[s + stage];
+        } else if stage == 4usize {
+            trace[t + 4] = scratch[s + 4];
+            trace[t + 5] = scratch[s + 5];
+            trace[t + 6] = scratch[s + 6];
+            trace[t + 7] = scratch[s + 7];
+        } else if stage == 5usize {
+            trace[t + 8] = scratch[s + 5];
+        } else if stage == 6usize {
+            trace[t + 9] = scratch[s + 5];
+        } else if stage == 7usize {
+            trace[t + 10] = scratch[s + 2];
+            trace[t + 11] = scratch[s + 8];
+        } else if stage == 8usize {
+            trace[t + 12] = scratch[s + 2];
+        } else if stage == 9usize {
+            trace[t + 13] = scratch[s + 9];
+        } else if stage == 10usize {
+            trace[t + 14] = output[i];
+        } else if stage == 11usize {
+            trace[t + 15] = output[i];
+        } else {
+            trace[t + 16] = output[i];
+        }
+    }
+}
+
+fn arg(handle: &Handle, n: usize) -> BufferArg {
+    // SAFETY: fixture creates full checked original allocations/views before any
+    // launch and pins their leases in the single original physical flight.
+    unsafe { BufferArg::from_raw_parts(handle.clone(), n) }
+}
+fn grid(n: usize) -> CubeCount {
+    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
+}
+
+fn certify(client: &Client, input: &Handle, output: &Handle, status: &Handle, n: usize) {
+    // SAFETY: original buffers/count/status are preflighted; n<=524288<MAX_WORD_OPS.
+    unsafe {
+        crate::resident_bindings::certify_interval::launch_unchecked(
+            client,
+            grid(n),
+            CubeDim::new_1d(32),
+            arg(input, n),
+            arg(status, 1),
+            -256.0,
+            256.0,
+            n,
+        );
+        crate::program_kernels::certify_finite_strided::launch_unchecked(
+            client,
+            grid(n),
+            CubeDim::new_1d(32),
+            arg(output, n),
+            arg(status, 1),
+            0,
+            n as u32,
+            1,
+            n,
+        );
+    }
+}
+
+fn original(
+    client: &Client,
+    input: &Handle,
+    multiplier: Option<&Handle>,
+    scratch: &Handle,
+    output: &Handle,
+    trace: &Handle,
+    n: usize,
+) -> usize {
+    // SAFETY: all complete original leases are already in original submission.
+    unsafe {
+        for stage in 0..12 {
+            crate::silu::silu_step::launch_unchecked(
+                client,
+                grid(n),
+                CubeDim::new_1d(32),
+                arg(input, n),
+                arg(scratch, n * crate::silu::SCRATCH),
+                arg(output, n),
+                n,
+                stage,
+            );
+            snapshot::launch_unchecked(
+                client,
+                grid(n),
+                CubeDim::new_1d(32),
+                arg(scratch, n * 10),
+                arg(output, n),
+                arg(trace, n * 17),
+                n,
+                stage,
+            );
+        }
+        if let Some(m) = multiplier {
+            crate::silu::silu_gate::launch_unchecked(
+                client,
+                grid(n),
+                CubeDim::new_1d(32),
+                arg(m, n),
+                arg(output, n),
+                n,
+            );
+        }
+        snapshot::launch_unchecked(
+            client,
+            grid(n),
+            CubeDim::new_1d(32),
+            arg(scratch, n * 10),
+            arg(output, n),
+            arg(trace, n * 17),
+            n,
+            12,
+        );
+    }
+    12 + usize::from(multiplier.is_some())
+}
+
+fn words(bytes: &Bytes, n: usize) -> Result<Vec<u32>> {
+    if bytes.len() != (n + 2 * PAD) * 4 {
+        return Err(Error::State("ordered SiLU complete canary readback".into()));
+    }
+    let data = bytes
+        .chunks_exact(4)
+        .map(|x| u32::from_ne_bytes(x.try_into().expect("word")))
+        .collect::<Vec<_>>();
+    if data[..PAD]
+        .iter()
+        .chain(&data[PAD + n..])
+        .any(|&x| x != CANARY)
+    {
+        return Err(Error::State(
+            "ordered SiLU prefix/suffix canary overwritten".into(),
+        ));
+    }
+    Ok(data[PAD..PAD + n].to_vec())
+}
+
+#[cfg(target_arch = "x86_64")]
+fn cpu_math_environment() -> Result<u32> {
+    let mut value = 0u32;
+    // SAFETY: writes one valid local MXCSR word; does not alter FP state.
+    unsafe {
+        std::arch::asm!("stmxcsr [{}]",in(reg) &mut value,options(nostack,preserves_flags));
+    }
+    if value & 0xe040 != 0 {
+        return Err(Error::Unsupported(
+            "original CPU SiLU oracle requires RN, FTZ/DAZ off".into(),
+        ));
+    }
+    // Exception sticky flags may change during legitimate source arithmetic.
+    Ok(value & 0xffc0)
+}
+#[cfg(not(target_arch = "x86_64"))]
+fn cpu_math_environment() -> Result<u32> {
+    Err(Error::Unsupported(
+        "closed original Haswell CPU oracle".into(),
+    ))
+}
+
+async fn case(
+    service: &Arc<ResidentService>,
+    rows: usize,
+    width: usize,
+    recipe: usize,
+    multiplied: bool,
+    fault: &str,
+) -> Result<usize> {
+    let n = geometry(rows, width)?;
+    let mut source = inputs(width, rows, recipe);
+    if fault == "nonfinite" {
+        source[0] = 0x7fc12345;
+    }
+    if fault == "range" {
+        source[0] = 0x43800001;
+    }
+    let multiplier = (0..n)
+        .map(|i| {
+            [
+                0x3f800000, 0xbf800000, 0, 0x80000000, 1, 0x80000001, 0x40000000, 0xc0000000,
+            ][i % 8]
+        })
+        .collect::<Vec<_>>();
+    let multiplier = if recipe == 3 {
+        source.clone()
+    } else {
+        multiplier
+    };
+    let sizes = [n, n, n * 10, n, n, n, n * 17, n * 17, 1, 1, 1];
+    let bytes = sizes.iter().map(|n| (n + 2 * PAD) * 4).sum::<usize>();
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: bytes as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(0)?;
+    service
+        .client()
+        .memory_cleanup()
+        .map_err(|e| Error::State(format!("ordered SiLU cold pool cleanup: {e}")))?;
+    let mut leases = Vec::new();
+    for (index, &size) in sizes.iter().enumerate() {
+        let mut data = vec![CANARY; size + 2 * PAD];
+        if index < 2 {
+            data[PAD..PAD + size].copy_from_slice(if index == 0 { &source } else { &multiplier });
+        }
+        if index >= 8 {
+            data[PAD] = if fault == "preexisting" { 5 } else { 0 };
+        }
+        leases.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
+    }
+    let initialized = init.complete_sync().await?;
+    if initialized.outputs().len() != leases.len() {
+        return Err(Error::State("ordered SiLU initialization coverage".into()));
+    }
+    drop(initialized);
+    let handles = leases
+        .iter()
+        .zip(sizes)
+        .map(|(l, n)| view(l, n))
+        .collect::<Vec<_>>();
+    let client = service.client();
+    let multiplier_handle = multiplied.then_some(&handles[if recipe == 3 { 0 } else { 1 }]);
+    let binding = |output: usize, trace: Option<usize>| Bindings {
+        input: &handles[0],
+        multiplier: multiplier_handle,
+        output: &handles[output],
+        trace: trace.map(|i| &handles[i]),
+    };
+    let traced = PreparedOrderedSilu::prepare(client, rows, width, binding(4, Some(7)))?;
+    let compact = PreparedOrderedSilu::prepare(client, rows, width, binding(5, None))?;
+    let stable = (
+        service.resident_bytes()?,
+        service.uploaded_value_bytes(),
+        service.pending_events()?,
+    );
+    let invalid = PreparedOrderedSilu::prepare(
+        client,
+        rows,
+        width,
+        Bindings {
+            input: &handles[0],
+            multiplier: None,
+            output: &handles[0],
+            trace: None,
+        },
+    );
+    if !matches!(invalid,Err(ref e) if e.to_string().contains("alias")) {
+        return Err(Error::State("ordered SiLU prelaunch alias accepted".into()));
+    }
+    let mut foreign = handles[0].clone();
+    foreign.service.service = std::any::TypeId::of::<()>();
+    if PreparedOrderedSilu::prepare(
+        client,
+        rows,
+        width,
+        Bindings {
+            input: &foreign,
+            multiplier: None,
+            output: &handles[4],
+            trace: None,
+        },
+    )
+    .is_ok()
+        || stable
+            != (
+                service.resident_bytes()?,
+                service.uploaded_value_bytes(),
+                service.pending_events()?,
+            )
+    {
+        return Err(Error::State(
+            "ordered SiLU cold refusal changed original ledger/uploads".into(),
+        ));
+    }
+    for issue in 0..3 {
+        let mut bad = handles[0].clone();
+        if issue == 0 {
+            bad.offset_end = Some(bad.size() + 1);
+        } else if issue == 1 {
+            bad.offset_start = Some(bad.offset_start.unwrap_or(0) + 1);
+        } else {
+            bad.offset_end = Some(bad.offset_end.unwrap_or(0) + 4);
+        }
+        if PreparedOrderedSilu::prepare(
+            client,
+            rows,
+            width,
+            Bindings {
+                input: &bad,
+                multiplier: None,
+                output: &handles[4],
+                trace: None,
+            },
+        )
+        .is_ok()
+        {
+            return Err(Error::State(
+                "ordered SiLU malformed/short view accepted".into(),
+            ));
+        }
+    }
+    if PreparedOrderedSilu::prepare(
+        client,
+        rows,
+        width,
+        Bindings {
+            input: &handles[0],
+            multiplier: None,
+            output: &handles[7],
+            trace: Some(&handles[7]),
+        },
+    )
+    .is_ok()
+    {
+        return Err(Error::State(
+            "ordered SiLU trace/output alias accepted".into(),
+        ));
+    }
+    // Cancelling an unsubmitted original queue ticket cannot launch this kernel.
+    let queue_before = service.queue().stats()?;
+    let cancelled = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: 4,
+        }],
+        None,
+    )?;
+    drop(cancelled);
+    let queue_after = service.queue().stats()?;
+    if queue_after.cancelled_queued != queue_before.cancelled_queued + 1
+        || stable
+            != (
+                service.resident_bytes()?,
+                service.uploaded_value_bytes(),
+                service.pending_events()?,
+            )
+    {
+        return Err(Error::State(
+            "ordered SiLU prelaunch cancellation touched physical data".into(),
+        ));
+    }
+    let accesses = leases
+        .iter()
+        .enumerate()
+        .map(|(i, l)| {
+            (
+                l.clone(),
+                if i < 2 {
+                    DeviceAccess::Read
+                } else {
+                    DeviceAccess::Write
+                },
+            )
+        })
+        .collect::<Vec<_>>();
+    let reads = leases
+        .iter()
+        .map(|l| l.readback(0, l.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
+    let uploads = service.uploaded_value_bytes();
+    let flight = service.submit(0, &accesses)?;
+    let old = original(
+        client,
+        &handles[0],
+        multiplier_handle,
+        &handles[2],
+        &handles[3],
+        &handles[6],
+        n,
+    );
+    let native = traced.launch(client)? + compact.launch(client)?;
+    for (out, status) in [(3, 8), (4, 9), (5, 10)] {
+        certify(client, &handles[0], &handles[out], &handles[status], n);
+    }
+    if fault == "unknown" {
+        drop(flight);
+        drop(slot);
+        let budget = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
+        if !budget.fenced
+            || budget.reserved == 0
+            || service.pending_events()? == 0
+            || service.ensure_usable().is_ok()
+            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
+        {
+            return Err(Error::State(
+                "ordered SiLU unknown completion escaped original fence/quarantine".into(),
+            ));
+        }
+        return Ok(0);
+    }
+    let mxcsr = cpu_math_environment()?;
+    let mut checked = 0;
+    let mut rejected_status = false;
+    let result=flight.complete_with_readback(&leases[2..],&reads,|read| {
+        if read.len()!=sizes.len() {return Err(Error::State("ordered SiLU full readback coverage".into()));}
+        let data=read.iter().zip(sizes).map(|(b,n)|words(b,n)).collect::<Result<Vec<_>>>()?;
+        if data[0]!=source || data[1]!=multiplier {return Err(Error::State("ordered SiLU read-only input changed".into()));}
+        if data[8]!=data[9] || data[8]!=data[10] {return Err(Error::State("ordered SiLU original finite/status bits changed".into()));}
+        if !fault.is_empty() && fault!="known-ready-abandon" {
+            let expected=match fault {"preexisting"=>5,"nonfinite"=>18,"range"=>16,_=>0};
+            if data[8][0]!=expected {return Err(Error::State("ordered SiLU invalid/preexisting status bits changed".into()));}
+            rejected_status=true;
+            return Err(Error::State("ordered SiLU original status rejected before Ready".into()));
+        }
+        if data[8][0]!=0 {return Err(Error::State("ordered SiLU unexpected finite/status refusal".into()));}
+        for (at,&source_word) in source.iter().enumerate() {
+            if data[3][at]!=data[4][at] || data[3][at]!=data[5][at] {
+                return Err(Error::State(format!("ordered SiLU physical final bits at{at}: old={:08x} trace={:08x} compact={:08x}",data[3][at],data[4][at],data[5][at])));
+            }
+            for checkpoint in 0..17 {
+                if data[6][at*17+checkpoint]!=data[7][at*17+checkpoint] {
+                    return Err(Error::State(format!("ordered SiLU physical checkpoint{checkpoint} at{at}: old={:08x} native={:08x}",data[6][at*17+checkpoint],data[7][at*17+checkpoint])));
+                }
+            }
+            let x=f32::from_bits(source_word);
+            let cpu_exp=zerogravity_cpu::math::ggml_source_exp(0.0-x);
+            if data[6][at*17+12]!=cpu_exp.to_bits() || data[6][at*17+13]!=(1.0+cpu_exp).to_bits() {
+                return Err(Error::State(format!("ordered SiLU original CPU exp/denominator bits at{at}")));
+            }
+            let cpu=zerogravity_cpu::math::ggml_source_silu(x);
+            let cpu=if multiplied {f32::from_bits(multiplier[at])*cpu} else {cpu};
+            if data[3][at]!=cpu.to_bits() {
+                return Err(Error::State(format!("ordered SiLU original CPU source bits at{at}: GPU={:08x} CPU={:08x}",data[3][at],cpu.to_bits())));
+            }
+            checked+=22;
+        }
+        if cpu_math_environment()?!=mxcsr {return Err(Error::State("ordered SiLU CPU oracle math state changed".into()));}
+        Ok(())
+    }).await;
+    if rejected_status {
+        drop(slot);
+        if !matches!(result,Err(ref e) if e.to_string().contains("original status rejected"))
+            || service.ensure_usable().is_ok()
+            || service.pending_events()? == 0
+            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
+        {
+            return Err(Error::State(
+                "ordered SiLU invalid certificate escaped original quarantine".into(),
+            ));
+        }
+        return Ok(0);
+    }
+    let (ready, data) = result?;
+    if old != 12 + usize::from(multiplied)
+        || native != 2
+        || ready.outputs().len() != leases.len() - 2
+        || data.len() != leases.len()
+        || service.uploaded_value_bytes() != uploads
+    {
+        return Err(Error::State(
+            "ordered SiLU actual source launch/Ready/read/upload coverage".into(),
+        ));
+    }
+    // Known Ready private-candidate abandonment publishes no SessionAuthority/
+    // StateRoot. Fullmodel per-request cancellation remains a separate gate.
+    drop(ready);
+    drop(data);
+    slot.finish()?;
+    Ok(checked)
+}
+
+#[test]
+#[ignore = "ROOT release CUDA original queue/leases physical differential only"]
+fn physical_ordered_silu() -> Result<()> {
+    futures_lite::future::block_on(async {
+        if cfg!(debug_assertions) {
+            return Err(Error::Unsupported(
+                "ordered SiLU qualification requires release".into(),
+            ));
+        }
+        let fault = std::env::var("ZG_ORDERED_SILU_FAULT").unwrap_or_default();
+        if !matches!(
+            fault.as_str(),
+            "" | "unknown" | "preexisting" | "nonfinite" | "range" | "known-ready-abandon"
+        ) {
+            return Err(Error::Malformed("ordered SiLU exact fault selector".into()));
+        }
+        let source = SourceIdentity::for_new_allocation()?;
+        let service = ResidentService::new(
+            Device::Cuda(cubecl::device::CudaDevice::new(0)),
+            source,
+            ResourceOwner::new(
+                "ordered-silu-synthetic-operator",
+                "original-physical-oracle",
+            )?,
+            256 * 1024 * *2,
+            640 * 1024 * *2,
+        )?;
+        let key = service.physical_key();
+        let mut compared = 0;
+        if fault.is_empty() {
+            for rows in [1, 2, 4, 8] {
+                for width in [1, 31, 33, 128, 2048, 6144, 11008, 65536] {
+                    for recipe in 0..3 {
+                        for multiplied in [false, true] {
+                            compared += case(&service, rows, width, recipe, multiplied, "").await?;
+                            if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+                                return Err(Error::State(
+                                    "ordered SiLU live private lease/event after case".into(),
+                                ));
+                            }
+                        }
+                    }
+                }
+            }
+            for rows in [1, 2, 4, 8] {
+                compared += case(&service, rows, 33, 3, true, "").await?;
+                for multiplied in [false, true] {
+                    compared += case(&service, rows, 5140, 4, multiplied, "").await?;
+                }
+            }
+        } else {
+            compared += case(&service, 8, 33, 0, true, &fault).await?;
+        }
+        let unsafe_terminal = matches!(
+            fault.as_str(),
+            "unknown" | "preexisting" | "nonfinite" | "range"
+        );
+        drop(service);
+        let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
+        if unsafe_terminal {
+            if !final_state.fenced || final_state.reserved == 0 {
+                return Err(Error::State(
+                    "ordered SiLU uncertain envelope released".into(),
+                ));
+            }
+        } else if final_state.fenced || final_state.reserved != 0 {
+            return Err(Error::State(
+                "ordered SiLU final original physical envelope".into(),
+            ));
+        }
+        println!(
+            "CUDA_ORDERED_SILU_OPERATOR_PASS B=1,2,4,8 numerical_bit_pairs={compared} fault={fault:?} old_math_launches=12_or13 native_math_launches=1 physical_trace_checkpoints=17 compact_vs_trace=exact CPU_source=exact status_policy=original canaries=exact queue_leases_Ready_quarantine=original fullmodel_HTTP_speed=UNQUALIFIED"
+        );
+        Ok(())
+    })
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/ordered_silu/tests.rs
@@ -0,0 +1,38 @@
+use super::*;
+
+#[test]
+fn native_geometry_is_exact_without_padding() -> Result<()> {
+    for rows in [1, 2, 4, 8] {
+        for width in [1, 31, 32, 33, 65_536] {
+            assert_eq!(geometry(rows, width)?, rows * width);
+        }
+    }
+    for (rows, width) in [
+        (0, 1),
+        (3, 1),
+        (16, 1),
+        (8, 0),
+        (1, 65_537),
+        (8, usize::MAX),
+    ] {
+        assert!(geometry(rows, width).is_err());
+    }
+    Ok(())
+}
+
+#[test]
+fn checked_spans_never_hide_short_misaligned_or_overflowed_views() -> Result<()> {
+    assert_eq!(checked_span(80, 16, 16, 12)?, 16..64);
+    for (size, start, trim, words) in [
+        (80, 1, 15, 1),
+        (80, 16, 81, 1),
+        (80, 64, 20, 1),
+        (80, 16, 16, 13),
+        (80, 16, 16, 0),
+        (80, 16, 16, usize::MAX),
+        (u64::MAX, u64::MAX - 1, 0, 1),
+    ] {
+        assert!(checked_span(size, start, trim, words).is_err());
+    }
+    Ok(())
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/ordered_silu.rs
@@ -0,0 +1,356 @@
+//! Dormant CUDA-only ordered version of the original12-launch source SiLU.
+//! No exp intrinsic, new polynomial, source profile or production selection.
+use crate::source_accumulation::{SourceAccumulation, ensure_capability};
+use crate::source_float::{correct_quotient, round_source, widen_source};
+use cubecl::{client::Client, prelude::*, server::Handle};
+use cubecl_common::device::ServiceId;
+use std::ops::Range;
+use zerogravity_core::{Error, Result};
+
+const MAX_ROW_WORDS: usize = 65_536;
+const LANES: u32 = 32;
+pub(crate) const TRACE_WORDS: usize = 17;
+
+fn geometry(rows: usize, width: usize) -> Result<usize> {
+    if !matches!(rows, 1 | 2 | 4 | 8) || width == 0 || width > MAX_ROW_WORDS {
+        return Err(Error::Unsupported(
+            "ordered SiLU native row geometry".into(),
+        ));
+    }
+    rows.checked_mul(width)
+        .filter(|&n| u32::try_from(n).is_ok())
+        .ok_or_else(|| Error::Limit("ordered SiLU aggregate domain".into()))
+}
+
+fn checked_span(size: u64, start: u64, trim: u64, words: usize) -> Result<Range<u64>> {
+    let end = size
+        .checked_sub(trim)
+        .ok_or_else(|| Error::State("ordered SiLU end exceeds allocation".into()))?;
+    let used = end
+        .checked_sub(start)
+        .ok_or_else(|| Error::State("ordered SiLU start exceeds end".into()))?;
+    let bytes = u64::try_from(words)
+        .ok()
+        .and_then(|n| n.checked_mul(4))
+        .ok_or_else(|| Error::Limit("ordered SiLU byte domain".into()))?;
+    if bytes == 0 || !start.is_multiple_of(4) || !used.is_multiple_of(4) || used < bytes {
+        return Err(Error::State("ordered SiLU buffer alignment/size".into()));
+    }
+    Ok(start..start + bytes)
+}
+
+fn binding(client: &Client, handle: &Handle, words: usize) -> Result<Range<u64>> {
+    if handle.service != client.service_id() {
+        return Err(Error::State("ordered SiLU foreign service".into()));
+    }
+    checked_span(
+        handle.size(),
+        handle.offset_start.unwrap_or(0),
+        handle.offset_end.unwrap_or(0),
+        words,
+    )
+}
+
+pub(crate) struct Bindings<'a> {
+    pub(crate) input: &'a Handle,
+    pub(crate) multiplier: Option<&'a Handle>,
+    pub(crate) output: &'a Handle,
+    pub(crate) trace: Option<&'a Handle>,
+}
+
+/// Caller preflights ALL model nodes before its original flight; leases, source,
+/// status, interval/finite checks, Ready/cancel/quarantine remain original owners.
+/// This binding allocates/uploads no device storage, submits no physical work,
+/// and never clears status.
+pub(crate) struct PreparedOrderedSilu {
+    service: ServiceId,
+    input: Handle,
+    multiplier: Handle,
+    output: Handle,
+    trace: Handle,
+    count: usize,
+    multiplied: bool,
+    record: bool,
+}
+
+impl PreparedOrderedSilu {
+    pub(crate) fn prepare(
+        client: &Client,
+        rows: usize,
+        width: usize,
+        bindings: Bindings<'_>,
+    ) -> Result<Self> {
+        let Bindings {
+            input,
+            multiplier,
+            output,
+            trace,
+        } = bindings;
+        if !crate::activation_quant_fused::is_cuda_service(client) {
+            return Err(Error::Unsupported(
+                "ordered SiLU CUDA-only candidate".into(),
+            ));
+        }
+        ensure_capability(client, SourceAccumulation::WideF64)?;
+        let count = geometry(rows, width)?;
+        let hardware = &client.properties().hardware;
+        if hardware.max_bindings < 4
+            || hardware.max_units_per_cube < LANES
+            || hardware.max_cube_dim.0 < LANES
+            || hardware.max_cube_dim.1 == 0
+            || hardware.max_cube_dim.2 == 0
+            || hardware.max_cube_count.0 < (count as u32).div_ceil(LANES)
+            || hardware.max_cube_count.1 == 0
+            || hardware.max_cube_count.2 == 0
+        {
+            return Err(Error::Unsupported(
+                "ordered SiLU CUDA grid/binding limits".into(),
+            ));
+        }
+        let input_range = binding(client, input, count)?;
+        let output_range = binding(client, output, count)?;
+        let reads = [
+            Some((input, input_range)),
+            if let Some(m) = multiplier {
+                Some((m, binding(client, m, count)?))
+            } else {
+                None
+            },
+        ];
+        let writes = [
+            Some((output, output_range)),
+            if let Some(t) = trace {
+                Some((t, binding(client, t, count * TRACE_WORDS)?))
+            } else {
+                None
+            },
+        ];
+        for (index, left) in writes.iter().enumerate() {
+            if let Some((left, left_span)) = left {
+                for (right, right_span) in reads.iter().chain(writes[..index].iter()).flatten() {
+                    if (*left).clone().binding().memory.id()
+                        == (*right).clone().binding().memory.id()
+                        && left_span.start < right_span.end
+                        && right_span.start < left_span.end
+                    {
+                        return Err(Error::State("ordered SiLU writable alias".into()));
+                    }
+                }
+            }
+        }
+        Ok(Self {
+            service: client.service_id(),
+            input: input.clone(),
+            // Inactive aliases have no device access under compile-time branches.
+            multiplier: multiplier.unwrap_or(input).clone(),
+            output: output.clone(),
+            trace: trace.unwrap_or(output).clone(),
+            count,
+            multiplied: multiplier.is_some(),
+            record: trace.is_some(),
+        })
+    }
+
+    /// ONE actual source launch; no inferred driver count or timing claim.
+    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
+        if client.service_id() != self.service {
+            return Err(Error::State("ordered SiLU service changed".into()));
+        }
+        // SAFETY: complete spans, exact original service and disjoint writes were
+        // checked cold. Original flight must already retain all owning leases.
+        unsafe {
+            ordered::launch_unchecked(
+                client,
+                CubeCount::Static((self.count as u32).div_ceil(LANES), 1, 1),
+                CubeDim::new_1d(LANES),
+                BufferArg::from_raw_parts(self.input.clone(), self.count),
+                BufferArg::from_raw_parts(self.multiplier.clone(), self.count),
+                BufferArg::from_raw_parts(self.output.clone(), self.count),
+                BufferArg::from_raw_parts(
+                    self.trace.clone(),
+                    if self.record {
+                        self.count * TRACE_WORDS
+                    } else {
+                        self.count
+                    },
+                ),
+                self.count,
+                self.multiplied,
+                self.record,
+            );
+        }
+        Ok(1)
+    }
+}
+
+// Explicit PTX .rn instructions, no .ftz. gpu_asm without `pure` lowers to
+// volatile asm; f32 output constraints retain each rounded source result even
+// when immediately widened by the next stage. No float-cast-only assumption.
+#[cube]
+fn rn32(value: f64) -> f32 {
+    let output: f32;
+    gpu_asm!("cvt.rn.f32.f64 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
+    output
+}
+#[cube]
+fn wide(value: f32) -> f64 {
+    let output: f64;
+    gpu_asm!("cvt.f64.f32 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
+    output
+}
+#[cube]
+fn add64(left: f64, right: f64) -> f64 {
+    let output: f64;
+    gpu_asm!("add.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
+    output
+}
+#[cube]
+fn sub64(left: f64, right: f64) -> f64 {
+    let output: f64;
+    gpu_asm!("sub.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
+    output
+}
+#[cube]
+fn mul64(left: f64, right: f64) -> f64 {
+    let output: f64;
+    gpu_asm!("mul.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
+    output
+}
+#[cube]
+fn div64(left: f64, right: f64) -> f64 {
+    let output: f64;
+    gpu_asm!("div.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
+    output
+}
+#[cube]
+fn fma64(a: f64, b: f64, c: f64) -> f64 {
+    let output: f64;
+    gpu_asm!("fma.rn.f64 {out}, {a}, {b}, {c};",out=out(_) output,a=in(_) a,b=in(_) b,c=in(_) c,options(nomem));
+    output
+}
+#[cube]
+fn negative_input(input: f32) -> f32 {
+    let output: f32;
+    gpu_asm!("sub.rn.f32 {out}, 0f00000000, {value};",out=out(_) output,value=in(_) input,options(nomem));
+    output
+}
+#[cube]
+fn stored_bits(input: f32) -> u32 {
+    let output: u32;
+    gpu_asm!("mov.b32 {out}, {value};",out=out(_) output,value=in(_) input,options(nomem));
+    output
+}
+#[cube]
+fn negate(input: f32) -> f32 {
+    f32::reinterpret(u32::reinterpret(input) ^ 0x80000000u32)
+}
+#[cube]
+fn wide_fma(a: f32, b: f32, c: f32) -> f32 {
+    // A finite f32 product is exact in f64. One explicit F64 FMA has the
+    // same f64 result as the original potentially contracted wide expression.
+    rn32(fma64(wide(a), wide(b), wide(c)))
+}
+
+#[cube(launch_unchecked)]
+fn ordered(
+    input: &[u32],
+    multiplier: &[u32],
+    output: &mut [u32],
+    trace: &mut [u32],
+    #[comptime] count: usize,
+    #[comptime] multiplied: bool,
+    #[comptime] record: bool,
+) {
+    let i = ABSOLUTE_POS;
+    if i < count {
+        let original = f32::reinterpret(input[i]);
+        let x = negative_input(original);
+        let z = wide_fma(
+            x,
+            f32::reinterpret(0x3fb8aa3bu32),
+            f32::reinterpret(0x4b400000u32),
+        );
+        let n = rn32(sub64(wide(z), wide(f32::reinterpret(0x4b400000u32))));
+        let b0 = wide_fma(negate(n), f32::reinterpret(0x3f317200u32), x);
+        let b = wide_fma(negate(n), f32::reinterpret(0x35bfbe8eu32), b0);
+        let u = rn32(mul64(wide(b), wide(b)));
+        let j0 = wide_fma(
+            f32::reinterpret(0x3c072010u32),
+            b,
+            f32::reinterpret(0x3d2b9f17u32),
+        );
+        let j1 = wide_fma(
+            f32::reinterpret(0x3e2aaf33u32),
+            b,
+            f32::reinterpret(0x3efffedbu32),
+        );
+        let j2 = rn32(mul64(wide(f32::reinterpret(0x3f7ffff6u32)), wide(b)));
+        let j3 = wide_fma(j0, u, j1);
+        let j4 = wide_fma(j3, u, j2);
+        let e = u32::reinterpret(z) << 23u32;
+        let k = f32::reinterpret(e + 0x3f800000u32);
+        let mut base = wide_fma(j4, k, k);
+        let mut factor = 1.0f32;
+        if f32::reinterpret(u32::reinterpret(n) & 0x7fffffffu32) > 126.0f32 {
+            let mut g = 0u32;
+            if n <= 0.0f32 {
+                g = 0x82000000u32;
+            }
+            let s1 = f32::reinterpret(g + 0x7f000000u32);
+            let s2 = f32::reinterpret(e - g);
+            if f32::reinterpret(u32::reinterpret(n) & 0x7fffffffu32) > 192.0f32 {
+                base = s1;
+            } else {
+                base = wide_fma(s2, j4, s2);
+            }
+            factor = s1;
+        }
+        let exponential = rn32(mul64(wide(base), wide(factor)));
+        let denominator = rn32(add64(1.0f64, wide(exponential)));
+        // Reuse exact original manual subnormal packing and neighboring-quotient
+        // correction. Opaque F32 register bits separate these two stages.
+        let estimate_bits = stored_bits(round_source(
+            div64(widen_source(original), widen_source(denominator)),
+            input[i] & 0x80000000u32,
+        ));
+        let quotient = correct_quotient(original, denominator, f32::reinterpret(estimate_bits));
+        let quotient_bits = stored_bits(quotient);
+        let mut result = quotient_bits;
+        if multiplied {
+            let m = f32::reinterpret(multiplier[i]);
+            result = stored_bits(round_source(
+                mul64(
+                    widen_source(m),
+                    widen_source(f32::reinterpret(quotient_bits)),
+                ),
+                (multiplier[i] ^ quotient_bits) & 0x80000000u32,
+            ));
+        }
+        output[i] = result;
+        if record {
+            let t = i * 17;
+            trace[t] = u32::reinterpret(z);
+            trace[t + 1] = u32::reinterpret(n);
+            trace[t + 2] = u32::reinterpret(b0);
+            trace[t + 3] = u32::reinterpret(b);
+            trace[t + 4] = u32::reinterpret(u);
+            trace[t + 5] = u32::reinterpret(j0);
+            trace[t + 6] = u32::reinterpret(j1);
+            trace[t + 7] = u32::reinterpret(j2);
+            trace[t + 8] = u32::reinterpret(j3);
+            trace[t + 9] = u32::reinterpret(j4);
+            trace[t + 10] = u32::reinterpret(base);
+            trace[t + 11] = u32::reinterpret(factor);
+            trace[t + 12] = u32::reinterpret(exponential);
+            trace[t + 13] = u32::reinterpret(denominator);
+            trace[t + 14] = estimate_bits;
+            trace[t + 15] = quotient_bits;
+            trace[t + 16] = result;
+        }
+    }
+}
+
+#[cfg(all(test, feature = "qualification", feature = "cuda"))]
+mod qualification;
+#[cfg(test)]
+mod tests;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/frozen/ROOT_WIRING.patch

Original UTF-8 bytes: 275; SHA256 `93b30ed69411fc007b14ba1c04ca9e6aa6f3aea445b7755f6771b1b246a8d92f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -91,3 +91,6 @@
     ResidentSnapshot, ResidentToken,
 };
 pub use resident_provider::CubeResidentProvider;
+
+#[cfg(all(test, feature = "qualification", feature = "cuda"))]
+mod ordered_silu;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-compiler-witness-review-20261020/DUMP_COLLISION_ONLY.patch

Original UTF-8 bytes: 5466; SHA256 `eeacb5ea3ae18d4ec4eab731bcba934375846533940cb5f3df21c2b8e036f3cd`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs
+++ b/crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs
@@ -579,12 +579,9 @@
     }
 }
 
-/// Writes the PTX for `kernel_id` under the directory named by `CUBECL_CUDA_DUMP_PTX`, if that
-/// variable is set.
-///
-/// Both backends end up here, so the two can be diffed instruction for instruction. That is the
-/// comparison worth making: the PTX is deterministic, where a wall-clock measurement on a laptop
-/// GPU is not.
+/// Writes the exact PTX handed to the driver only when the explicit dump switch is set.
+/// Each attempt has a unique create-new path and full compilation id sidecar;
+/// truncated display suffixes cannot overwrite settings-distinct witnesses.
 fn dump_ptx(kernel_id: &KernelId, ptx: &[c_char]) {
     let Some(dir) = std::env::var_os("CUBECL_CUDA_DUMP_PTX") else {
         return;
@@ -594,18 +591,54 @@
         return;
     }
 
-    // The id is a type path plus its settings, so it carries every character a path cannot.
-    let name: String = kernel_id
-        .to_string()
-        .chars()
-        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
-        .collect();
-    // ...and is long enough to blow past NAME_MAX on its own.
-    let name = &name[name.len().saturating_sub(180)..];
-
+    // Same complete Debug id as the full compilation logger. The hash is an
+    // index, never identity proof; sequence + create_new prevent overwriting.
+    let id = format!("{kernel_id:#?}");
     // SAFETY: the PTX handed to the driver is a null-terminated C string.
     let text = unsafe { CStr::from_ptr(ptx.as_ptr()) };
-    let _ = std::fs::write(dir.join(format!("{name}.ptx")), text.to_bytes());
+    // Diagnostic I/O remains best effort; a witness validator must reject any
+    // missing, partial or unpaired artifact instead of inferring coverage.
+    let _ = write_ptx_dump(&dir, &id, text.to_bytes());
+}
+
+static PTX_DUMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
+
+fn write_ptx_dump(
+    dir: &std::path::Path,
+    kernel_id: &str,
+    ptx: &[u8],
+) -> std::io::Result<std::path::PathBuf> {
+    use std::io::Write;
+    use std::sync::atomic::Ordering;
+
+    let hash = StableHasher::hash_one(&kernel_id);
+    for _ in 0..64 {
+        let sequence = PTX_DUMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
+        let stem = format!("ptx-p{}-s{sequence:016x}-k{hash:032x}", std::process::id());
+        let path = dir.join(format!("{stem}.ptx"));
+        let mut output = match std::fs::OpenOptions::new()
+            .write(true)
+            .create_new(true)
+            .open(&path)
+        {
+            Ok(output) => output,
+            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
+            Err(error) => return Err(error),
+        };
+        // Reserve both files before writing. Never overwrite an existing id or
+        // PTX, even on hash collision, sequence wrap, pid reuse or repeated id.
+        let mut id = std::fs::OpenOptions::new()
+            .write(true)
+            .create_new(true)
+            .open(path.with_extension("kernel-id.txt"))?;
+        id.write_all(kernel_id.as_bytes())?;
+        output.write_all(ptx)?;
+        return Ok(path);
+    }
+    Err(std::io::Error::new(
+        std::io::ErrorKind::AlreadyExists,
+        "CUDA PTX diagnostic path collisions exhausted",
+    ))
 }
 
 /// A refused launch error : maps to a poisoned-device error if that call happened right after
@@ -624,6 +657,46 @@
 mod tests {
     use super::cache_namespace;
     use crate::compiler::CudaBackend;
+
+    #[test]
+    fn ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct() {
+        struct OwnedDirectory(std::path::PathBuf);
+        impl Drop for OwnedDirectory {
+            fn drop(&mut self) {
+                let _ = std::fs::remove_dir_all(&self.0);
+            }
+        }
+        let sequence =
+            super::PTX_DUMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
+        let path = std::env::temp_dir().join(format!(
+            "cubecl-ptx-dump-test-{}-{sequence}",
+            std::process::id()
+        ));
+        std::fs::create_dir(&path).unwrap();
+        let directory = OwnedDirectory(path);
+        // Synthetic transport fixtures, never compiler or numerical witnesses.
+        let suffix = "same settings suffix ".repeat(20);
+        let first = format!("full-first-id {suffix}");
+        let second = format!("full-second-id {suffix}");
+        let samples = [
+            (first.as_str(), b"synthetic ptx one\n".as_slice()),
+            (second.as_str(), b"synthetic ptx two\n".as_slice()),
+            (first.as_str(), b"synthetic repeated id\n".as_slice()),
+        ];
+        let mut paths = Vec::new();
+        for (id, ptx) in samples {
+            let path = super::write_ptx_dump(&directory.0, id, ptx).unwrap();
+            assert!(!paths.contains(&path));
+            assert!(path.file_name().unwrap().len() < 255);
+            assert_eq!(std::fs::read(&path).unwrap(), ptx);
+            assert_eq!(
+                std::fs::read_to_string(path.with_extension("kernel-id.txt")).unwrap(),
+                id
+            );
+            paths.push(path);
+        }
+        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 6);
+    }
 
     #[test]
     fn cpp_cache_namespace_is_stable() {

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 12278; SHA256 `b8a2f59f32e3d86b6784c7dc1bda0889143d21f922f37df2c7c751b2fdf623f5`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/resident_program.rs b/crates/zerogravity-gpu/src/resident_program.rs
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -1009,6 +1009,8 @@
                 _ => Ok(None),
             })
             .collect::<Result<Vec<_>>>()?;
+        #[cfg(feature = "cuda")]
+        program.preflight_silu(client, 1)?;
         for handle in program.owned_handles() {
             program.owned_ranges.insert(BufferRange::new(&handle)?)?;
         }
@@ -1106,6 +1108,17 @@
         handles
     }
     pub fn validate_runtime(
+        &self,
+        client: &Client,
+        p: RuntimeParameters,
+        bindings: &[StateBinding],
+    ) -> Result<()> {
+        #[cfg(feature = "cuda")]
+        self.preflight_silu(client, 1)?;
+        self.validate_state_runtime(client, p, bindings)
+    }
+
+    fn validate_state_runtime(
         &self,
         client: &Client,
         p: RuntimeParameters,
@@ -1616,42 +1629,7 @@
                     );
                     Ok(4)
                 }
-                PlanOperation::Silu { multiplied } => {
-                    let n = input(0).elements;
-                    crate::resident_bindings::certify_interval::launch_unchecked(
-                        client,
-                        grid(n),
-                        threads(),
-                        arg(&input(0).handle, n),
-                        arg(&self.status, 1),
-                        -256.0,
-                        256.0,
-                        n,
-                    );
-                    for stage in 0..12 {
-                        crate::silu::silu_step::launch_unchecked(
-                            client,
-                            grid(n),
-                            threads(),
-                            arg(&input(0).handle, n),
-                            arg(s(0), n * crate::silu::SCRATCH),
-                            arg(&output(0).handle, n),
-                            n,
-                            stage,
-                        );
-                    }
-                    if *multiplied {
-                        crate::silu::silu_gate::launch_unchecked(
-                            client,
-                            grid(n),
-                            threads(),
-                            arg(&input(1).handle, n),
-                            arg(&output(0).handle, n),
-                            n,
-                        );
-                    }
-                    Ok(13 + usize::from(*multiplied))
-                }
+                PlanOperation::Silu { .. } => self.launch_silu(client, node, 1),
                 PlanOperation::Add => {
                     let n = input(0).elements;
                     crate::elementwise::residual_add::launch_unchecked(
@@ -2684,3 +2662,5 @@
 #[cfg(test)]
 #[path = "resident_program/typed_refusal_tests.rs"]
 mod typed_refusal_tests;
+
+mod silu;
diff --git a/crates/zerogravity-gpu/src/resident_program/batch.rs b/crates/zerogravity-gpu/src/resident_program/batch.rs
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -323,6 +323,8 @@
                 .map(Some)
             })
             .collect::<Result<Vec<_>>>()?;
+        #[cfg(feature = "cuda")]
+        inner.preflight_silu(client, rows)?;
         Ok(Self {
             inner,
             rows,
@@ -388,10 +390,13 @@
         if parameters.len() != self.rows || bindings.len() != self.rows {
             return Err(Error::Malformed("resident batch row count changed".into()));
         }
+        #[cfg(feature = "cuda")]
+        self.inner.preflight_silu(client, self.rows)?;
         let mut priors = Vec::new();
         let mut candidates = Vec::new();
         for (parameter, binding) in parameters.iter().zip(bindings) {
-            self.inner.validate_runtime(client, *parameter, binding)?;
+            self.inner
+                .validate_state_runtime(client, *parameter, binding)?;
             for node in self.inner.plan.nodes() {
                 if let PlanOperation::Embed { tensor, .. } = &node.operation
                     && parameter.token as usize >= tensor.rows
@@ -1299,6 +1304,7 @@
                 }
                 // Original source kernels, accumulation order, physical rounding
                 // boundaries and attention padding remain the authority.
+                PlanOperation::Silu { .. } => self.inner.launch_silu(client, node, self.rows),
                 _ => self.inner.launch_node(client, index, node, p, &self.packed),
             }
         }
diff --git a/crates/zerogravity-gpu/src/resident_program/silu.rs b/crates/zerogravity-gpu/src/resident_program/silu.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/silu.rs
@@ -0,0 +1,157 @@
+//! Same original SiLU buffers/certificates; no persistent binding metadata.
+use super::*;
+
+#[cfg(feature = "cuda")]
+fn ordered_width(rows: usize, elements: usize) -> Result<Option<usize>> {
+    if !matches!(rows, 1 | 2 | 4 | 8) || !elements.is_multiple_of(rows) {
+        return Err(Error::State("resident SiLU row partition changed".into()));
+    }
+    let width = elements / rows;
+    Ok(crate::ordered_silu::supports_geometry(rows, width).then_some(width))
+}
+
+impl ResidentProgram {
+    #[cfg(feature = "cuda")]
+    fn ordered_silu_binding(
+        &self,
+        client: &Client,
+        node: &PlanNode,
+        rows: usize,
+    ) -> Result<Option<crate::ordered_silu::PreparedOrderedSilu>> {
+        if !crate::activation_quant_fused::is_cuda_service(client) {
+            return Ok(None);
+        }
+        if self.plan.numerical_profile()
+            != zerogravity_core::accelerator::program::ProgramNumericalProfile::GgufSourceV1
+            || self.plan.is_operator_slice()
+        {
+            return Err(Error::Unsupported(
+                "resident SiLU numerical contract".into(),
+            ));
+        }
+        let PlanOperation::Silu { multiplied } = &node.operation else {
+            return Err(Error::State("resident SiLU operation changed".into()));
+        };
+        let input = |index: usize| -> Result<&DeviceBuffer> {
+            node.inputs
+                .get(index)
+                .and_then(|id| self.values.get(id.0 as usize))
+                .ok_or_else(|| Error::State("resident SiLU input binding absent".into()))
+        };
+        let input = input(0)?;
+        let output = node
+            .outputs
+            .first()
+            .and_then(|id| self.values.get(id.0 as usize))
+            .ok_or_else(|| Error::State("resident SiLU output binding absent".into()))?;
+        let multiplier = if *multiplied {
+            Some(
+                node.inputs
+                    .get(1)
+                    .and_then(|id| self.values.get(id.0 as usize))
+                    .ok_or_else(|| Error::State("resident SiLU multiplier absent".into()))?,
+            )
+        } else {
+            None
+        };
+        if output.elements != input.elements
+            || multiplier.is_some_and(|value| value.elements != input.elements)
+        {
+            return Err(Error::State("resident SiLU value geometry changed".into()));
+        }
+        let Some(width) = ordered_width(rows, input.elements)? else {
+            // Only the immutable closed geometry selects the original route.
+            // Hardware/range/alias errors from prepare always propagate.
+            return Ok(None);
+        };
+        crate::ordered_silu::PreparedOrderedSilu::prepare(
+            client,
+            rows,
+            width,
+            crate::ordered_silu::Bindings {
+                input: &input.handle,
+                multiplier: multiplier.map(|value| &value.handle),
+                output: &output.handle,
+                trace: None,
+            },
+        )
+        .map(Some)
+    }
+
+    /// Full immutable plan, including StateOnly-skipped tail nodes. Successful
+    /// prepare uses only stack bindings and original Arc handle clones.
+    #[cfg(feature = "cuda")]
+    fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
+        Self::require_legacy_contract(&self.plan)?;
+        for node in self.plan.nodes() {
+            if matches!(node.operation, PlanOperation::Silu { .. }) {
+                self.ordered_silu_binding(client, node, rows)?;
+            }
+        }
+        Ok(())
+    }
+
+    fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
+        let PlanOperation::Silu { multiplied } = &node.operation else {
+            return Err(Error::State("resident SiLU operation changed".into()));
+        };
+        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
+        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
+        let s = |i: usize| &self.scratch[i];
+        let n = input(0).elements;
+        // Repeat the exact cold binding before this node's first launch. No
+        // mutable origin/range/rows, persistent table or Err fallback exists.
+        #[cfg(feature = "cuda")]
+        let ordered = self.ordered_silu_binding(client, node, rows)?;
+        #[cfg(not(feature = "cuda"))]
+        let _ = rows;
+        // SAFETY: original typed plan/ranges and owning submission retain the
+        // same original input/output/status/scratch leases as before factoring.
+        unsafe {
+            crate::resident_bindings::certify_interval::launch_unchecked(
+                client,
+                grid(n),
+                threads(),
+                arg(&input(0).handle, n),
+                arg(&self.status, 1),
+                -256.0,
+                256.0,
+                n,
+            );
+        }
+        #[cfg(feature = "cuda")]
+        if let Some(ordered) = ordered {
+            return Ok(1 + ordered.launch(client)?);
+        }
+        // Original CUDA-outside-geometry and Vulkan body, byte-equivalent
+        // modulo indentation to the pre-integration PlanOperation::Silu arm.
+        unsafe {
+            for stage in 0..12 {
+                crate::silu::silu_step::launch_unchecked(
+                    client,
+                    grid(n),
+                    threads(),
+                    arg(&input(0).handle, n),
+                    arg(s(0), n * crate::silu::SCRATCH),
+                    arg(&output(0).handle, n),
+                    n,
+                    stage,
+                );
+            }
+            if *multiplied {
+                crate::silu::silu_gate::launch_unchecked(
+                    client,
+                    grid(n),
+                    threads(),
+                    arg(&input(1).handle, n),
+                    arg(&output(0).handle, n),
+                    n,
+                );
+            }
+        }
+        Ok(13 + usize::from(*multiplied))
+    }
+}
+
+#[cfg(all(test, feature = "cuda"))]
+mod tests;
diff --git a/crates/zerogravity-gpu/src/resident_program/silu/tests.rs b/crates/zerogravity-gpu/src/resident_program/silu/tests.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/silu/tests.rs
@@ -0,0 +1,27 @@
+use super::*;
+
+#[test]
+fn packed_domains_keep_actual_rows_instead_of_flattening_to_b1() -> Result<()> {
+    for rows in [2, 4, 8] {
+        for width in [1, 31, 32, 33, 65_536] {
+            assert_eq!(ordered_width(rows, rows * width)?, Some(width));
+        }
+    }
+    // This packed B8 shape exceeds the B1 width limit, but every genuine row
+    // remains in the accepted domain. No padding/request is introduced.
+    assert_eq!(ordered_width(8, 8 * 65_536)?, Some(65_536));
+    assert_eq!(ordered_width(1, 8 * 65_536)?, None);
+    Ok(())
+}
+
+#[test]
+fn only_known_geometry_uses_original_route_and_partition_changes_refuse() -> Result<()> {
+    for rows in [1, 2, 4, 8] {
+        assert_eq!(ordered_width(rows, 0)?, None);
+        assert_eq!(ordered_width(rows, rows * 65_537)?, None);
+    }
+    for (rows, words) in [(0, 32), (3, 96), (16, 512), (2, 65), (4, 129), (8, 257)] {
+        assert!(ordered_width(rows, words).is_err());
+    }
+    Ok(())
+}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/ORDERED_GEOMETRY_SEAM.patch

Original UTF-8 bytes: 1318; SHA256 `775bbe0bb8fbd7022fbd3bca64d85dfa7af8e9cc332d37518bc91618a2d74592`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/ordered_silu.rs b/crates/zerogravity-gpu/src/ordered_silu.rs
--- a/crates/zerogravity-gpu/src/ordered_silu.rs
+++ b/crates/zerogravity-gpu/src/ordered_silu.rs
@@ -1,5 +1,5 @@
-//! Dormant CUDA-only ordered version of the original12-launch source SiLU.
-//! No exp intrinsic, new polynomial, source profile or production selection.
+//! CUDA-only ordered version of the original12-launch source SiLU.
+//! No exp intrinsic, new polynomial or numerical profile.
 use crate::source_accumulation::{SourceAccumulation, ensure_capability};
 use crate::source_float::{correct_quotient, round_source, widen_source};
 use cubecl::{client::Client, prelude::*, server::Handle};
@@ -11,8 +11,13 @@
 const LANES: u32 = 32;
 pub(crate) const TRACE_WORDS: usize = 17;
 
+/// Exact closed support; callers select the original route outside this geometry.
+pub(crate) fn supports_geometry(rows: usize, width: usize) -> bool {
+    matches!(rows, 1 | 2 | 4 | 8) && width > 0 && width <= MAX_ROW_WORDS
+}
+
 fn geometry(rows: usize, width: usize) -> Result<usize> {
-    if !matches!(rows, 1 | 2 | 4 | 8) || width == 0 || width > MAX_ROW_WORDS {
+    if !supports_geometry(rows, width) {
         return Err(Error::Unsupported(
             "ordered SiLU native row geometry".into(),
         ));

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/ROOT_WIRING.patch

Original UTF-8 bytes: 346; SHA256 `b4c44ebf38575efefa7896cc6f7dc56d2620df90ef4e04e39426273389d69658`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/lib.rs b/crates/zerogravity-gpu/src/lib.rs
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -92,5 +92,5 @@
 };
 pub use resident_provider::CubeResidentProvider;
 
-#[cfg(all(test, feature = "qualification", feature = "cuda"))]
+#[cfg(feature = "cuda")]
 mod ordered_silu;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/VISIBILITY_ONLY.patch

Original UTF-8 bytes: 1053; SHA256 `01ecb17b43fe65c68125ec647f8d052b022178cdc146eb418fd7e6adf708372f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/silu.rs
+++ b/crates/zerogravity-gpu/src/resident_program/silu.rs
@@ -81,7 +81,7 @@
     /// Full immutable plan, including StateOnly-skipped tail nodes. Successful
     /// prepare uses only stack bindings and original Arc handle clones.
     #[cfg(feature = "cuda")]
-    fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
+    pub(super) fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
         Self::require_legacy_contract(&self.plan)?;
         for node in self.plan.nodes() {
             if matches!(node.operation, PlanOperation::Silu { .. }) {
@@ -91,7 +91,7 @@
         Ok(())
     }
 
-    fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
+    pub(super) fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
         let PlanOperation::Silu { multiplied } = &node.operation else {
             return Err(Error::State("resident SiLU operation changed".into()));
         };

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/frozen/VISIBILITY_ONLY.patch

Original UTF-8 bytes: 1053; SHA256 `01ecb17b43fe65c68125ec647f8d052b022178cdc146eb418fd7e6adf708372f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/silu.rs
+++ b/crates/zerogravity-gpu/src/resident_program/silu.rs
@@ -81,7 +81,7 @@
     /// Full immutable plan, including StateOnly-skipped tail nodes. Successful
     /// prepare uses only stack bindings and original Arc handle clones.
     #[cfg(feature = "cuda")]
-    fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
+    pub(super) fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
         Self::require_legacy_contract(&self.plan)?;
         for node in self.plan.nodes() {
             if matches!(node.operation, PlanOperation::Silu { .. }) {
@@ -91,7 +91,7 @@
         Ok(())
     }
 
-    fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
+    pub(super) fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
         let PlanOperation::Silu { multiplied } = &node.operation else {
             return Err(Error::State("resident SiLU operation changed".into()));
         };

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 12278; SHA256 `b8a2f59f32e3d86b6784c7dc1bda0889143d21f922f37df2c7c751b2fdf623f5`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/resident_program.rs b/crates/zerogravity-gpu/src/resident_program.rs
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -1009,6 +1009,8 @@
                 _ => Ok(None),
             })
             .collect::<Result<Vec<_>>>()?;
+        #[cfg(feature = "cuda")]
+        program.preflight_silu(client, 1)?;
         for handle in program.owned_handles() {
             program.owned_ranges.insert(BufferRange::new(&handle)?)?;
         }
@@ -1106,6 +1108,17 @@
         handles
     }
     pub fn validate_runtime(
+        &self,
+        client: &Client,
+        p: RuntimeParameters,
+        bindings: &[StateBinding],
+    ) -> Result<()> {
+        #[cfg(feature = "cuda")]
+        self.preflight_silu(client, 1)?;
+        self.validate_state_runtime(client, p, bindings)
+    }
+
+    fn validate_state_runtime(
         &self,
         client: &Client,
         p: RuntimeParameters,
@@ -1616,42 +1629,7 @@
                     );
                     Ok(4)
                 }
-                PlanOperation::Silu { multiplied } => {
-                    let n = input(0).elements;
-                    crate::resident_bindings::certify_interval::launch_unchecked(
-                        client,
-                        grid(n),
-                        threads(),
-                        arg(&input(0).handle, n),
-                        arg(&self.status, 1),
-                        -256.0,
-                        256.0,
-                        n,
-                    );
-                    for stage in 0..12 {
-                        crate::silu::silu_step::launch_unchecked(
-                            client,
-                            grid(n),
-                            threads(),
-                            arg(&input(0).handle, n),
-                            arg(s(0), n * crate::silu::SCRATCH),
-                            arg(&output(0).handle, n),
-                            n,
-                            stage,
-                        );
-                    }
-                    if *multiplied {
-                        crate::silu::silu_gate::launch_unchecked(
-                            client,
-                            grid(n),
-                            threads(),
-                            arg(&input(1).handle, n),
-                            arg(&output(0).handle, n),
-                            n,
-                        );
-                    }
-                    Ok(13 + usize::from(*multiplied))
-                }
+                PlanOperation::Silu { .. } => self.launch_silu(client, node, 1),
                 PlanOperation::Add => {
                     let n = input(0).elements;
                     crate::elementwise::residual_add::launch_unchecked(
@@ -2684,3 +2662,5 @@
 #[cfg(test)]
 #[path = "resident_program/typed_refusal_tests.rs"]
 mod typed_refusal_tests;
+
+mod silu;
diff --git a/crates/zerogravity-gpu/src/resident_program/batch.rs b/crates/zerogravity-gpu/src/resident_program/batch.rs
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -323,6 +323,8 @@
                 .map(Some)
             })
             .collect::<Result<Vec<_>>>()?;
+        #[cfg(feature = "cuda")]
+        inner.preflight_silu(client, rows)?;
         Ok(Self {
             inner,
             rows,
@@ -388,10 +390,13 @@
         if parameters.len() != self.rows || bindings.len() != self.rows {
             return Err(Error::Malformed("resident batch row count changed".into()));
         }
+        #[cfg(feature = "cuda")]
+        self.inner.preflight_silu(client, self.rows)?;
         let mut priors = Vec::new();
         let mut candidates = Vec::new();
         for (parameter, binding) in parameters.iter().zip(bindings) {
-            self.inner.validate_runtime(client, *parameter, binding)?;
+            self.inner
+                .validate_state_runtime(client, *parameter, binding)?;
             for node in self.inner.plan.nodes() {
                 if let PlanOperation::Embed { tensor, .. } = &node.operation
                     && parameter.token as usize >= tensor.rows
@@ -1299,6 +1304,7 @@
                 }
                 // Original source kernels, accumulation order, physical rounding
                 // boundaries and attention padding remain the authority.
+                PlanOperation::Silu { .. } => self.inner.launch_silu(client, node, self.rows),
                 _ => self.inner.launch_node(client, index, node, p, &self.packed),
             }
         }
diff --git a/crates/zerogravity-gpu/src/resident_program/silu.rs b/crates/zerogravity-gpu/src/resident_program/silu.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/silu.rs
@@ -0,0 +1,157 @@
+//! Same original SiLU buffers/certificates; no persistent binding metadata.
+use super::*;
+
+#[cfg(feature = "cuda")]
+fn ordered_width(rows: usize, elements: usize) -> Result<Option<usize>> {
+    if !matches!(rows, 1 | 2 | 4 | 8) || !elements.is_multiple_of(rows) {
+        return Err(Error::State("resident SiLU row partition changed".into()));
+    }
+    let width = elements / rows;
+    Ok(crate::ordered_silu::supports_geometry(rows, width).then_some(width))
+}
+
+impl ResidentProgram {
+    #[cfg(feature = "cuda")]
+    fn ordered_silu_binding(
+        &self,
+        client: &Client,
+        node: &PlanNode,
+        rows: usize,
+    ) -> Result<Option<crate::ordered_silu::PreparedOrderedSilu>> {
+        if !crate::activation_quant_fused::is_cuda_service(client) {
+            return Ok(None);
+        }
+        if self.plan.numerical_profile()
+            != zerogravity_core::accelerator::program::ProgramNumericalProfile::GgufSourceV1
+            || self.plan.is_operator_slice()
+        {
+            return Err(Error::Unsupported(
+                "resident SiLU numerical contract".into(),
+            ));
+        }
+        let PlanOperation::Silu { multiplied } = &node.operation else {
+            return Err(Error::State("resident SiLU operation changed".into()));
+        };
+        let input = |index: usize| -> Result<&DeviceBuffer> {
+            node.inputs
+                .get(index)
+                .and_then(|id| self.values.get(id.0 as usize))
+                .ok_or_else(|| Error::State("resident SiLU input binding absent".into()))
+        };
+        let input = input(0)?;
+        let output = node
+            .outputs
+            .first()
+            .and_then(|id| self.values.get(id.0 as usize))
+            .ok_or_else(|| Error::State("resident SiLU output binding absent".into()))?;
+        let multiplier = if *multiplied {
+            Some(
+                node.inputs
+                    .get(1)
+                    .and_then(|id| self.values.get(id.0 as usize))
+                    .ok_or_else(|| Error::State("resident SiLU multiplier absent".into()))?,
+            )
+        } else {
+            None
+        };
+        if output.elements != input.elements
+            || multiplier.is_some_and(|value| value.elements != input.elements)
+        {
+            return Err(Error::State("resident SiLU value geometry changed".into()));
+        }
+        let Some(width) = ordered_width(rows, input.elements)? else {
+            // Only the immutable closed geometry selects the original route.
+            // Hardware/range/alias errors from prepare always propagate.
+            return Ok(None);
+        };
+        crate::ordered_silu::PreparedOrderedSilu::prepare(
+            client,
+            rows,
+            width,
+            crate::ordered_silu::Bindings {
+                input: &input.handle,
+                multiplier: multiplier.map(|value| &value.handle),
+                output: &output.handle,
+                trace: None,
+            },
+        )
+        .map(Some)
+    }
+
+    /// Full immutable plan, including StateOnly-skipped tail nodes. Successful
+    /// prepare uses only stack bindings and original Arc handle clones.
+    #[cfg(feature = "cuda")]
+    fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
+        Self::require_legacy_contract(&self.plan)?;
+        for node in self.plan.nodes() {
+            if matches!(node.operation, PlanOperation::Silu { .. }) {
+                self.ordered_silu_binding(client, node, rows)?;
+            }
+        }
+        Ok(())
+    }
+
+    fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
+        let PlanOperation::Silu { multiplied } = &node.operation else {
+            return Err(Error::State("resident SiLU operation changed".into()));
+        };
+        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
+        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
+        let s = |i: usize| &self.scratch[i];
+        let n = input(0).elements;
+        // Repeat the exact cold binding before this node's first launch. No
+        // mutable origin/range/rows, persistent table or Err fallback exists.
+        #[cfg(feature = "cuda")]
+        let ordered = self.ordered_silu_binding(client, node, rows)?;
+        #[cfg(not(feature = "cuda"))]
+        let _ = rows;
+        // SAFETY: original typed plan/ranges and owning submission retain the
+        // same original input/output/status/scratch leases as before factoring.
+        unsafe {
+            crate::resident_bindings::certify_interval::launch_unchecked(
+                client,
+                grid(n),
+                threads(),
+                arg(&input(0).handle, n),
+                arg(&self.status, 1),
+                -256.0,
+                256.0,
+                n,
+            );
+        }
+        #[cfg(feature = "cuda")]
+        if let Some(ordered) = ordered {
+            return Ok(1 + ordered.launch(client)?);
+        }
+        // Original CUDA-outside-geometry and Vulkan body, byte-equivalent
+        // modulo indentation to the pre-integration PlanOperation::Silu arm.
+        unsafe {
+            for stage in 0..12 {
+                crate::silu::silu_step::launch_unchecked(
+                    client,
+                    grid(n),
+                    threads(),
+                    arg(&input(0).handle, n),
+                    arg(s(0), n * crate::silu::SCRATCH),
+                    arg(&output(0).handle, n),
+                    n,
+                    stage,
+                );
+            }
+            if *multiplied {
+                crate::silu::silu_gate::launch_unchecked(
+                    client,
+                    grid(n),
+                    threads(),
+                    arg(&input(1).handle, n),
+                    arg(&output(0).handle, n),
+                    n,
+                );
+            }
+        }
+        Ok(13 + usize::from(*multiplied))
+    }
+}
+
+#[cfg(all(test, feature = "cuda"))]
+mod tests;
diff --git a/crates/zerogravity-gpu/src/resident_program/silu/tests.rs b/crates/zerogravity-gpu/src/resident_program/silu/tests.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/silu/tests.rs
@@ -0,0 +1,27 @@
+use super::*;
+
+#[test]
+fn packed_domains_keep_actual_rows_instead_of_flattening_to_b1() -> Result<()> {
+    for rows in [2, 4, 8] {
+        for width in [1, 31, 32, 33, 65_536] {
+            assert_eq!(ordered_width(rows, rows * width)?, Some(width));
+        }
+    }
+    // This packed B8 shape exceeds the B1 width limit, but every genuine row
+    // remains in the accepted domain. No padding/request is introduced.
+    assert_eq!(ordered_width(8, 8 * 65_536)?, Some(65_536));
+    assert_eq!(ordered_width(1, 8 * 65_536)?, None);
+    Ok(())
+}
+
+#[test]
+fn only_known_geometry_uses_original_route_and_partition_changes_refuse() -> Result<()> {
+    for rows in [1, 2, 4, 8] {
+        assert_eq!(ordered_width(rows, 0)?, None);
+        assert_eq!(ordered_width(rows, rows * 65_537)?, None);
+    }
+    for (rows, words) in [(0, 32), (3, 96), (16, 512), (2, 65), (4, 129), (8, 257)] {
+        assert!(ordered_width(rows, words).is_err());
+    }
+    Ok(())
+}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/frozen/ORDERED_GEOMETRY_SEAM.patch

Original UTF-8 bytes: 1318; SHA256 `775bbe0bb8fbd7022fbd3bca64d85dfa7af8e9cc332d37518bc91618a2d74592`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/ordered_silu.rs b/crates/zerogravity-gpu/src/ordered_silu.rs
--- a/crates/zerogravity-gpu/src/ordered_silu.rs
+++ b/crates/zerogravity-gpu/src/ordered_silu.rs
@@ -1,5 +1,5 @@
-//! Dormant CUDA-only ordered version of the original12-launch source SiLU.
-//! No exp intrinsic, new polynomial, source profile or production selection.
+//! CUDA-only ordered version of the original12-launch source SiLU.
+//! No exp intrinsic, new polynomial or numerical profile.
 use crate::source_accumulation::{SourceAccumulation, ensure_capability};
 use crate::source_float::{correct_quotient, round_source, widen_source};
 use cubecl::{client::Client, prelude::*, server::Handle};
@@ -11,8 +11,13 @@
 const LANES: u32 = 32;
 pub(crate) const TRACE_WORDS: usize = 17;
 
+/// Exact closed support; callers select the original route outside this geometry.
+pub(crate) fn supports_geometry(rows: usize, width: usize) -> bool {
+    matches!(rows, 1 | 2 | 4 | 8) && width > 0 && width <= MAX_ROW_WORDS
+}
+
 fn geometry(rows: usize, width: usize) -> Result<usize> {
-    if !matches!(rows, 1 | 2 | 4 | 8) || width == 0 || width > MAX_ROW_WORDS {
+    if !supports_geometry(rows, width) {
         return Err(Error::Unsupported(
             "ordered SiLU native row geometry".into(),
         ));

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/frozen/ROOT_WIRING.patch

Original UTF-8 bytes: 346; SHA256 `b4c44ebf38575efefa7896cc6f7dc56d2620df90ef4e04e39426273389d69658`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/lib.rs b/crates/zerogravity-gpu/src/lib.rs
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -92,5 +92,5 @@
 };
 pub use resident_provider::CubeResidentProvider;
 
-#[cfg(all(test, feature = "qualification", feature = "cuda"))]
+#[cfg(feature = "cuda")]
 mod ordered_silu;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/SCALAR_INPUT_ONLY.patch

Original UTF-8 bytes: 3313; SHA256 `0c4c5c4e5411ad65fe55efb5eba1e5ef64d873d254cab03670f380c3e921fdad`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/ordered_silu.rs b/crates/zerogravity-gpu/src/ordered_silu.rs
--- a/crates/zerogravity-gpu/src/ordered_silu.rs
+++ b/crates/zerogravity-gpu/src/ordered_silu.rs
@@ -186,56 +186,74 @@
 // Explicit PTX .rn instructions, no .ftz. gpu_asm without `pure` lowers to
 // volatile asm; f32 output constraints retain each rounded source result even
 // when immediately widened by the next stage. No float-cast-only assumption.
+// gpu_asm scalar inputs must be SSA values: its generic HasValue path also
+// accepts pointers. Bit-only reinterpretation forces the frontend read_value
+// load for prior mutable asm outputs without changing any floating-point bits.
 #[cube]
 fn rn32(value: f64) -> f32 {
+    let value = f64::reinterpret(u64::reinterpret(value));
     let output: f32;
     gpu_asm!("cvt.rn.f32.f64 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
     output
 }
 #[cube]
 fn wide(value: f32) -> f64 {
+    let value = f32::reinterpret(u32::reinterpret(value));
     let output: f64;
     gpu_asm!("cvt.f64.f32 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
     output
 }
 #[cube]
 fn add64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("add.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn sub64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("sub.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn mul64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("mul.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn div64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("div.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn fma64(a: f64, b: f64, c: f64) -> f64 {
+    let a = f64::reinterpret(u64::reinterpret(a));
+    let b = f64::reinterpret(u64::reinterpret(b));
+    let c = f64::reinterpret(u64::reinterpret(c));
     let output: f64;
     gpu_asm!("fma.rn.f64 {out}, {a}, {b}, {c};",out=out(_) output,a=in(_) a,b=in(_) b,c=in(_) c,options(nomem));
     output
 }
 #[cube]
 fn negative_input(input: f32) -> f32 {
+    let input = f32::reinterpret(u32::reinterpret(input));
     let output: f32;
     gpu_asm!("sub.rn.f32 {out}, 0f00000000, {value};",out=out(_) output,value=in(_) input,options(nomem));
     output
 }
 #[cube]
 fn stored_bits(input: f32) -> u32 {
+    let input = f32::reinterpret(u32::reinterpret(input));
     let output: u32;
     gpu_asm!("mov.b32 {out}, {value};",out=out(_) output,value=in(_) input,options(nomem));
     output

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/frozen/SCALAR_INPUT_ONLY.patch

Original UTF-8 bytes: 3313; SHA256 `0c4c5c4e5411ad65fe55efb5eba1e5ef64d873d254cab03670f380c3e921fdad`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/ordered_silu.rs b/crates/zerogravity-gpu/src/ordered_silu.rs
--- a/crates/zerogravity-gpu/src/ordered_silu.rs
+++ b/crates/zerogravity-gpu/src/ordered_silu.rs
@@ -186,56 +186,74 @@
 // Explicit PTX .rn instructions, no .ftz. gpu_asm without `pure` lowers to
 // volatile asm; f32 output constraints retain each rounded source result even
 // when immediately widened by the next stage. No float-cast-only assumption.
+// gpu_asm scalar inputs must be SSA values: its generic HasValue path also
+// accepts pointers. Bit-only reinterpretation forces the frontend read_value
+// load for prior mutable asm outputs without changing any floating-point bits.
 #[cube]
 fn rn32(value: f64) -> f32 {
+    let value = f64::reinterpret(u64::reinterpret(value));
     let output: f32;
     gpu_asm!("cvt.rn.f32.f64 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
     output
 }
 #[cube]
 fn wide(value: f32) -> f64 {
+    let value = f32::reinterpret(u32::reinterpret(value));
     let output: f64;
     gpu_asm!("cvt.f64.f32 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
     output
 }
 #[cube]
 fn add64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("add.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn sub64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("sub.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn mul64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("mul.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn div64(left: f64, right: f64) -> f64 {
+    let left = f64::reinterpret(u64::reinterpret(left));
+    let right = f64::reinterpret(u64::reinterpret(right));
     let output: f64;
     gpu_asm!("div.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
     output
 }
 #[cube]
 fn fma64(a: f64, b: f64, c: f64) -> f64 {
+    let a = f64::reinterpret(u64::reinterpret(a));
+    let b = f64::reinterpret(u64::reinterpret(b));
+    let c = f64::reinterpret(u64::reinterpret(c));
     let output: f64;
     gpu_asm!("fma.rn.f64 {out}, {a}, {b}, {c};",out=out(_) output,a=in(_) a,b=in(_) b,c=in(_) c,options(nomem));
     output
 }
 #[cube]
 fn negative_input(input: f32) -> f32 {
+    let input = f32::reinterpret(u32::reinterpret(input));
     let output: f32;
     gpu_asm!("sub.rn.f32 {out}, 0f00000000, {value};",out=out(_) output,value=in(_) input,options(nomem));
     output
 }
 #[cube]
 fn stored_bits(input: f32) -> u32 {
+    let input = f32::reinterpret(u32::reinterpret(input));
     let output: u32;
     gpu_asm!("mov.b32 {out}, {value};",out=out(_) output,value=in(_) input,options(nomem));
     output

````

## Source: .agents/runs/serving-and-speculation-parity-root-2026-10-19/b6-worktree-base.patch

Original UTF-8 bytes: 71125; SHA256 `bef0f1ee5b0d978e93537b8a4ba6904110c1be3a2f17d0cf5e5892f9b92c757d`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-cli/src/serve/gpu_batch.rs b/crates/zerogravity-cli/src/serve/gpu_batch.rs
index fb0bacf..fe96efd 100644
--- a/crates/zerogravity-cli/src/serve/gpu_batch.rs
+++ b/crates/zerogravity-cli/src/serve/gpu_batch.rs
@@ -1,7 +1,7 @@
 //! Finite, deterministic source-aware GPU request wave formation.
 //!
 //! A wave is scheduler membership, not physical completion or memory credit.
-//! The GPU worker separately selects qualified native B2 or original B1
+//! The GPU worker separately selects qualified native bands or original B1
 //! execution and reports actual Ready/dispatch/root receipts.
 
 use super::engine::ServiceError;
@@ -30,6 +30,31 @@ pub(super) struct Limits {
     pub max_tokens: usize,
     pub max_scratch: usize,
 }
+
+/// Contiguous, unpadded native bands; B1 carries an unmatched survivor.
+/// Ceiling 1 is the final prelaunch capacity fallback.
+pub(super) fn native_bands(count: usize, ceiling: usize) -> Result<Vec<usize>, ServiceError> {
+    if count > 64 || !matches!(ceiling, 1 | 2 | 4 | 8) {
+        return Err(ServiceError::new(
+            400,
+            "invalid_native_bands",
+            "native bands require at most 64 rows and ceiling 1/2/4/8",
+        ));
+    }
+    let mut remaining = count;
+    let mut bands = Vec::new();
+    for width in [8, 4, 2, 1] {
+        if width > ceiling {
+            continue;
+        }
+        while remaining >= width {
+            bands.push(width);
+            remaining -= width;
+        }
+    }
+    Ok(bands)
+}
+
 /// Every caller's ready work is included exactly once, using first-seen
 /// source/phase/frontier grouping and finite quota-split waves. No sorting
 /// across requests or silent dropping of a queue tail.
@@ -107,6 +132,53 @@ pub(super) fn form(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, Service
 #[cfg(test)]
 mod tests {
     use super::*;
+
+    #[test]
+    fn native_bands_cover_every_row_in_order_without_padding() {
+        for ceiling in [1, 2, 4, 8] {
+            for count in 0..=64 {
+                let bands = native_bands(count, ceiling).unwrap();
+                assert_eq!(bands.iter().sum::<usize>(), count);
+                assert!(bands.windows(2).all(|pair| pair[0] >= pair[1]));
+                let mut frontier = 0;
+                let mut rows = Vec::new();
+                for width in bands {
+                    assert!(matches!(width, 1 | 2 | 4 | 8));
+                    assert!(width <= ceiling);
+                    assert!(frontier + width <= count);
+                    for larger in [2, 4, 8] {
+                        assert!(larger <= width || larger > ceiling || larger > count - frontier);
+                    }
+                    rows.extend(frontier..frontier + width);
+                    frontier += width;
+                }
+                assert_eq!(rows, (0..count).collect::<Vec<_>>());
+            }
+        }
+    }
+
+    #[test]
+    fn native_bands_repartition_survivors_and_lower_capacity_ceilings() {
+        assert_eq!(native_bands(8, 8).unwrap(), [8]);
+        assert_eq!(native_bands(7, 8).unwrap(), [4, 2, 1]);
+        assert_eq!(native_bands(6, 8).unwrap(), [4, 2]);
+        assert_eq!(native_bands(5, 8).unwrap(), [4, 1]);
+        assert_eq!(native_bands(3, 4).unwrap(), [2, 1]);
+        assert_eq!(native_bands(8, 4).unwrap(), [4, 4]);
+        assert_eq!(native_bands(4, 2).unwrap(), [2, 2]);
+        assert_eq!(native_bands(2, 1).unwrap(), [1, 1]);
+    }
+
+    #[test]
+    fn native_bands_reject_unbounded_or_unsupported_shapes() {
+        for ceiling in [0, 3, 5, 6, 7, 9, usize::MAX] {
+            assert!(native_bands(8, ceiling).is_err());
+        }
+        for count in [65, usize::MAX] {
+            assert!(native_bands(count, 8).is_err());
+        }
+    }
+
     fn r(source: usize, phase: Phase, frontier: usize, tokens: usize) -> Ready {
         Ready {
             source,
diff --git a/crates/zerogravity-cli/src/serve/gpu_engine.rs b/crates/zerogravity-cli/src/serve/gpu_engine.rs
index 154eb6d..6f4b993 100644
--- a/crates/zerogravity-cli/src/serve/gpu_engine.rs
+++ b/crates/zerogravity-cli/src/serve/gpu_engine.rs
@@ -1,6 +1,6 @@
 //! Resident GPU OpenAI serving: one loaded model source, independent device
 //! StateRoots, one physical workspace and terminal completion per GPU step.
-//! Opt-in CUDA B2 waves use the original native ForwardPlan; other work
+//! Opt-in CUDA B2/B4/B8 waves use the original native ForwardPlan; other work
 //! interleaves through B1. Device feedback retains each SessionAuthority.
 
 use super::engine::{
@@ -42,7 +42,7 @@ pub(super) struct GpuProfile {
     pub output: usize,
     pub provider_mib: usize,
     pub device_mib: usize,
-    pub native_b2: bool,
+    pub native_rows: usize,
 }
 
 /// CubeCL CUDA's C++ backend requires a toolkit INCLUDE root in addition
@@ -109,7 +109,7 @@ struct GpuModel {
     tokenizer: Arc<QwenTokenizer>,
     group: usize,
     hidden_width: usize,
-    native_b2: bool,
+    native_rows: usize,
 }
 #[derive(Default)]
 struct WaveStats {
@@ -128,6 +128,8 @@ struct WaveStats {
     model_tokens: AtomicU64,
     committed_roots: AtomicU64,
     fused_batches: AtomicU64,
+    native_bands: [AtomicU64; 3],
+    native_admission_refusals: AtomicU64,
     fused_rows: AtomicU64,
     native_projections: AtomicU64,
     native_sampler: AtomicU64,
@@ -157,13 +159,15 @@ pub(super) struct GpuMetrics {
     pub model_tokens: u64,
     pub committed_roots: u64,
     pub fused_batches: u64,
+    pub native_bands: [u64; 3],
+    pub native_admission_refusals: u64,
     pub fused_rows: u64,
     pub native_projections: u64,
     pub native_sampler: u64,
     pub discarded_cohorts: u64,
     pub output_tokens: u64,
     pub parked: usize,
-    pub native_b2: bool,
+    pub native_rows: usize,
 }
 pub(super) struct GpuRouter {
     models: Arc<BTreeMap<String, Arc<GpuModel>>>,
@@ -171,7 +175,7 @@ pub(super) struct GpuRouter {
     next_id: AtomicU64,
     stats: Arc<WaveStats>,
     pub(super) max_output: usize,
-    native_b2: bool,
+    native_rows: usize,
 }
 impl GpuRouter {
     pub(super) fn validate_profile(
@@ -211,14 +215,18 @@ impl GpuRouter {
             output,
             provider_mib,
             device_mib,
-            native_b2,
+            native_rows,
         } = profile;
         Self::validate_profile(workers, active, context, output, provider_mib, device_mib)?;
-        if native_b2 && (backend != GpuBackend::Cuda || active < 2) {
+        if native_rows != 0
+            && (!matches!(native_rows, 2 | 4 | 8)
+                || backend != GpuBackend::Cuda
+                || active < native_rows)
+        {
             return Err(ServiceError::new(
                 400,
                 "unsupported_batch_mode",
-                "fused serving requires CUDA and at least two active slots",
+                "fused serving requires CUDA, rows=2/4/8 and active>=rows",
             ));
         }
         preflight_cuda_toolkit(backend)?;
@@ -283,7 +291,7 @@ impl GpuRouter {
                 tokenizer: Arc::new(tokenizer),
                 group: unique.len(),
                 hidden_width,
-                native_b2,
+                native_rows,
             });
             unique.insert(path, Arc::clone(&shared));
             models.insert(alias, shared);
@@ -295,7 +303,7 @@ impl GpuRouter {
         let worker_stats = Arc::clone(&stats);
         thread::Builder::new()
             .name("zg-resident-gpu".into())
-            .spawn(move || worker(receiver, sources, active, worker_stats, native_b2))
+            .spawn(move || worker(receiver, sources, active, worker_stats, native_rows))
             .map_err(|e| ServiceError::new(500, "worker_spawn", e.to_string()))?;
         Ok(Self {
             models,
@@ -303,7 +311,7 @@ impl GpuRouter {
             next_id: AtomicU64::new(1),
             stats,
             max_output: output,
-            native_b2,
+            native_rows,
         })
     }
     pub(super) fn metrics(&self) -> GpuMetrics {
@@ -323,13 +331,17 @@ impl GpuRouter {
             model_tokens: self.stats.model_tokens.load(Ordering::Relaxed),
             committed_roots: self.stats.committed_roots.load(Ordering::Relaxed),
             fused_batches: self.stats.fused_batches.load(Ordering::Relaxed),
+            native_bands: std::array::from_fn(|i| {
+                self.stats.native_bands[i].load(Ordering::Relaxed)
+            }),
+            native_admission_refusals: self.stats.native_admission_refusals.load(Ordering::Relaxed),
             fused_rows: self.stats.fused_rows.load(Ordering::Relaxed),
             native_projections: self.stats.native_projections.load(Ordering::Relaxed),
             native_sampler: self.stats.native_sampler.load(Ordering::Relaxed),
             discarded_cohorts: self.stats.discarded_cohorts.load(Ordering::Relaxed),
             output_tokens: self.stats.output_tokens.load(Ordering::Relaxed),
             parked: self.stats.parked.load(Ordering::Relaxed),
-            native_b2: self.native_b2,
+            native_rows: self.native_rows,
         }
     }
     pub(super) fn model_ids(&self) -> impl Iterator<Item = &str> {
@@ -398,7 +410,7 @@ struct GpuActive {
     prompt: Vec<u32>,
     cursor: usize,
     device_token: Option<ResidentToken>,
-    native_b2: bool,
+    native_rows: usize,
     packet_presented: bool,
     blocked_since: Option<Instant>,
 }
@@ -448,7 +460,7 @@ impl GpuActive {
             prompt,
             cursor: 0,
             device_token: None,
-            native_b2: model.native_b2,
+            native_rows: model.native_rows,
             packet_presented: false,
             blocked_since: None,
         })
@@ -475,7 +487,7 @@ impl GpuActive {
     }
     fn prefill_count(&self) -> usize {
         let remaining = self.prompt.len() - self.cursor;
-        if self.native_b2 && remaining > 1 {
+        if self.native_rows != 0 && remaining > 1 {
             (remaining - 1).min(PREFILL_QUANTUM)
         } else {
             remaining.min(PREFILL_QUANTUM)
@@ -658,10 +670,18 @@ fn worker(
     models: Arc<BTreeMap<String, Arc<GpuModel>>>,
     active_limit: usize,
     stats: Arc<WaveStats>,
-    native_b2: bool,
+    native_rows: usize,
 ) {
     let mut active = VecDeque::<GpuActive>::new();
     loop {
+        active.retain(|row| {
+            if row.cancelled.load(Ordering::Acquire) {
+                stats.cancelled.fetch_add(1, Ordering::Relaxed);
+                false
+            } else {
+                true
+            }
+        });
         while active.len() < active_limit {
             let next = if active.is_empty() {
                 match rx.recv_timeout(Duration::from_millis(25)) {
@@ -713,7 +733,11 @@ fn worker(
         let waves = match gpu_batch::form(
             &ready,
             Limits {
-                max_members: if native_b2 { 2 } else { active_limit },
+                max_members: if native_rows != 0 {
+                    native_rows
+                } else {
+                    active_limit
+                },
                 max_tokens: 64,
                 max_scratch: 256 * 1024 * 1024,
             },
@@ -731,7 +755,7 @@ fn worker(
         for wave in waves {
             // Validate source/epoch/unique committed session roots BEFORE
             // issuing even the first physical GPU member. P12 source
-            // batch admission is shared by native B2 and original B1 paths.
+            // batch admission is shared by native bands and original B1 paths.
             let admitted = wave
                 .indices
                 .iter()
@@ -776,7 +800,7 @@ fn worker(
                     .fetch_add(wave.indices.len() as u64, Ordering::Relaxed);
             }
             // Exact prompt tails share the existing checked root publisher.
-            // Native mode uses B2 kernels; interleaved mode keeps B1 graphs.
+            // Native mode uses multi-row kernels; interleaved mode keeps B1 graphs.
             let batch_ready = wave.indices.len() > 1
                 && wave.indices.iter().all(|&i| {
                     pending[i].as_ref().is_some_and(|row| {
@@ -791,11 +815,11 @@ fn worker(
                     .iter()
                     .map(|&i| pending[i].take().expect("unique admitted wave row"))
                     .collect::<Vec<_>>();
-                finish_prompt_cohort(members, &mut active, &stats, native_b2);
+                finish_prompt_cohort(members, &mut active, &stats, native_rows);
                 continue;
             }
-            if native_b2
-                && wave.indices.len() == 2
+            if native_rows != 0
+                && wave.indices.len() > 1
                 && wave.indices.iter().all(|&i| {
                     pending[i]
                         .as_ref()
@@ -807,7 +831,7 @@ fn worker(
                     .iter()
                     .map(|&i| pending[i].take().expect("unique admitted decode row"))
                     .collect();
-                advance_decode_cohort(members, &mut active, &stats);
+                advance_decode_cohort(members, &mut active, &stats, native_rows);
                 continue;
             }
             for index in wave.indices {
diff --git a/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs b/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
index 9f5d20c..0242734 100644
--- a/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
+++ b/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
@@ -5,37 +5,59 @@ pub(super) fn finish_prompt_cohort(
     members: Vec<GpuActive>,
     active: &mut VecDeque<GpuActive>,
     stats: &WaveStats,
-    native_b2: bool,
+    native_rows: usize,
 ) {
-    let members = members
-        .into_iter()
-        .filter_map(|row| eligible_member(row, active, stats))
-        .collect::<Vec<_>>();
-    if members.len() < 2 {
-        for row in members {
-            advance_member(row, active, stats);
-        }
-        return;
-    }
-    execute(members, active, stats, Phase::Prefill, native_b2);
+    regroup(members, active, stats, Phase::Prefill, native_rows);
 }
 
 pub(super) fn advance_decode_cohort(
     members: Vec<GpuActive>,
     active: &mut VecDeque<GpuActive>,
     stats: &WaveStats,
+    native_rows: usize,
+) {
+    regroup(members, active, stats, Phase::Decode, native_rows);
+}
+
+fn regroup(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    phase: Phase,
+    native_rows: usize,
 ) {
     // EOS, limits, cancellation and output credit can remove any wave row.
     // Repartition eligible survivors, never pad a native batch with a stop.
-    let mut eligible = members
+    let eligible = members
         .into_iter()
         .filter_map(|row| eligible_member(row, active, stats))
-        .collect::<VecDeque<_>>();
-    while let Some(first) = eligible.pop_front() {
-        if let Some(second) = eligible.pop_front() {
-            execute(vec![first, second], active, stats, Phase::Decode, true);
+        .collect::<Vec<_>>();
+    if native_rows == 0 {
+        if phase == Phase::Prefill && eligible.len() > 1 {
+            execute(eligible, active, stats, phase, 0);
         } else {
-            advance_member(first, active, stats);
+            for row in eligible {
+                advance_member(row, active, stats);
+            }
+        }
+        return;
+    }
+    let bands = match gpu_batch::native_bands(eligible.len(), native_rows) {
+        Ok(bands) => bands,
+        Err(error) => {
+            for row in eligible {
+                stats.fail(&row, error.clone());
+            }
+            return;
+        }
+    };
+    let mut remaining = eligible.into_iter();
+    for width in bands {
+        if width == 1 {
+            advance_member(remaining.next().expect("planned native row"), active, stats);
+        } else {
+            let members = remaining.by_ref().take(width).collect();
+            execute(members, active, stats, phase, width);
         }
     }
 }
@@ -45,8 +67,15 @@ fn execute(
     active: &mut VecDeque<GpuActive>,
     stats: &WaveStats,
     phase: Phase,
-    native_b2: bool,
+    native_rows: usize,
 ) {
+    if members
+        .iter()
+        .any(|row| row.cancelled.load(Ordering::Acquire))
+    {
+        regroup(members, active, stats, phase, native_rows);
+        return;
+    }
     let Some(first) = members.first() else { return };
     let model = Arc::clone(&first.resident);
     let roots = members
@@ -64,7 +93,8 @@ fn execute(
                         || !row.packet_presented
                         || row.device_token.is_none()
                         || row.blocked_since.is_some()))
-        }) || (native_b2 && members.len() != 2)
+        }) || (native_rows != 0
+            && (!matches!(native_rows, 2 | 4 | 8) || members.len() != native_rows))
         {
             return Err(zerogravity::Error::State(
                 "incompatible original GPU cohort".into(),
@@ -92,20 +122,29 @@ fn execute(
                 )
             })
             .collect::<Vec<_>>();
-        block_on(model.prepare_device_tokens_fused_cohort(&inputs))
+        block_on(model.prepare_device_tokens_fused_cohort_admitted(&inputs))
     } else {
         let inputs = members
             .iter()
             .map(|row| (&row.session, row.prompt[row.cursor]))
             .collect::<Vec<_>>();
-        if native_b2 {
-            block_on(model.prepare_greedy_fused_cohort(&inputs))
+        if native_rows != 0 {
+            block_on(model.prepare_greedy_fused_cohort_admitted(&inputs))
         } else {
-            block_on(model.prepare_greedy_cohort(&inputs))
+            block_on(model.prepare_greedy_cohort(&inputs)).map(Some)
         }
     };
     let cohort = match prepared {
-        Ok(value) => value,
+        Ok(Some(value)) => value,
+        Ok(None) => {
+            stats
+                .native_admission_refusals
+                .fetch_add(1, Ordering::Relaxed);
+            // The admitted API guarantees no launch or root/packet mutation.
+            // Each capacity retry strictly reduces the attempted native band.
+            regroup(members, active, stats, phase, native_rows / 2);
+            return;
+        }
         Err(error) => {
             let error = ServiceError::inference(error);
             for row in members {
@@ -116,7 +155,7 @@ fn execute(
     };
     let native = cohort.fused_dispatches() != 0 && cohort.native_greedy_dispatches() != 0;
     record_completion(stats, &cohort, native);
-    if native_b2 && !native {
+    if native_rows != 0 && !native {
         drop(cohort);
         stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
         for row in members {
@@ -142,13 +181,11 @@ fn execute(
         .iter()
         .any(|row| row.cancelled.load(Ordering::Acquire))
     {
-        // Known completion: discard all private results, preserve prior roots
-        // and packet-presented marks. Survivors may consume the same packet B1.
+        // Known completion: discard private results, preserve original roots
+        // and packet-presented marks, then regroup only eligible survivors.
         drop(cohort);
         stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
-        for row in members {
-            advance_member(row, active, stats);
-        }
+        regroup(members, active, stats, phase, native_rows);
         return;
     }
     let mut originals = members
@@ -183,17 +220,17 @@ fn execute(
     }
 }
 
-fn record_completion(stats: &WaveStats, cohort: &PreparedResidentCohort, native_b2: bool) {
+fn record_completion(stats: &WaveStats, cohort: &PreparedResidentCohort, native: bool) {
     let count = cohort.len() as u64;
     stats.completed.fetch_add(count, Ordering::Relaxed);
     stats
         .gpu_forwards
-        .fetch_add(if native_b2 { 1 } else { count }, Ordering::Relaxed);
+        .fetch_add(if native { 1 } else { count }, Ordering::Relaxed);
     stats
         .kernel_dispatches
         .fetch_add(cohort.dispatches() as u64, Ordering::Relaxed);
     stats.model_tokens.fetch_add(count, Ordering::Relaxed);
-    if native_b2 {
+    if native {
         stats.fused_batches.fetch_add(1, Ordering::Relaxed);
         stats.fused_rows.fetch_add(count, Ordering::Relaxed);
         stats
@@ -202,5 +239,13 @@ fn record_completion(stats: &WaveStats, cohort: &PreparedResidentCohort, native_
         stats
             .native_sampler
             .fetch_add(cohort.native_greedy_dispatches() as u64, Ordering::Relaxed);
+        if let Some(index) = match cohort.len() {
+            2 => Some(0),
+            4 => Some(1),
+            8 => Some(2),
+            _ => None,
+        } {
+            stats.native_bands[index].fetch_add(1, Ordering::Relaxed);
+        }
     }
 }
diff --git a/crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs b/crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
index 434f408..03172bd 100644
--- a/crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
+++ b/crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
@@ -56,6 +56,11 @@ struct Fixture {
 }
 impl Fixture {
     fn new(name: &str) -> Self {
+        Self::with_native_rows(name, 2)
+    }
+
+    fn with_native_rows(name: &str, native_rows: usize) -> Self {
+        assert!(matches!(native_rows, 2 | 4 | 8));
         if cfg!(debug_assertions) {
             panic!("physical serving qualification requires --release");
         }
@@ -99,7 +104,7 @@ impl Fixture {
             tokenizer,
             group: 0,
             hidden_width: source.geometry().hidden_size as usize,
-            native_b2: true,
+            native_rows,
         });
         Self {
             model,
@@ -301,11 +306,13 @@ fn round(active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
         active.extend(parked);
         return;
     }
+    let native_rows = eligible[0].native_rows;
+    assert!(eligible.iter().all(|lane| lane.native_rows == native_rows));
     let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
     let waves = gpu_batch::form(
         &ready,
         Limits {
-            max_members: 2,
+            max_members: native_rows,
             max_tokens: 64,
             max_scratch: 256 * 1024 * 1024,
         },
@@ -319,13 +326,13 @@ fn round(active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
             .map(|index| pending[index].take().unwrap())
             .collect::<Vec<_>>();
         if members.iter().all(|lane| lane.cursor == lane.prompt.len()) {
-            advance_decode_cohort(members, active, stats);
-        } else if members.len() == 2
+            advance_decode_cohort(members, active, stats, native_rows);
+        } else if members.len() > 1
             && members
                 .iter()
                 .all(|lane| lane.cursor + 1 == lane.prompt.len())
         {
-            finish_prompt_cohort(members, active, stats, true);
+            finish_prompt_cohort(members, active, stats, native_rows);
         } else {
             for lane in members {
                 advance_member(lane, active, stats);
@@ -691,6 +698,435 @@ fn qualify(name: &str) {
     );
 }
 
+struct Observed {
+    id: u64,
+    maximum: usize,
+    trace: Trace,
+}
+
+fn observed_lane(
+    fixture: &Fixture,
+    input: &str,
+    maximum: usize,
+    capacity: usize,
+    b1: bool,
+) -> (GpuActive, Observed) {
+    let (mut lane, trace) = fixture.lane(input, maximum, capacity);
+    if b1 {
+        b1_prompt(&mut lane, &fixture.stats);
+    }
+    let id = lane.session.binding().session().get();
+    (lane, Observed { id, maximum, trace })
+}
+
+fn reference_rows(fixture: &Fixture, active: &VecDeque<GpuActive>) -> BTreeMap<u64, Reference> {
+    let references = active
+        .iter()
+        .map(|lane| {
+            (
+                lane.session.binding().session().get(),
+                Reference::new(fixture, lane),
+            )
+        })
+        .collect::<BTreeMap<_, _>>();
+    assert_eq!(references.len(), active.len(), "independent session IDs");
+    references
+}
+
+fn finish_observed(
+    active: &mut VecDeque<GpuActive>,
+    fixture: &Fixture,
+    references: &mut BTreeMap<u64, Reference>,
+    observed: &mut [Observed],
+) {
+    finish(
+        active,
+        fixture,
+        references,
+        &mut observed
+            .iter_mut()
+            .map(|row| &mut row.trace)
+            .collect::<Vec<_>>(),
+    );
+    for row in observed {
+        row.trace
+            .verify(&references[&row.id], &fixture.model.tokenizer, row.maximum);
+    }
+}
+
+fn band_counts(stats: &WaveStats) -> [u64; 3] {
+    stats
+        .native_bands
+        .each_ref()
+        .map(|count| count.load(Ordering::Acquire))
+}
+
+fn require_bands(stats: &WaveStats, before: [u64; 3], expected: &[usize]) {
+    let actual = band_counts(stats);
+    for (index, width) in [2, 4, 8].into_iter().enumerate() {
+        assert_eq!(
+            actual[index] - before[index],
+            expected.iter().filter(|&&band| band == width).count() as u64,
+            "actual completed native B{width} cohorts"
+        );
+    }
+}
+
+fn prior_roots(active: &VecDeque<GpuActive>) -> BTreeMap<u64, (u64, u64)> {
+    active
+        .iter()
+        .map(|lane| {
+            let binding = lane.session.binding();
+            (
+                binding.session().get(),
+                (binding.root(), binding.frontier()),
+            )
+        })
+        .collect()
+}
+
+fn require_single_transition(active: &VecDeque<GpuActive>, prior: &BTreeMap<u64, (u64, u64)>) {
+    for lane in active {
+        let binding = lane.session.binding();
+        let (root, frontier) = prior[&binding.session().get()];
+        assert_eq!(binding.root(), root + 1, "one original root publication");
+        assert_eq!(binding.frontier(), frontier + 1, "one consumed token");
+    }
+}
+
+fn survivor_bands(width: usize) -> &'static [usize] {
+    match width {
+        4 => &[2, 1],
+        8 => &[4, 2, 1],
+        _ => panic!("wide fixture requires B4/B8"),
+    }
+}
+
+fn wide_prompt_lengths(fixture: &Fixture) {
+    let width = fixture.model.native_rows;
+    let mut active = VecDeque::new();
+    let mut observed = Vec::new();
+    for (index, input) in ["a", "b", "c", "d", "e", "f", "g", "h"]
+        .into_iter()
+        .take(width)
+        .enumerate()
+    {
+        let (lane, trace) = observed_lane(fixture, input, index + 1, EVENTS_PER_REQUEST, false);
+        assert_eq!(lane.prompt.len(), 1, "equal-frontier original token inputs");
+        active.push_back(lane);
+        observed.push(trace);
+    }
+    assert_eq!(
+        active
+            .iter()
+            .map(|lane| lane.prompt[0])
+            .collect::<std::collections::BTreeSet<_>>()
+            .len(),
+        width,
+        "distinct source input token IDs"
+    );
+    let prior = prior_roots(&active);
+    let bands = band_counts(&fixture.stats);
+    round(&mut active, &fixture.stats);
+    assert_eq!(active.len(), width);
+    require_bands(&fixture.stats, bands, &[width]);
+    require_single_transition(&active, &prior);
+    let mut references = reference_rows(fixture, &active);
+    assert!(
+        references
+            .values()
+            .all(|reference| { reference.sequence[0].0 != fixture.model.tokenizer.eos() })
+    );
+    let prior = prior_roots(&active);
+    let bands = band_counts(&fixture.stats);
+    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
+    let forwards = fixture.stats.gpu_forwards.load(Ordering::Acquire);
+    round(&mut active, &fixture.stats);
+    assert_eq!(
+        active.len(),
+        width - 1,
+        "maximum=1 row retires before feedback"
+    );
+    require_bands(&fixture.stats, bands, survivor_bands(width));
+    require_single_transition(&active, &prior);
+    assert_eq!(
+        fixture.stats.committed_roots.load(Ordering::Acquire),
+        committed + (width - 1) as u64
+    );
+    assert_eq!(
+        fixture.stats.gpu_forwards.load(Ordering::Acquire),
+        forwards + survivor_bands(width).len() as u64,
+        "native packet includes the exact B1 residual handoff"
+    );
+    update(&active, &mut references);
+    finish_observed(&mut active, fixture, &mut references, &mut observed);
+    println!(
+        "P12_WIDE_PROMPT_RETIRE_PASS rows={width} distinct_source_tokens_and_sessions=true \
+        maxima=1..{width} native_to_B1=true CPU_B1_packets_all42=exact text_usage_Done=once"
+    );
+}
+
+fn wide_b1_handoff(fixture: &Fixture) {
+    let width = fixture.model.native_rows;
+    let mut active = VecDeque::new();
+    let mut observed = Vec::new();
+    for _ in 0..width {
+        let (lane, trace) = observed_lane(fixture, "a", 3, EVENTS_PER_REQUEST, true);
+        active.push_back(lane);
+        observed.push(trace);
+    }
+    let mut references = reference_rows(fixture, &active);
+    let prior = prior_roots(&active);
+    let bands = band_counts(&fixture.stats);
+    round(&mut active, &fixture.stats);
+    assert_eq!(active.len(), width);
+    require_bands(&fixture.stats, bands, &[width]);
+    require_single_transition(&active, &prior);
+    update(&active, &mut references);
+    finish_observed(&mut active, fixture, &mut references, &mut observed);
+    println!("P12_WIDE_B1_NATIVE_HANDOFF_PASS rows={width} CPU_B1_packets_all42=exact");
+}
+
+fn wide_cancel_after_ready(fixture: &Fixture) {
+    let width = fixture.model.native_rows;
+    let mut active = VecDeque::new();
+    let mut observed = Vec::new();
+    for _ in 0..width {
+        let (lane, trace) = observed_lane(fixture, "a", 4, EVENTS_PER_REQUEST, true);
+        active.push_back(lane);
+        observed.push(trace);
+    }
+    let mut references = reference_rows(fixture, &active);
+    let cancelled_index = width / 2;
+    let mut cancelled = observed.remove(cancelled_index);
+    let cancelled_reference = references.remove(&cancelled.id).unwrap();
+    let saved = active[cancelled_index].session.snapshot();
+    let prior = prior_roots(&active);
+    let cancel = Arc::clone(&active[cancelled_index].cancelled);
+    AFTER_COHORT_READY.with(|hook| {
+        assert!(hook.borrow().is_none());
+        *hook.borrow_mut() = Some(Box::new(move || cancel.store(true, Ordering::Release)));
+    });
+    let bands = band_counts(&fixture.stats);
+    let discarded = fixture.stats.discarded_cohorts.load(Ordering::Acquire);
+    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
+    let model_tokens = fixture.stats.model_tokens.load(Ordering::Acquire);
+    let cancelled_before = fixture.stats.cancelled.load(Ordering::Acquire);
+    round(&mut active, &fixture.stats);
+    AFTER_COHORT_READY.with(|hook| assert!(hook.borrow().is_none(), "real Ready hook reached"));
+    assert_eq!(active.len(), width - 1);
+    assert!(
+        active
+            .iter()
+            .all(|lane| lane.session.binding().session().get() != cancelled.id)
+    );
+    let mut physical = vec![width];
+    physical.extend_from_slice(survivor_bands(width));
+    require_bands(&fixture.stats, bands, &physical);
+    require_single_transition(&active, &prior);
+    assert_eq!(
+        fixture.stats.discarded_cohorts.load(Ordering::Acquire),
+        discarded + 1
+    );
+    assert_eq!(
+        fixture.stats.cancelled.load(Ordering::Acquire),
+        cancelled_before + 1
+    );
+    assert_eq!(
+        fixture.stats.committed_roots.load(Ordering::Acquire),
+        committed + (width - 1) as u64
+    );
+    assert_eq!(
+        fixture.stats.model_tokens.load(Ordering::Acquire),
+        model_tokens + (2 * width - 1) as u64,
+        "whole discarded native cohort plus actual survivor forwards"
+    );
+    update(&active, &mut references);
+    cancelled_reference.snapshot(&saved);
+    cancelled.trace.drain();
+    assert!(cancelled.trace.done.is_empty());
+    let presented = fixture
+        .model
+        .tokenizer
+        .decode(&[cancelled_reference.sequence[0].0], true)
+        .unwrap();
+    let expected_text = if presented.contains('�') {
+        ""
+    } else {
+        &presented
+    };
+    assert_eq!(cancelled.trace.text, expected_text);
+    finish_observed(&mut active, fixture, &mut references, &mut observed);
+    cancelled_reference.snapshot(&saved);
+    cancelled.trace.drain();
+    assert!(cancelled.trace.done.is_empty());
+    assert_eq!(cancelled.trace.text, expected_text);
+    println!(
+        "P12_WIDE_READY_CANCEL_PASS rows={width} whole_candidate_discard=true \
+        survivors_regrouped=true original_roots_packets_all42=exact text_usage_Done=once"
+    );
+}
+
+fn wide_credit_park(fixture: &Fixture) {
+    let width = fixture.model.native_rows;
+    let mut active = VecDeque::new();
+    let mut observed = Vec::new();
+    for index in 0..width {
+        let (lane, trace) = observed_lane(
+            fixture,
+            if index == 0 { "a" } else { "b" },
+            8,
+            if index == 0 { 1 } else { EVENTS_PER_REQUEST },
+            true,
+        );
+        active.push_back(lane);
+        observed.push(trace);
+    }
+    let slow_id = observed[0].id;
+    let mut references = reference_rows(fixture, &active);
+    for _ in 0..12 {
+        round(&mut active, &fixture.stats);
+        update(&active, &mut references);
+        for row in observed.iter_mut().skip(1) {
+            row.trace.drain();
+        }
+        if active.iter().any(|lane| {
+            lane.session.binding().session().get() == slow_id && lane.blocked_since.is_some()
+        }) {
+            break;
+        }
+    }
+    assert_eq!(
+        active.len(),
+        width,
+        "healthy rows remain live at slow-reader park"
+    );
+    let slow = active
+        .iter()
+        .find(|lane| lane.session.binding().session().get() == slow_id)
+        .unwrap();
+    assert!(slow.blocked_since.is_some());
+    assert!(slow.packet_presented);
+    let binding = slow.session.binding();
+    let token = slow.device_token.as_ref().unwrap();
+    let packet = (token.id, token.max_bits);
+    for _ in 0..2 {
+        let prior = prior_roots(&active);
+        let bands = band_counts(&fixture.stats);
+        round(&mut active, &fixture.stats);
+        update(&active, &mut references);
+        require_bands(&fixture.stats, bands, survivor_bands(width));
+        let mut healthy = 0;
+        for lane in &active {
+            let current = lane.session.binding();
+            if current.session().get() != slow_id {
+                assert_eq!(current.root(), prior[&current.session().get()].0 + 1);
+                assert_eq!(current.frontier(), prior[&current.session().get()].1 + 1);
+                healthy += 1;
+            }
+        }
+        assert_eq!(healthy, width - 1, "every healthy survivor progresses");
+        let slow = active
+            .iter()
+            .find(|lane| lane.session.binding().session().get() == slow_id)
+            .unwrap();
+        assert_eq!(slow.session.binding(), binding);
+        assert!(slow.packet_presented && slow.blocked_since.is_some());
+        let token = slow.device_token.as_ref().unwrap();
+        assert_eq!((token.id, token.max_bits), packet);
+        for row in observed.iter_mut().skip(1) {
+            row.trace.drain();
+        }
+    }
+    observed[0].trace.one_credit();
+    finish_observed(&mut active, fixture, &mut references, &mut observed);
+    println!(
+        "P12_WIDE_REAL_CREDIT_PASS rows={width} channel_capacity=1 \
+        healthy_survivors_progress=true parked_packet_root_all42=exact resumed_events=once"
+    );
+}
+
+fn wide_source_eos(fixture: &Fixture) {
+    let width = fixture.model.native_rows;
+    let input =
+        find_source_eos(fixture).expect("wide cohort requires an actual trained-source EOS");
+    let length = fixture
+        .model
+        .tokenizer
+        .prompt_tokens(&input, false)
+        .unwrap()
+        .len();
+    let sibling = " a".repeat(length);
+    assert_eq!(
+        fixture
+            .model
+            .tokenizer
+            .prompt_tokens(&sibling, false)
+            .unwrap()
+            .len(),
+        length
+    );
+    let mut active = VecDeque::new();
+    let mut observed = Vec::new();
+    for index in 0..width {
+        let (lane, trace) = observed_lane(
+            fixture,
+            if index == 0 { &input } else { &sibling },
+            if index == 0 { 9 } else { 10 },
+            EVENTS_PER_REQUEST,
+            false,
+        );
+        active.push_back(lane);
+        observed.push(trace);
+    }
+    let bands = band_counts(&fixture.stats);
+    for _ in 0..32 {
+        if active.iter().all(|lane| lane.cursor == lane.prompt.len()) {
+            break;
+        }
+        round(&mut active, &fixture.stats);
+    }
+    assert_eq!(active.len(), width);
+    assert!(active.iter().all(|lane| lane.cursor == lane.prompt.len()));
+    require_bands(&fixture.stats, bands, &[width]);
+    let mut references = reference_rows(fixture, &active);
+    finish_observed(&mut active, fixture, &mut references, &mut observed);
+    assert_eq!(
+        observed[0].trace.done[0].3, "stop",
+        "trained EOS must retire its row"
+    );
+    println!(
+        "P12_WIDE_TRAINED_EOS_PASS rows={width} source_selected_EOS=true \
+        CPU_B1_packets_all42=exact EOS_excluded_usage=true terminal_Done=once"
+    );
+}
+
+fn qualify_wide(name: &str, native_rows: usize) {
+    let fixture = Fixture::with_native_rows(name, native_rows);
+    wide_prompt_lengths(&fixture);
+    wide_b1_handoff(&fixture);
+    wide_cancel_after_ready(&fixture);
+    wide_credit_park(&fixture);
+    wide_source_eos(&fixture);
+    assert!(fixture.stats.kernel_dispatches.load(Ordering::Acquire) > 0);
+    assert!(fixture.stats.native_projections.load(Ordering::Acquire) > 0);
+    assert!(fixture.stats.native_sampler.load(Ordering::Acquire) > 0);
+    assert!(band_counts(&fixture.stats)[if native_rows == 4 { 1 } else { 2 }] > 0);
+    assert_eq!(
+        fixture
+            .stats
+            .native_admission_refusals
+            .load(Ordering::Acquire),
+        0
+    );
+    fixture.close();
+    println!(
+        "P12_WIDE_NATIVE_WORKER_PHYSICAL_PASS source={name} rows={native_rows} \
+        original_authorities_CPU_B1_all42=exact final_active=0 final_reserved=0 \
+        actual_HTTP_wire=separate_gate vLLM_performance=not_qualified"
+    );
+}
+
 #[test]
 #[ignore = "physical release CUDA; approved original .8B; isolated sequential GPU coordinator"]
 fn original_qwen08b_native_b2_worker() {
@@ -702,3 +1138,27 @@ fn original_qwen08b_native_b2_worker() {
 fn original_qwen2b_native_b2_worker() {
     qualify("qwen35-2b");
 }
+
+#[test]
+#[ignore = "physical release CUDA B4; approved original .8B; isolated sequential GPU coordinator"]
+fn original_qwen08b_native_b4_worker() {
+    qualify_wide("qwen35-0.8b", 4);
+}
+
+#[test]
+#[ignore = "physical release CUDA B8; approved original .8B; isolated sequential GPU coordinator"]
+fn original_qwen08b_native_b8_worker() {
+    qualify_wide("qwen35-0.8b", 8);
+}
+
+#[test]
+#[ignore = "physical release CUDA B4; approved original 2B; isolated sequential GPU coordinator"]
+fn original_qwen2b_native_b4_worker() {
+    qualify_wide("qwen35-2b", 4);
+}
+
+#[test]
+#[ignore = "physical release CUDA B8; approved original 2B; isolated sequential GPU coordinator"]
+fn original_qwen2b_native_b8_worker() {
+    qualify_wide("qwen35-2b", 8);
+}
diff --git a/crates/zerogravity-cli/src/serve/mod.rs b/crates/zerogravity-cli/src/serve/mod.rs
index 21fe39a..059a3b6 100644
--- a/crates/zerogravity-cli/src/serve/mod.rs
+++ b/crates/zerogravity-cli/src/serve/mod.rs
@@ -52,8 +52,8 @@ impl Router {
                 let m = v.metrics();
                 serde_json::json!({"kind":"resident_gpu",
                     "physical_accounting":"ResidentService/ResourceArbiter",
-                    "batch_mode":if m.native_b2 {"fused"} else {"interleaved"},
-                    "native_batch_rows_max":if m.native_b2 {2} else {1},
+                    "batch_mode":if m.native_rows != 0 {"fused"} else {"interleaved"},
+                    "native_batch_rows_max":m.native_rows.max(1),
                     "gpu_step_attempts":m.attempted,
                     "completed_gpu_quanta":m.completed,
                     "completed_quantum_unit":"request work, including terminal presentation",
@@ -65,6 +65,8 @@ impl Router {
                     "native_projection_dispatches_completed":m.native_projections,
                     "native_sampler_dispatches_completed":m.native_sampler,
                     "gpu_fused_rows_completed":m.fused_rows,
+                    "gpu_fused_batches_by_rows":{"2":m.native_bands[0],"4":m.native_bands[1],"8":m.native_bands[2]},
+                    "native_admission_refusals":m.native_admission_refusals,
                     "discarded_ready_cohorts":m.discarded_cohorts,
                     "accepted_output_tokens":m.output_tokens,
                     "output_credit_parked_requests":m.parked,
@@ -111,6 +113,8 @@ struct YamlConfig {
     device_mib: usize,
     #[serde(default = "default_batch_mode")]
     batch_mode: String,
+    #[serde(default = "default_batch_rows")]
+    batch_rows: usize,
 }
 #[derive(Debug, Deserialize)]
 #[serde(deny_unknown_fields)]
@@ -145,6 +149,9 @@ fn default_provider_mib() -> usize {
 fn default_device_mib() -> usize {
     4096
 }
+fn default_batch_rows() -> usize {
+    2
+}
 fn default_batch_mode() -> String {
     "interleaved".into()
 }
@@ -163,6 +170,7 @@ impl YamlConfig {
             provider_mib: default_provider_mib(),
             device_mib: default_device_mib(),
             batch_mode: default_batch_mode(),
+            batch_rows: default_batch_rows(),
         };
         if args.first().is_some_and(|x| x == "--config") {
             if args.len() != 2 {
@@ -219,6 +227,7 @@ impl YamlConfig {
                 "--provider-mib" => config.provider_mib = parse_usize(value, flag)?,
                 "--device-mib" => config.device_mib = parse_usize(value, flag)?,
                 "--batch-mode" => config.batch_mode = value.clone(),
+                "--batch-rows" => config.batch_rows = parse_usize(value, flag)?,
                 _ => {
                     return Err(ServiceError::new(
                         400,
@@ -240,14 +249,19 @@ fn parse_usize(value: &str, flag: &str) -> Result<usize, ServiceError> {
         )
     })
 }
-fn validate_batch_mode(mode: &str, backend: &str, active: usize) -> Result<(), ServiceError> {
+fn validate_batch_mode(
+    mode: &str,
+    backend: &str,
+    active: usize,
+    rows: usize,
+) -> Result<(), ServiceError> {
     match mode {
         "interleaved" => Ok(()),
-        "fused" if backend == "cuda" && active >= 2 => Ok(()),
+        "fused" if backend == "cuda" && matches!(rows, 2 | 4 | 8) && active >= rows => Ok(()),
         _ => Err(ServiceError::new(
             400,
             "unsupported_batch_mode",
-            "use interleaved, or fused with CUDA and at least two active slots",
+            "use interleaved, or fused with CUDA, rows=2/4/8 and active>=rows",
         )),
     }
 }
@@ -261,13 +275,14 @@ pub fn run(args: &[String]) -> Result<(), ServiceError> {
              [--workers 2] [--active-per-worker 2] \
              [--max-context 1024] [--max-output 128] \
              [--host-mib 4096] [--provider-mib 2048] [--device-mib 4096] \
-             [--batch-mode interleaved|fused]\n\
+             [--batch-mode interleaved|fused] [--batch-rows 2|4|8]\n\
              ENDPOINTS: GET /health, GET /v1/models, \
              POST /v1/chat/completions, POST /v1/completions.\n\
              Qwen3.5 greedy text CPU or resident GPU, no inference fallback. \
              GPU: workers=1, context=512. Default interleaved; fused enables \
-             CUDA B2 matching-source/frontier prompt tails and decode \
-             with active>=2. Unmatched rows and other prefill use B1. \
+             CUDA B2/B4/B8 matching-source/frontier prompt tails and decode \
+             with active>=rows. Default rows=2; survivors regroup without padding. \
+             Unmatched rows and other prefill use B1. \
              CUDA needs CUDA_PATH with runtime+CCCL headers. \
              Loopback only; optional ZEROGRAVITY_API_KEY."
         );
@@ -295,6 +310,7 @@ pub fn run(args: &[String]) -> Result<(), ServiceError> {
         &options.batch_mode,
         &options.backend,
         options.active_per_worker,
+        options.batch_rows,
     )?;
     let declared = options.models.into_iter().map(|m| (m.id, m.path)).collect();
     let backend = options.backend.as_str();
@@ -333,7 +349,11 @@ pub fn run(args: &[String]) -> Result<(), ServiceError> {
                         output: options.max_output,
                         provider_mib: options.provider_mib,
                         device_mib: options.device_mib,
-                        native_b2: options.batch_mode == "fused",
+                        native_rows: if options.batch_mode == "fused" {
+                            options.batch_rows
+                        } else {
+                            0
+                        },
                     },
                 )?)
             }
@@ -369,9 +389,9 @@ pub fn run(args: &[String]) -> Result<(), ServiceError> {
 mod tests {
     use super::*;
     #[test]
-    fn native_serving_mode_is_cuda_b2_only_before_model_or_device_lookup() {
+    fn native_serving_mode_is_bounded_before_model_or_device_lookup() {
         for backend in ["cpu", "cuda", "vulkan-nvidia", "vulkan-amd", "rocm"] {
-            assert!(validate_batch_mode("interleaved", backend, 1).is_ok());
+            assert!(validate_batch_mode("interleaved", backend, 1, 2).is_ok());
             if backend != "cuda" {
                 let err = run(&[
                     "--backend".into(),
@@ -385,8 +405,13 @@ mod tests {
                 assert_eq!(err.code, "unsupported_batch_mode");
             }
         }
-        assert!(validate_batch_mode("fused", "cuda", 2).is_ok());
-        assert!(validate_batch_mode("fused", "cuda", 8).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 2, 2).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 4, 4).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 8, 8).is_ok());
+        for rows in [0, 1, 3, 5, 6, 7, 9, 16] {
+            assert!(validate_batch_mode("fused", "cuda", 8, rows).is_err());
+        }
+        assert!(validate_batch_mode("fused", "cuda", 4, 8).is_err());
         for mode in ["fused", "fused-b4", "unknown"] {
             let err = run(&[
                 "--backend".into(),
diff --git a/crates/zerogravity-cli/tests/serve_gpu.rs b/crates/zerogravity-cli/tests/serve_gpu.rs
index 526922d..13597aa 100644
--- a/crates/zerogravity-cli/tests/serve_gpu.rs
+++ b/crates/zerogravity-cli/tests/serve_gpu.rs
@@ -28,6 +28,13 @@ struct OwnedServer {
 
 impl OwnedServer {
     fn start(source: &Path) -> Proof<Self> {
+        Self::start_rows(source, 2)
+    }
+
+    fn start_rows(source: &Path, rows: usize) -> Proof<Self> {
+        if !matches!(rows, 2 | 4 | 8) {
+            return Err("physical HTTP server requires native rows2/4/8".into());
+        }
         for _ in 0..3 {
             let reserved = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
             let address = reserved.local_addr().map_err(|e| e.to_string())?;
@@ -42,12 +49,14 @@ impl OwnedServer {
                 .arg("cuda")
                 .arg("--batch-mode")
                 .arg("fused")
+                .arg("--batch-rows")
+                .arg(rows.to_string())
                 .arg("--listen")
                 .arg(address.to_string())
                 .arg("--workers")
                 .arg("1")
                 .arg("--active-per-worker")
-                .arg("4")
+                .arg(if rows == 2 { "4" } else { "8" })
                 .arg("--max-context")
                 .arg("512")
                 .arg("--max-output")
@@ -108,6 +117,7 @@ impl OwnedServer {
                 if let Ok(health) = server.health()
                     && health["status"] == "ok"
                     && health["admission"]["batch_mode"] == "fused"
+                    && health["admission"]["native_batch_rows_max"] == rows
                     && let Ok(models) = get_json(server.address, "/v1/models")
                     && models["data"].as_array().is_some_and(|models| {
                         models.len() == 2
@@ -150,7 +160,11 @@ impl OwnedServer {
     }
 
     fn wait_health(&self, predicate: impl Fn(&Value) -> bool) -> Proof<Value> {
-        let deadline = Instant::now() + REQUEST_LIMIT;
+        self.wait_health_for(REQUEST_LIMIT, predicate)
+    }
+
+    fn wait_health_for(&self, limit: Duration, predicate: impl Fn(&Value) -> bool) -> Proof<Value> {
+        let deadline = Instant::now() + limit;
         let mut last = Value::Null;
         while Instant::now() < deadline {
             if let Ok(health) = self.health() {
@@ -586,9 +600,18 @@ fn sse_completion(
 }
 
 fn disconnect_after_first_text(
+    response: Response,
+    alias: &str,
+    expected_prefix: &str,
+) -> Proof<()> {
+    disconnect_text_when(response, alias, expected_prefix, || Ok(()))
+}
+
+fn disconnect_text_when(
     mut response: Response,
     alias: &str,
     expected_prefix: &str,
+    before_disconnect: impl FnOnce() -> Proof<()>,
 ) -> Proof<()> {
     if response.status != 200 || !response.chunked {
         return Err("canceled request lacked real SSE response".into());
@@ -609,6 +632,7 @@ fn disconnect_after_first_text(
                 if !expected_prefix.starts_with(text) {
                     return Err("cancel target text differs from CPU prefix".into());
                 }
+                before_disconnect()?;
                 response.disconnect();
                 return Ok(());
             }
@@ -651,6 +675,66 @@ fn queued_pair(server: &OwnedServer, first: &Value, second: &Value) -> Proof<(Re
     Ok((Response::read(first)?, Response::read(second)?))
 }
 
+/// Real queued requests, all written before reading any response. Cancel the
+/// observable GPU quantum's owner so active8 can admit all eight queued rows.
+fn queued_group(server: &OwnedServer, bodies: &[Value]) -> Proof<Vec<Response>> {
+    if !matches!(bodies.len(), 4 | 8) {
+        return Err("wide HTTP group requires four or eight real requests".into());
+    }
+    let before = server.health()?["admission"].clone();
+    let attempts = counter(&before, "gpu_step_attempts")?;
+    let tokens = counter(&before, "completed_model_token_steps")?;
+    let gate_body = payload(
+        &server.aliases[0],
+        &"A real GPU prefill gate. ".repeat(12),
+        8,
+        true,
+    );
+    let gate = open_request(server.address, "POST", "/v1/completions", Some(&gate_body))?;
+    server.wait_health(|health| {
+        health["active_sessions"] == 1
+            && health["gpu_step_attempts"]
+                .as_u64()
+                .is_some_and(|n| n > attempts)
+            && health["completed_model_token_steps"].as_u64() == Some(tokens)
+    })?;
+    let streams = bodies
+        .iter()
+        .map(|body| open_request(server.address, "POST", "/v1/completions", Some(body)))
+        .collect::<Proof<Vec<_>>>()?;
+    server.wait_health_for(Duration::from_secs(5), |health| {
+        health["queued_requests"] == bodies.len()
+            && health["completed_model_token_steps"].as_u64() == Some(tokens)
+    })?;
+    let _ = gate.shutdown(Shutdown::Both);
+    drop(gate);
+    streams.into_iter().map(Response::read).collect()
+}
+
+fn band_counter(health: &Value, width: usize) -> Proof<u64> {
+    counter(&health["gpu_fused_batches_by_rows"], &width.to_string())
+}
+
+fn physical_accounting(health: &Value) -> Proof<()> {
+    let b2 = band_counter(health, 2)?;
+    let b4 = band_counter(health, 4)?;
+    let b8 = band_counter(health, 8)?;
+    let batches = b2 + b4 + b8;
+    if counter(health, "failed_gpu_quanta")? != 0
+        || health["active_sessions"] != 0
+        || health["queued_requests"] != 0
+        || counter(health, "completed_forward_kernel_dispatches")? == 0
+        || counter(health, "native_projection_dispatches_completed")? == 0
+        || counter(health, "gpu_fused_batches")? != batches
+        || counter(health, "gpu_fused_rows_completed")? != 2 * b2 + 4 * b4 + 8 * b8
+        || counter(health, "native_sampler_dispatches_completed")? != 4 * batches
+        || counter(health, "native_admission_refusals")? != 0
+    {
+        return Err(format!("wide HTTP physical accounting differs: {health}"));
+    }
+    Ok(())
+}
+
 fn proof_cases(
     server: &OwnedServer,
     oracle: &Oracle,
@@ -759,6 +843,156 @@ fn proof_cases(
     Ok(())
 }
 
+fn validate_group(
+    responses: Vec<Response>,
+    bodies: &[Value],
+    oracle: &Oracle,
+    tokenizer: &QwenTokenizer,
+) -> Proof<()> {
+    if responses.len() != bodies.len() {
+        return Err("HTTP group lost a request/response".into());
+    }
+    for (response, body) in responses.into_iter().zip(bodies) {
+        let alias = body["model"].as_str().ok_or("group alias missing")?;
+        let maximum = body["max_tokens"].as_u64().ok_or("group maximum missing")? as usize;
+        let expected = oracle.expected(tokenizer, maximum)?;
+        if body["stream"] == true {
+            sse_completion(response, alias, &expected, oracle.prompt_tokens)?;
+        } else {
+            json_completion(response, alias, &expected, oracle.prompt_tokens)?;
+        }
+    }
+    Ok(())
+}
+
+fn wide_proof_cases(
+    server: &OwnedServer,
+    width: usize,
+    oracle: &Oracle,
+    eos: &Oracle,
+    tokenizer: &QwenTokenizer,
+) -> Proof<()> {
+    let initial = server.health()?["admission"].clone();
+    if initial["batch_mode"] != "fused" || initial["native_batch_rows_max"] != width {
+        return Err(format!("HTTP did not advertise native CUDA B{width}"));
+    }
+    if oracle.ids.len() != 8 || oracle.stopped {
+        return Err("wide retirement proof requires eight original nonterminal tokens".into());
+    }
+    let bodies = (0..width)
+        .map(|index| {
+            payload(
+                &server.aliases[index % 2],
+                &oracle.prompt,
+                (index + 2).min(8),
+                index % 2 == 1,
+            )
+        })
+        .collect::<Vec<_>>();
+    let responses = queued_group(server, &bodies)?;
+    validate_group(responses, &bodies, oracle, tokenizer)?;
+    let lengths = server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
+    if band_counter(&lengths, width)? < band_counter(&initial, width)? + 2
+        || band_counter(&lengths, 2)? <= band_counter(&initial, 2)?
+        || (width == 8 && band_counter(&lengths, 4)? <= band_counter(&initial, 4)?)
+    {
+        return Err(format!(
+            "actual wide HTTP group lacks full-width and odd-survivor receipts: {lengths}"
+        ));
+    }
+    physical_accounting(&lengths)?;
+
+    let eos_expected = eos.expected(tokenizer, 8)?;
+    if !eos.stopped || eos_expected.2 != "stop" || eos_expected.1 >= 8 {
+        return Err("wide HTTP proof lacks original trained-EOS evidence".into());
+    }
+    let eos_bodies = (0..width)
+        .map(|index| payload(&server.aliases[index % 2], &eos.prompt, 8, index % 2 == 0))
+        .collect::<Vec<_>>();
+    let responses = queued_group(server, &eos_bodies)?;
+    validate_group(responses, &eos_bodies, eos, tokenizer)?;
+    let after_eos =
+        server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
+    if band_counter(&after_eos, width)? <= band_counter(&lengths, width)? {
+        return Err("trained-EOS group did not physically execute its native width".into());
+    }
+    physical_accounting(&after_eos)?;
+
+    let expected8 = oracle.expected(tokenizer, 8)?;
+    let cancel_bodies = (0..width)
+        .map(|index| payload(&server.aliases[index % 2], &oracle.prompt, 8, true))
+        .collect::<Vec<_>>();
+    let mut responses = queued_group(server, &cancel_bodies)?;
+    let cancel = responses.remove(0);
+    let mut survivors = Vec::new();
+    for (index, response) in responses.into_iter().enumerate() {
+        let alias = server.aliases[(index + 1) % 2].clone();
+        let expected = expected8.clone();
+        let prompt_tokens = oracle.prompt_tokens;
+        survivors.push(thread::spawn(move || {
+            sse_completion(response, &alias, &expected, prompt_tokens)
+        }));
+    }
+    let disconnected = disconnect_text_when(cancel, &server.aliases[0], &expected8.0, || {
+        let live = server.health()?["admission"].clone();
+        let accepted = counter(&live, "accepted_output_tokens")?
+            - counter(&after_eos, "accepted_output_tokens")?;
+        if live["active_sessions"] != width
+            || live["queued_requests"] != 0
+            || band_counter(&live, width)? <= band_counter(&after_eos, width)?
+            || accepted >= (8 * width) as u64
+        {
+            return Err(format!(
+                "cancel target was not demonstrably live in the real wide cohort: {live}"
+            ));
+        }
+        Ok(())
+    });
+    let mut survivor_error = None;
+    for survivor in survivors {
+        let result = survivor
+            .join()
+            .map_err(|_| "wide HTTP survivor reader panicked".to_owned())
+            .and_then(|result| result);
+        if let Err(error) = result {
+            survivor_error.get_or_insert(error);
+        }
+    }
+    disconnected?;
+    if let Some(error) = survivor_error {
+        return Err(error);
+    }
+    let cancelled = counter(&after_eos, "cancelled_requests")?;
+    let after_cancel = server.wait_health(|h| {
+        h["active_sessions"] == 0
+            && h["queued_requests"] == 0
+            && h["cancelled_requests"]
+                .as_u64()
+                .is_some_and(|count| count >= cancelled + 2)
+    })?;
+    if band_counter(&after_cancel, width)? <= band_counter(&after_eos, width)? {
+        return Err("live canceled HTTP group never executed its native width".into());
+    }
+    let replay_body = payload(&server.aliases[0], &oracle.prompt, 8, true);
+    let replay = Response::read(open_request(
+        server.address,
+        "POST",
+        "/v1/completions",
+        Some(&replay_body),
+    )?)?;
+    sse_completion(replay, &server.aliases[0], &expected8, oracle.prompt_tokens)?;
+    let final_health =
+        server.wait_health(|h| h["active_sessions"] == 0 && h["queued_requests"] == 0)?;
+    physical_accounting(&final_health)?;
+    println!(
+        "P12_HTTP_CUDA_NATIVE_WIDE_PASS rows={width} actual_queued_aliases=shared \
+        JSON_SSE_text_usage=CPU_exact trained_EOS_JSON_SSE=stop_exact terminal_usage_DONE=once \
+        live_disconnect_survivors_replay=exact weighted_native_rows_and_sampler=exact \
+        all42_state=internal_gate vLLM_performance=not_qualified health={final_health}"
+    );
+    Ok(())
+}
+
 fn qualify(name: &str) -> Proof<()> {
     let _serial = GPU_HTTP_LOCK
         .lock()
@@ -810,6 +1044,61 @@ fn qualify(name: &str) -> Proof<()> {
     })
 }
 
+fn qualify_wide(name: &str) -> Proof<()> {
+    let _serial = GPU_HTTP_LOCK
+        .lock()
+        .map_err(|_| "physical HTTP test lock poisoned")?;
+    if cfg!(debug_assertions) {
+        return Err("physical HTTP proof requires cargo test --release".into());
+    }
+    let directory: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
+        .join("../../target/models")
+        .join(name);
+    let model_path = directory.join("model.gguf");
+    for file in [
+        &model_path,
+        &directory.join("config.json"),
+        &directory.join("tokenizer.json"),
+    ] {
+        if !file.is_file() {
+            return Err(format!(
+                "approved original source missing: {}",
+                file.display()
+            ));
+        }
+    }
+    let source = LoadedQwenModel::open(&model_path, &directory.join("config.json"))
+        .map_err(|e| e.to_string())?;
+    let tokenizer =
+        QwenTokenizer::from_file(&directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
+    let mut chosen = None;
+    for prompt in [
+        "The sequence is 1, 2, 3,",
+        "Continue counting: one, two, three,",
+        "A simple example is",
+    ] {
+        let oracle = Oracle::compute(&source, &tokenizer, prompt)?;
+        if oracle.ids.len() == 8 && !oracle.stopped && !oracle.expected(&tokenizer, 8)?.0.is_empty()
+        {
+            chosen = Some(oracle);
+            break;
+        }
+    }
+    let oracle =
+        chosen.ok_or("no bounded original prompt supplied eight nonterminal source tokens")?;
+    let eos = find_source_eos(&source, &tokenizer)?;
+    for width in [4, 8] {
+        let server = OwnedServer::start_rows(&model_path, width)?;
+        wide_proof_cases(&server, width, &oracle, &eos, &tokenizer).map_err(|error| {
+            format!(
+                "{name} B{width}: {error}\nowned HTTP server stderr:\n{}",
+                server.diagnostics()
+            )
+        })?;
+    }
+    Ok(())
+}
+
 #[test]
 #[ignore = "release physical CUDA HTTP; approved original .8B source; run serially"]
 fn original_qwen35_08b_native_b2_http_json_sse_disconnect_and_replay() -> Proof<()> {
@@ -821,3 +1110,15 @@ fn original_qwen35_08b_native_b2_http_json_sse_disconnect_and_replay() -> Proof<
 fn original_qwen35_2b_native_b2_http_json_sse_disconnect_and_replay() -> Proof<()> {
     qualify("qwen35-2b")
 }
+
+#[test]
+#[ignore = "release physical CUDA B4/B8 HTTP; approved original .8B source; run serially"]
+fn original_qwen35_08b_native_wide_http_json_sse_disconnect_and_replay() -> Proof<()> {
+    qualify_wide("qwen35-0.8b")
+}
+
+#[test]
+#[ignore = "release physical CUDA B4/B8 HTTP; approved original 2B source; run serially"]
+fn original_qwen35_2b_native_wide_http_json_sse_disconnect_and_replay() -> Proof<()> {
+    qualify_wide("qwen35-2b")
+}
diff --git a/crates/zerogravity-gpu/src/resident_inference.rs b/crates/zerogravity-gpu/src/resident_inference.rs
index c899665..f50380e 100644
--- a/crates/zerogravity-gpu/src/resident_inference.rs
+++ b/crates/zerogravity-gpu/src/resident_inference.rs
@@ -9,6 +9,9 @@ mod alias_qualification;
 mod checkpoint;
 #[cfg(all(feature = "portable-checkpoint", feature = "qualification"))]
 pub(crate) use checkpoint::qualify as qualify_checkpoint;
+#[cfg(all(test, feature = "qualification", feature = "cuda"))]
+#[path = "resident_inference/admission_qualification.rs"]
+mod admission_qualification;
 #[cfg(feature = "qualification")]
 #[path = "resident_inference/cohort_qualification.rs"]
 mod cohort_qualification;
@@ -520,15 +523,26 @@ impl ResidentModel {
             drop(initialization.complete_sync().await?);
             slot.finish()?;
         } else {
-            for r in &self.states {
+            output = self.allocate_uninitialized_state()?;
+        }
+        Ok(output)
+    }
+    /// Caller holds the original allocation admission lock; no GPU command.
+    fn allocate_uninitialized_state(&self) -> Result<Vec<Domain>> {
+        self.service.ensure_usable()?;
+        if self.state_bytes > self.service.remaining_capacity()? {
+            return Err(Error::Limit("resident state admission exhausted".into()));
+        }
+        self.states
+            .iter()
+            .map(|r| {
                 let mut allocate = |n: usize| {
                     self.service
                         .allocate_uninitialized(n * 4, BufferRole::RecurrentState)
                 };
-                output.push(make_domain(r, &mut allocate)?);
-            }
-        }
-        Ok(output)
+                make_domain(r, &mut allocate)
+            })
+            .collect()
     }
     fn validate_domains(&self, domains: &[Domain], initialized: bool) -> Result<()> {
         if domains.len() != self.states.len() {
diff --git a/crates/zerogravity-gpu/src/resident_inference/fused.rs b/crates/zerogravity-gpu/src/resident_inference/fused.rs
index 2a1bc9b..7e2b11b 100644
--- a/crates/zerogravity-gpu/src/resident_inference/fused.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/fused.rs
@@ -20,6 +20,16 @@ impl ResidentModel {
         self.prepare_fused_mode(rows, None, Readout::Greedy).await
     }
 
+    /// None means only a checked capacity refusal before any GPU submission.
+    /// Errors, including unknown completion, must never select a smaller retry.
+    pub async fn prepare_greedy_fused_cohort_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, u32)],
+    ) -> Result<Option<PreparedResidentCohort>> {
+        self.prepare_fused_mode_admitted(rows, None, Readout::Greedy)
+            .await
+    }
+
     /// Read original packet words on the GPU under their exact producing roots.
     /// Public host ID/max metadata never selects the next source embedding row.
     pub async fn prepare_device_tokens_fused_cohort(
@@ -35,12 +45,37 @@ impl ResidentModel {
             .await
     }
 
+    /// Device packets retain their original producer and physical lease checks.
+    pub async fn prepare_device_tokens_fused_cohort_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, &ResidentToken)],
+    ) -> Result<Option<PreparedResidentCohort>> {
+        let sessions = rows
+            .iter()
+            .map(|(session, _)| (*session, 0))
+            .collect::<Vec<_>>();
+        let tokens = rows.iter().map(|(_, token)| *token).collect::<Vec<_>>();
+        self.prepare_fused_mode_admitted(&sessions, Some(&tokens), Readout::Greedy)
+            .await
+    }
+
     async fn prepare_fused_mode(
         self: &Arc<Self>,
         rows: &[(&ResidentSession, u32)],
         device_tokens: Option<&[&ResidentToken]>,
         readout: Readout,
     ) -> Result<PreparedResidentCohort> {
+        self.prepare_fused_mode_admitted(rows, device_tokens, readout)
+            .await?
+            .ok_or_else(|| Error::Limit("native resident batch admission exhausted".into()))
+    }
+
+    async fn prepare_fused_mode_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, u32)],
+        device_tokens: Option<&[&ResidentToken]>,
+        readout: Readout,
+    ) -> Result<Option<PreparedResidentCohort>> {
         if !self.native_batch {
             return Err(Error::Unsupported(
                 "native resident batch requires CUDA".into(),
@@ -48,6 +83,11 @@ impl ResidentModel {
         }
         let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
         let roots = ResidentBatchRows::new(&bases, 1)?;
+        if !matches!(rows.len(), 2 | 4 | 8) {
+            return Err(Error::Unsupported(
+                "resident native batch rows must be 2/4/8".into(),
+            ));
+        }
         if roots.source() != self.service.source_identity() {
             return Err(Error::State("native resident batch source mismatch".into()));
         }
@@ -87,9 +127,60 @@ impl ResidentModel {
             self.validate_domains(&session.domains, true)?;
             attempts.push(session.authority.begin_advance(1)?);
         }
+        let staging_bytes = self
+            .state_bytes
+            .checked_mul(rows.len())
+            .ok_or_else(|| Error::Limit("native resident batch admission overflow".into()))?;
+        let packet_bytes = if matches!(readout, Readout::Greedy) {
+            rows.len() * 12
+        } else {
+            0
+        };
+        let allocation = self.allocation.lock().await;
+        // This lock follows any prior actual terminal completion. Account the
+        // complete candidate/packet/workspace footprint before retiring an idle
+        // band or allocating a single candidate. No scheduler byte credits.
+        {
+            let mut workspace = self.workspace.lock().await;
+            self.service.ensure_usable()?;
+            let same_band = workspace
+                .batch
+                .as_ref()
+                .is_some_and(|b| b.rows == rows.len());
+            let reclaimable = if same_band {
+                0
+            } else {
+                workspace
+                    .batch
+                    .as_ref()
+                    .map(|b| b.program.budget().owned_bytes())
+                    .transpose()?
+                    .unwrap_or(0)
+            };
+            let workspace_bytes = if same_band {
+                0
+            } else {
+                BatchProgram::preflight(&workspace.program, rows.len())?.owned_bytes()?
+            };
+            let required = staging_bytes
+                .checked_add(packet_bytes)
+                .and_then(|n| n.checked_add(workspace_bytes))
+                .ok_or_else(|| Error::Limit("native resident batch admission overflow".into()))?;
+            let available = self
+                .service
+                .remaining_capacity()?
+                .checked_add(reclaimable)
+                .ok_or_else(|| Error::Limit("native resident batch capacity overflow".into()))?;
+            if required > available || staging_bytes as u64 > self.service.budget().remaining()? {
+                return Ok(None);
+            }
+            if !same_band {
+                workspace.batch = None;
+            }
+        }
         let mut candidates = Vec::with_capacity(rows.len());
         for _ in rows {
-            candidates.push(self.allocate_state(false).await?);
+            candidates.push(self.allocate_uninitialized_state()?);
         }
         let bindings = rows
             .iter()
@@ -104,13 +195,8 @@ impl ResidentModel {
                 valid_tokens: position + 1,
             })
             .collect::<Vec<_>>();
-        let staging_bytes = self
-            .state_bytes
-            .checked_mul(rows.len())
-            .ok_or_else(|| Error::Limit("native resident batch admission overflow".into()))?;
         // State initialization uses allocation -> queue. Preserve that order
         // while admitting a cold batch workspace, before selecting a slot.
-        let allocation = self.allocation.lock().await;
         let packets = if matches!(readout, Readout::Greedy) {
             (0..rows.len())
                 .map(|_| self.service.allocate_uninitialized(12, BufferRole::Output))
@@ -473,13 +559,13 @@ impl ResidentModel {
                 },
             });
         }
-        Ok(PreparedResidentCohort {
+        Ok(Some(PreparedResidentCohort {
             model: Arc::clone(self),
             roots,
             prepared,
             dispatches: submitted.dispatches,
             fused_dispatches: submitted.fused_dispatches,
             native_greedy_dispatches,
-        })
+        }))
     }
 }

````

## Source: .agents/runs/serving-and-speculation-parity-root-2026-10-19/command-host-fixture-strict-followup.patch

Original UTF-8 bytes: 2399; SHA256 `b87fdb7047e6f0f054a4e1efcccd6b2a0eafbb2e5258bcc77f656386ef20bed0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/src/device/handle/admission.rs
+++ b/src/device/handle/admission.rs
@@ -63,7 +63,7 @@
 pub struct CommandRequest<'a> {
     /// Actual transport registry service; not a PCI-card identity.
     pub service: ServiceId,
-    /// `None` stays unbound; no inference from the caller's ambient StreamId.
+    /// `None` stays unbound; no inference from the caller's ambient `StreamId`.
     pub stream: Option<StreamId>,
     /// Host operation being admitted.
     pub kind: CommandKind,
@@ -191,7 +191,7 @@
         self.service
     }
 
-    /// Explicit pinned Client logical stream, never ambient StreamId::current.
+    /// Explicit pinned Client logical stream, never ambient `StreamId::current`.
     pub fn stream_id(&self) -> StreamId {
         self.stream
     }
@@ -226,16 +226,15 @@
     hook.map(|hook| hook.admit(request)).transpose()
 }
 
+type BootstrapAdmission = (
+    Option<Arc<dyn CommandAdmission>>,
+    Option<OwnedCommandPermit>,
+);
+
 pub(super) fn bootstrap<S: DeviceService>(
     device: DeviceId,
     kind: CommandKind,
-) -> Result<
-    (
-        Option<Arc<dyn CommandAdmission>>,
-        Option<OwnedCommandPermit>,
-    ),
-    CommandAdmissionError,
-> {
+) -> Result<BootstrapAdmission, CommandAdmissionError> {
     // MUST be called before CHANNELS/GLOBAL/DEVICE_REGISTRY or actor locks.
     let hook = S::bootstrap_command_admission(device);
     let permit = admit(
--- a/src/device/handle/admission/tests.rs
+++ b/src/device/handle/admission/tests.rs
@@ -54,13 +54,14 @@
         let gate = Arc::as_ptr(&self.0) as usize;
         let mut capture = request.scope.is_some();
         ENTERED.with(|slot| {
-            if let Some(entered) = slot.get() {
-                if entered.gate == gate && entered.capture {
-                    if entered.service != request.service || entered.stream != request.stream {
-                        return Err(CommandAdmissionError::ForeignScope);
-                    }
-                    capture = true;
+            if let Some(entered) = slot.get()
+                && entered.gate == gate
+                && entered.capture
+            {
+                if entered.service != request.service || entered.stream != request.stream {
+                    return Err(CommandAdmissionError::ForeignScope);
                 }
+                capture = true;
             }
             Ok(())
         })?;

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 49892; SHA256 `c53732bb9acca5e980f18de8aa5fcbc99d4a273e3903c4c493812c26ea873ee8`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch/graph.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch/graph.rs
@@ -114,7 +114,7 @@
         {
             return Err(Error::State("stale/unpinned graph body receipt".into()));
         }
-        let prefix = self.launch_inputs(client, parameters, bindings, None);
+        let prefix = self.launch_inputs(client, parameters, bindings, None)?;
         // SAFETY: enclosing physical flight retains original graph and every
         // source/workspace lease; same pinned stream/workspace lock orders IO.
         unsafe { graph.replay() }
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification/physical.rs
@@ -0,0 +1,447 @@
+//! Physical original-per-row vs native bit copy, with original leases/Ready.
+use crate::resident_leases::{ResidentLease, ResidentService};
+use crate::resident_state_transport::*;
+use cubecl::{Device, bytes::Bytes};
+use std::{path::Path, sync::Arc};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+use zerogravity_model_qwen35::LoadedQwenModel;
+
+const PAD: usize = 4;
+const SENTINEL: u32 = 0xa5a55a5a;
+const SPECIAL: [u32; 10] = [
+    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000,
+    0x7fc12345,
+];
+
+fn value(row: usize, channel: usize, at: usize) -> u32 {
+    if at % 16 < SPECIAL.len() {
+        SPECIAL[(at + row + channel) % SPECIAL.len()]
+    } else {
+        0x3f000000
+            | (((at as u32).wrapping_mul(73)
+                ^ ((row as u32 + 1) * 0x10001)
+                ^ ((channel as u32 + 1) * 0x7001))
+                & 0x007fffff)
+    }
+}
+
+fn view(lease: &ResidentLease, words: usize) -> Handle {
+    lease
+        .handle()
+        .offset_start((PAD * 4) as u64)
+        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
+}
+
+fn rows(handles: &[Handle], channels: usize) -> Vec<Row<'_>> {
+    handles
+        .chunks_exact(channels)
+        .map(|group| Row {
+            first: &group[0],
+            second: (channels == 2).then(|| &group[1]),
+        })
+        .collect()
+}
+
+fn refusal(result: Result<PreparedTransport>, text: &str) -> Result<()> {
+    match result {
+        Err(error) if error.to_string().contains(text) => Ok(()),
+        _ => Err(Error::State(format!(
+            "state transport expected prelaunch refusal: {text}"
+        ))),
+    }
+}
+
+fn compare(read: &[Bytes], batch: usize, channels: usize, shape: Shape) -> Result<usize> {
+    let (words, total) = geometry(shape, batch)?;
+    let expected_count = 1 + batch * channels * 3 + channels * 2;
+    if read.len() != expected_count || read[0].len() != 4 || read[0][..] != [0, 0, 0, 0] {
+        return Err(Error::State(
+            "state transport complete status/read coverage".into(),
+        ));
+    }
+    let mut checked = 0;
+    let mut check = |bytes: &Bytes, count: usize, expected: &dyn Fn(usize) -> u32| -> Result<()> {
+        if bytes.len() != (count + 2 * PAD) * 4 {
+            return Err(Error::State(
+                "state transport complete canary readback".into(),
+            ));
+        }
+        for (at, raw) in bytes.chunks_exact(4).enumerate() {
+            let want = if at < PAD || at >= PAD + count {
+                SENTINEL
+            } else {
+                expected(at - PAD)
+            };
+            if raw != want.to_ne_bytes() {
+                return Err(Error::State(format!(
+                    "state transport physical bit/canary difference at{at}"
+                )));
+            }
+            checked += 1;
+        }
+        Ok(())
+    };
+    let mut index = 1;
+    for _private_set in 0..3 {
+        for row in 0..batch {
+            for channel in 0..channels {
+                check(&read[index], words, &|at| value(row, channel, at))?;
+                index += 1;
+            }
+        }
+    }
+    for _packed_set in 0..2 {
+        for channel in 0..channels {
+            check(&read[index], total, &|at| {
+                let (row, local) = match shape {
+                    Shape::Vector { .. } => (at / words, at % words),
+                    Shape::Kv { slab, .. } => (
+                        (at / slab) % batch,
+                        (at / (slab * batch)) * slab + at % slab,
+                    ),
+                };
+                value(row, channel, local)
+            })?;
+            index += 1;
+        }
+    }
+    if index != read.len() {
+        return Err(Error::State(
+            "state transport omitted physical row/channel".into(),
+        ));
+    }
+    Ok(checked)
+}
+
+async fn case(
+    service: &Arc<ResidentService>,
+    batch: usize,
+    shape: Shape,
+) -> Result<(usize, usize, usize)> {
+    let (words, total) = geometry(shape, batch)?;
+    let channels = if matches!(shape, Shape::Kv { .. }) {
+        2
+    } else {
+        1
+    };
+    let private_bytes = (words + 2 * PAD) * 4;
+    let packed_bytes = (total + 2 * PAD) * 4;
+    let bytes = private_bytes * batch * channels * 3 + packed_bytes * channels * 2 + 4;
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: bytes as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    if bytes > service.remaining_capacity()? {
+        return Err(Error::Limit("state transport fixture envelope".into()));
+    }
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(0)?;
+    let status = init.upload_bytes(u32::as_bytes(&[0u32]), BufferRole::Scratch)?;
+    let mut input = Vec::with_capacity(batch * channels);
+    for row in 0..batch {
+        for channel in 0..channels {
+            let mut data = vec![SENTINEL; words + 2 * PAD];
+            for at in 0..words {
+                data[PAD + at] = value(row, channel, at);
+            }
+            input.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
+        }
+    }
+    let mut output = Vec::with_capacity(2 * batch * channels + 2 * channels);
+    for _ in 0..2 * batch * channels {
+        output.push(init.upload_bytes(
+            u32::as_bytes(&vec![SENTINEL; words + 2 * PAD]),
+            BufferRole::Scratch,
+        )?);
+    }
+    for _ in 0..2 * channels {
+        output.push(init.upload_bytes(
+            u32::as_bytes(&vec![SENTINEL; total + 2 * PAD]),
+            BufferRole::Scratch,
+        )?);
+    }
+    let initialized = init.complete_sync().await?;
+    if initialized.outputs().len() != input.len() + output.len() + 1 {
+        return Err(Error::State(
+            "state transport initialization coverage".into(),
+        ));
+    }
+    drop(initialized);
+    let client = service.client();
+    let input_handles = input
+        .iter()
+        .map(|lease| view(lease, words))
+        .collect::<Vec<_>>();
+    let old_private = output[..batch * channels]
+        .iter()
+        .map(|lease| view(lease, words))
+        .collect::<Vec<_>>();
+    let new_private = output[batch * channels..2 * batch * channels]
+        .iter()
+        .map(|lease| view(lease, words))
+        .collect::<Vec<_>>();
+    let packed = output[2 * batch * channels..]
+        .iter()
+        .map(|lease| view(lease, total))
+        .collect::<Vec<_>>();
+    let input_rows = rows(&input_handles, channels);
+    let old_rows = rows(&old_private, channels);
+    let new_rows = rows(&new_private, channels);
+    let old_k = &packed[0];
+    let old_v = (channels == 2).then(|| &packed[1]);
+    let new_k = &packed[channels];
+    let new_v = (channels == 2).then(|| &packed[channels + 1]);
+    let gather =
+        PreparedTransport::prepare(client, &input_rows, new_k, new_v, shape, Direction::Gather)?;
+    let scatter =
+        PreparedTransport::prepare(client, &new_rows, new_k, new_v, shape, Direction::Scatter)?;
+    let stable_bytes = service.resident_bytes()?;
+    let stable_uploads = service.uploaded_value_bytes();
+    let mut foreign = input_handles[0].clone();
+    foreign.service.service = std::any::TypeId::of::<()>();
+    let mut invalid_rows = rows(&input_handles, channels);
+    invalid_rows[0].first = &foreign;
+    refusal(
+        PreparedTransport::prepare(
+            client,
+            &invalid_rows,
+            new_k,
+            new_v,
+            shape,
+            Direction::Gather,
+        ),
+        "service",
+    )?;
+    invalid_rows = rows(&input_handles, channels);
+    invalid_rows[batch - 1].first = invalid_rows[0].first;
+    refusal(
+        PreparedTransport::prepare(
+            client,
+            &invalid_rows,
+            new_k,
+            new_v,
+            shape,
+            Direction::Gather,
+        ),
+        "aliased",
+    )?;
+    invalid_rows = rows(&input_handles, channels);
+    invalid_rows[0].first = new_k; // Geometry is large enough: exercise the alias guard.
+    refusal(
+        PreparedTransport::prepare(
+            client,
+            &invalid_rows,
+            new_k,
+            new_v,
+            shape,
+            Direction::Gather,
+        ),
+        "aliased",
+    )?;
+    let mut bad_end = input_handles[0].clone();
+    bad_end.offset_end = Some(bad_end.size() + 1);
+    invalid_rows = rows(&input_handles, channels);
+    invalid_rows[0].first = &bad_end;
+    refusal(
+        PreparedTransport::prepare(
+            client,
+            &invalid_rows,
+            new_k,
+            new_v,
+            shape,
+            Direction::Gather,
+        ),
+        "end exceeds",
+    )?;
+    let mut bad_start = input_handles[0].clone();
+    bad_start.offset_start = Some(bad_start.size());
+    invalid_rows[0].first = &bad_start;
+    refusal(
+        PreparedTransport::prepare(
+            client,
+            &invalid_rows,
+            new_k,
+            new_v,
+            shape,
+            Direction::Gather,
+        ),
+        "start exceeds",
+    )?;
+    let mut short = input_handles[0].clone();
+    short.offset_end = Some(short.size() - (PAD + words - 1) as u64 * 4);
+    invalid_rows[0].first = &short;
+    refusal(
+        PreparedTransport::prepare(
+            client,
+            &invalid_rows,
+            new_k,
+            new_v,
+            shape,
+            Direction::Gather,
+        ),
+        "alignment/size",
+    )?;
+    if service.resident_bytes()? != stable_bytes
+        || service.uploaded_value_bytes() != stable_uploads
+        || service.pending_events()? != 0
+    {
+        return Err(Error::State(
+            "state transport prelaunch refusal allocated/uploaded/submitted".into(),
+        ));
+    }
+    let mut accesses = input
+        .iter()
+        .cloned()
+        .map(|lease| (lease, DeviceAccess::Read))
+        .collect::<Vec<_>>();
+    accesses.extend(
+        output
+            .iter()
+            .cloned()
+            .map(|lease| (lease, DeviceAccess::Write)),
+    );
+    accesses.push((status.clone(), DeviceAccess::Write));
+    let mut promoted = output.clone();
+    promoted.push(status.clone());
+    let ordered = std::iter::once(&status)
+        .chain(&input)
+        .chain(&output)
+        .collect::<Vec<_>>();
+    let reads = ordered
+        .iter()
+        .map(|lease| lease.readback(0, lease.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let flight = service.submit(0, &accesses)?;
+    let original = super::launch_original(
+        client,
+        &input_rows,
+        old_k,
+        old_v,
+        shape,
+        Direction::Gather,
+        &status.handle(),
+    )?;
+    let native = gather.launch(client)?;
+    let original = original
+        + super::launch_original(
+            client,
+            &old_rows,
+            old_k,
+            old_v,
+            shape,
+            Direction::Scatter,
+            &status.handle(),
+        )?;
+    let native = native + scatter.launch(client)?;
+    let mut checked = 0;
+    let (ready, data) = flight
+        .complete_with_readback(&promoted, &reads, |read| {
+            checked = compare(read, batch, channels, shape)?;
+            Ok(())
+        })
+        .await?;
+    slot.finish()?;
+    if original != 2 * batch
+        || native != 2
+        || service.uploaded_value_bytes() != stable_uploads
+        || ready.outputs().len() != promoted.len()
+        || data.len() != ordered.len()
+    {
+        return Err(Error::State(
+            "state transport physical count/completion/upload coverage".into(),
+        ));
+    }
+    drop(ready);
+    Ok((checked, original, native))
+}
+
+pub(crate) async fn qualify_words(device: Device, path: &Path, device_cap: u64) -> Result<()> {
+    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "state transport physical qualification requires release CUDA".into(),
+        ));
+    }
+    let directory = path
+        .parent()
+        .ok_or_else(|| Error::Malformed("state transport model directory".into()))?;
+    let source = LoadedQwenModel::open(path, &directory.join("config.json"))?;
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("state-bit-transport", "original-per-row-differential")?,
+        256 * 1024 * 1024,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    let mut totals = (0, 0, 0);
+    for batch in [2, 4, 8] {
+        for shape in [
+            Shape::Vector { words: 13 },
+            Shape::Vector { words: 18_432 },
+            Shape::Vector { words: 262_144 },
+            Shape::Kv {
+                slab: 5,
+                capacity: 3,
+            },
+            Shape::Kv {
+                slab: 512,
+                capacity: 512,
+            },
+        ] {
+            let actual = case(&service, batch, shape).await?;
+            totals.0 += actual.0;
+            totals.1 += actual.1;
+            totals.2 += actual.2;
+            if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+                return Err(Error::State(
+                    "state transport fixture retained private leases/events".into(),
+                ));
+            }
+        }
+    }
+    drop(service);
+    let final_budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_budget.reserved != 0 || final_budget.fenced {
+        return Err(Error::State(
+            "state transport final original physical envelope".into(),
+        ));
+    }
+    println!(
+        "CUDA_STATE_BIT_TRANSPORT_PASS B=2,4,8 compared_words={} original_per_row_dispatches={} native_domain_dispatches={} exact_U32_signedzero_subnormal_nonfinite_payload=1 prefix_suffix_canaries=unchanged old_physical_kernels=independent original_lease_Ready_queue=1 warm_uploads=0 final_reserved=0 published_roots=0 fullmodel_graph_worker_HTTP_pressure_abort_speed=separate_gates",
+        totals.0, totals.1, totals.2
+    );
+    Ok(())
+}
+
+#[test]
+#[ignore = "release CUDA original0.8B physical U32 transport; isolated GPU coordinator only"]
+fn original_qwen35_08b_native_state_word_transport() -> Result<()> {
+    let path =
+        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf");
+    futures_lite::future::block_on(qualify_words(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &path,
+        5120 * 1024 * 1024,
+    ))
+}
+
+#[test]
+#[ignore = "release CUDA original2B physical U32 transport; isolated GPU coordinator only"]
+fn original_qwen35_2b_native_state_word_transport() -> Result<()> {
+    let path =
+        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf");
+    futures_lite::future::block_on(qualify_words(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &path,
+        5120 * 1024 * 1024,
+    ))
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification.rs
@@ -0,0 +1,67 @@
+//! Original per-row physical kernels retained as the independent bit oracle.
+use super::*;
+use crate::resident_state_transport::{Direction, Row, Shape};
+
+/// Call only after both new/old complete physical bindings pass preflight.
+/// This is the original row loop and original kernels, never host expected math.
+pub(crate) fn launch_original(
+    client: &Client,
+    rows: &[Row<'_>],
+    dense_first: &Handle,
+    dense_second: Option<&Handle>,
+    shape: Shape,
+    direction: Direction,
+    status: &Handle,
+) -> Result<usize> {
+    let (words, total) = crate::resident_state_transport::geometry(shape, rows.len())?;
+    // Same real views as candidate preflight; no device fallback/hidden upload.
+    PreparedTransport::prepare(client, rows, dense_first, dense_second, shape, direction)?;
+    for (row, private) in rows.iter().enumerate() {
+        // SAFETY: actual preflight covers every original visible range and
+        // the outer original flight retains all physical leases/status.
+        unsafe {
+            match shape {
+                Shape::Vector { .. } => {
+                    let (input, output, input_words, output_words, source, destination) =
+                        if direction == Direction::Scatter {
+                            (dense_first, private.first, total, words, row * words, 0)
+                        } else {
+                            (private.first, dense_first, words, total, 0, row * words)
+                        };
+                    crate::program_kernels::copy_strided_words::launch_unchecked(
+                        client,
+                        grid(words),
+                        threads(),
+                        arg(input, input_words),
+                        arg(output, output_words),
+                        arg(status, 1),
+                        source as u32,
+                        destination as u32,
+                        words as u32,
+                        words as u32,
+                        1,
+                        words,
+                        false,
+                    );
+                }
+                Shape::Kv { slab, capacity } => transport_kv::launch_unchecked(
+                    client,
+                    grid(words),
+                    threads(),
+                    arg(private.first, words),
+                    arg(private.second.expect("checked original V"), words),
+                    arg(dense_first, total),
+                    arg(dense_second.expect("checked original denseV"), total),
+                    row,
+                    slab,
+                    capacity,
+                    rows.len(),
+                    direction == Direction::Scatter,
+                ),
+            }
+        }
+    }
+    Ok(rows.len())
+}
+
+mod physical;
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -7,7 +7,15 @@
 use super::*;
 use crate::activation_quant::{self, PARAMETER_WORDS};
 use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
+use crate::resident_state_transport::{Direction, PreparedTransport, Row, Shape};
 use crate::source_float::{round_source, widen_source};
+
+fn transport_shape(layout: StateLayout) -> Shape {
+    match layout {
+        StateLayout::Vector { elements } => Shape::Vector { words: elements },
+        StateLayout::Kv { slab, capacity } => Shape::Kv { slab, capacity },
+    }
+}
 
 /// One original plan, shared immutable tensors, and a private B-shaped workspace.
 pub(crate) struct BatchProgram {
@@ -36,6 +44,8 @@
 mod graph;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
 pub(crate) use graph::{BatchGraphProfile, BatchGraphReplay};
+#[cfg(all(test, feature = "qualification", feature = "cuda"))]
+mod state_transport_qualification;
 
 impl BatchProgram {
     /// New allocations only: the original source handles remain shared.
@@ -58,6 +68,7 @@
         }
         for state in &budget.states {
             bytes(scale(state.bytes / 4)?, 4)?;
+            crate::resident_state_transport::geometry(transport_shape(state.layout), rows)?;
         }
         budget.source_bytes = 0;
         budget.value_bytes = scale(budget.value_bytes)?;
@@ -101,6 +112,13 @@
             return Err(Error::Unsupported(
                 "native source batch requires CUDA lowering".into(),
             ));
+        }
+        for state in &preflight.states {
+            crate::resident_state_transport::ensure_capability(
+                client,
+                transport_shape(state.layout),
+                rows,
+            )?;
         }
         let mut owned = Vec::new();
         let mut recorded =
@@ -388,6 +406,10 @@
                 ));
             }
         }
+        // Both directions and every domain are checked before the enclosing
+        // original physical submission begins. No allocation/upload occurs.
+        self.prepare_state_transports(client, bindings, false)?;
+        self.prepare_state_transports(client, bindings, true)?;
         Ok(())
     }
 
@@ -470,7 +492,7 @@
         } else {
             self.validate_runtime(client, parameters, bindings)?;
         }
-        let prefix_dispatches = self.launch_inputs(client, parameters, bindings, device_tokens);
+        let prefix_dispatches = self.launch_inputs(client, parameters, bindings, device_tokens)?;
         let body = self.launch_body(client, parameters, output)?;
         self.launch_outputs(client, bindings, output, prefix_dispatches, body)
     }
@@ -481,7 +503,7 @@
         parameters: &[RuntimeParameters],
         bindings: &[Vec<StateBinding>],
         device_tokens: Option<&[DeviceBuffer]>,
-    ) -> usize {
+    ) -> Result<usize> {
         unsafe {
             crate::program_kernels::clear_step_status::launch_unchecked(
                 client,
@@ -549,7 +571,7 @@
                 self.rows,
             );
         }
-        3 + self.transport_state(client, bindings, false)
+        Ok(3 + self.transport_state(client, bindings, false)?)
     }
 
     /// Original nodes and their certifications, in unchanged source order.
@@ -614,7 +636,7 @@
         let mut dispatches = prefix_dispatches + body.dispatches;
         let fused_dispatches = body.fused_dispatches;
         let count = output.nodes(&self.inner.plan).len();
-        dispatches += self.transport_state(client, bindings, true);
+        dispatches += self.transport_state(client, bindings, true)?;
         let mut proposals = Vec::new();
         for row in bindings {
             for version in self.inner.plan.proposed_domains() {
@@ -677,82 +699,121 @@
         })
     }
 
+    fn prepare_state_transports(
+        &self,
+        client: &Client,
+        bindings: &[Vec<StateBinding>],
+        scatter: bool,
+    ) -> Result<Vec<PreparedTransport>> {
+        if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
+            return Err(Error::State("state transport domain/row coverage".into()));
+        }
+        let direction = if scatter {
+            Direction::Scatter
+        } else {
+            Direction::Gather
+        };
+        let mut prepared = Vec::with_capacity(self.packed.len());
+        for packed in &self.packed {
+            let dense = if scatter {
+                &packed.candidate
+            } else {
+                &packed.prior
+            };
+            let private = bindings
+                .iter()
+                .map(|row| {
+                    let binding = row
+                        .iter()
+                        .find(|binding| binding.slot == packed.slot)
+                        .ok_or_else(|| {
+                            Error::State("state transport original domain absent".into())
+                        })?;
+                    Ok(if scatter {
+                        &binding.candidate
+                    } else {
+                        &binding.prior
+                    })
+                })
+                .collect::<Result<Vec<_>>>()?;
+            match dense {
+                StateBuffers::Vector(dense) => {
+                    let words = dense.elements / self.rows;
+                    if words.checked_mul(self.rows) != Some(dense.elements) {
+                        return Err(Error::State("state transport dense vector geometry".into()));
+                    }
+                    let rows = private
+                        .iter()
+                        .map(|value| match value {
+                            StateBuffers::Vector(value) if value.elements == words => Ok(Row {
+                                first: &value.handle,
+                                second: None,
+                            }),
+                            _ => Err(Error::State("state transport vector row changed".into())),
+                        })
+                        .collect::<Result<Vec<_>>>()?;
+                    prepared.push(PreparedTransport::prepare(
+                        client,
+                        &rows,
+                        &dense.handle,
+                        None,
+                        Shape::Vector { words },
+                        direction,
+                    )?);
+                }
+                StateBuffers::Kv { keys, values } => {
+                    let capacity = self.inner.plan.prefix_capacity();
+                    let stride = self
+                        .rows
+                        .checked_mul(capacity)
+                        .filter(|&n| n != 0)
+                        .ok_or_else(|| Error::Limit("state transport KV stride".into()))?;
+                    let slab = keys.elements / stride;
+                    let words = slab
+                        .checked_mul(capacity)
+                        .ok_or_else(|| Error::Limit("state transport KV row words".into()))?;
+                    if slab.checked_mul(stride) != Some(keys.elements)
+                        || keys.elements != values.elements
+                    {
+                        return Err(Error::State("state transport dense KV geometry".into()));
+                    }
+                    let rows = private
+                        .iter()
+                        .map(|value| match value {
+                            StateBuffers::Kv { keys, values }
+                                if keys.elements == words && values.elements == words =>
+                            {
+                                Ok(Row {
+                                    first: &keys.handle,
+                                    second: Some(&values.handle),
+                                })
+                            }
+                            _ => Err(Error::State("state transport KV row changed".into())),
+                        })
+                        .collect::<Result<Vec<_>>>()?;
+                    prepared.push(PreparedTransport::prepare(
+                        client,
+                        &rows,
+                        &keys.handle,
+                        Some(&values.handle),
+                        Shape::Kv { slab, capacity },
+                        direction,
+                    )?);
+                }
+            }
+        }
+        Ok(prepared)
+    }
+
     fn transport_state(
         &self,
         client: &Client,
         bindings: &[Vec<StateBinding>],
         scatter: bool,
-    ) -> usize {
-        let mut count = 0;
-        for (row, bindings) in bindings.iter().enumerate() {
-            for packed in &self.packed {
-                let binding = bindings
-                    .iter()
-                    .find(|b| b.slot == packed.slot)
-                    .expect("validated domain");
-                let (private, dense) = if scatter {
-                    (&binding.candidate, &packed.candidate)
-                } else {
-                    (&binding.prior, &packed.prior)
-                };
-                match (private, dense) {
-                    (StateBuffers::Vector(private), StateBuffers::Vector(dense)) => {
-                        if scatter {
-                            count += self.copy_dense(
-                                client,
-                                &dense.handle,
-                                dense.elements,
-                                row * private.elements,
-                                &private.handle,
-                                private.elements,
-                                0,
-                                private.elements,
-                            );
-                        } else {
-                            count += self.copy_dense(
-                                client,
-                                &private.handle,
-                                private.elements,
-                                0,
-                                &dense.handle,
-                                dense.elements,
-                                row * private.elements,
-                                private.elements,
-                            );
-                        }
-                    }
-                    (
-                        StateBuffers::Kv { keys, values },
-                        StateBuffers::Kv {
-                            keys: dense_k,
-                            values: dense_v,
-                        },
-                    ) => {
-                        let capacity = self.inner.plan.prefix_capacity();
-                        let slab = keys.elements / capacity;
-                        unsafe {
-                            transport_kv::launch_unchecked(
-                                client,
-                                grid(keys.elements),
-                                threads(),
-                                arg(&keys.handle, keys.elements),
-                                arg(&values.handle, values.elements),
-                                arg(&dense_k.handle, dense_k.elements),
-                                arg(&dense_v.handle, dense_v.elements),
-                                row,
-                                slab,
-                                capacity,
-                                self.rows,
-                                scatter,
-                            );
-                        }
-                        count += 1;
-                    }
-                    _ => unreachable!("validated state layout"),
-                }
-            }
-        }
-        count
+    ) -> Result<usize> {
+        self.prepare_state_transports(client, bindings, scatter)?
+            .iter()
+            .try_fold(0, |count, transport| Ok(count + transport.launch(client)?))
     }
 
     fn certify(&self, client: &Client, source: &Handle, elements: usize) -> usize {
@@ -1638,6 +1699,7 @@
 }
 
 /// Private [T,H,D] handles ↔ physically packed [T,B,H,D] workspace.
+#[cfg(all(test, feature = "qualification", feature = "cuda"))]
 #[cube(launch_unchecked)]
 fn transport_kv(
     private_k: &mut [u32],
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_state_transport.rs
@@ -0,0 +1,562 @@
+//! Bounded U32 transport across independent original state leases.
+//!
+//! No source math, allocation, upload, authority, Ready or publication. The
+//! original caller validates source/generation and retains every physical lease
+//! before launch, including captured-graph IO and unknown-completion quarantine.
+use cubecl::{client::Client, prelude::*, server::Handle};
+use cubecl_common::device::ServiceId;
+use std::ops::Range;
+use zerogravity_core::{Error, Result};
+
+const MAX_ROW_WORDS: usize = 1_000_000;
+const MAX_TOTAL_WORDS: usize = 8_000_000;
+const LANES: u32 = 32;
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub(crate) enum Shape {
+    Vector { words: usize },
+    Kv { slab: usize, capacity: usize },
+}
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub(crate) enum Direction {
+    Gather,
+    Scatter,
+}
+
+pub(crate) fn geometry(shape: Shape, rows: usize) -> Result<(usize, usize)> {
+    if !matches!(rows, 2 | 4 | 8) {
+        return Err(Error::Unsupported("state transport native rows".into()));
+    }
+    let words = match shape {
+        Shape::Vector { words } => words,
+        Shape::Kv { slab, capacity } => {
+            if slab == 0 || capacity == 0 || capacity > 512 {
+                return Err(Error::Limit("state transport KV geometry".into()));
+            }
+            slab.checked_mul(capacity)
+                .ok_or_else(|| Error::Limit("state transport KV word overflow".into()))?
+        }
+    };
+    let total = words
+        .checked_mul(rows)
+        .filter(|&n| n <= MAX_TOTAL_WORDS && u32::try_from(n).is_ok())
+        .ok_or_else(|| Error::Limit("state transport aggregate word domain".into()))?;
+    if words == 0 || words > MAX_ROW_WORDS {
+        return Err(Error::Limit("state transport per-row word domain".into()));
+    }
+    Ok((words, total))
+}
+
+pub(crate) fn ensure_capability(client: &Client, shape: Shape, rows: usize) -> Result<()> {
+    if !crate::activation_quant_fused::is_cuda_service(client) {
+        return Err(Error::Unsupported(
+            "native state transport CUDA service".into(),
+        ));
+    }
+    let (_, total) = geometry(shape, rows)?;
+    let hardware = &client.properties().hardware;
+    let bindings = if matches!(shape, Shape::Kv { .. }) {
+        18
+    } else {
+        9
+    };
+    let cubes = (total as u32).div_ceil(LANES);
+    if hardware.max_bindings < bindings
+        || hardware.max_units_per_cube < LANES
+        || hardware.max_cube_dim.0 < LANES
+        || hardware.max_cube_dim.1 == 0
+        || hardware.max_cube_dim.2 == 0
+        || hardware.max_cube_count.0 < cubes
+        || hardware.max_cube_count.1 == 0
+        || hardware.max_cube_count.2 == 0
+    {
+        return Err(Error::Unsupported(
+            "state transport CUDA binding/grid limits".into(),
+        ));
+    }
+    Ok(())
+}
+
+fn checked_span(size: u64, start: u64, trim: u64, words: usize) -> Result<Range<u64>> {
+    let end = size
+        .checked_sub(trim)
+        .ok_or_else(|| Error::State("state transport end exceeds allocation".into()))?;
+    let used = end
+        .checked_sub(start)
+        .ok_or_else(|| Error::State("state transport start exceeds end".into()))?;
+    let needed = u64::try_from(words)
+        .ok()
+        .and_then(|n| n.checked_mul(4))
+        .ok_or_else(|| Error::Limit("state transport byte domain".into()))?;
+    if needed == 0 || !start.is_multiple_of(4) || used < needed {
+        return Err(Error::State("state transport buffer alignment/size".into()));
+    }
+    Ok(start..end)
+}
+
+fn buffer(client: &Client, handle: &Handle, words: usize) -> Result<Range<u64>> {
+    if handle.service != client.service_id() {
+        return Err(Error::State(
+            "state transport foreign physical service".into(),
+        ));
+    }
+    checked_span(
+        handle.size(),
+        handle.offset_start.unwrap_or(0),
+        handle.offset_end.unwrap_or(0),
+        words,
+    )
+}
+
+/// Original handles only. `second` exists exactly for an original KV domain.
+pub(crate) struct Row<'a> {
+    pub first: &'a Handle,
+    pub second: Option<&'a Handle>,
+}
+
+/// Checked metadata/handle clones, never a new allocation or lease capability.
+pub(crate) struct PreparedTransport {
+    service: ServiceId,
+    first: Vec<Handle>,
+    second: Vec<Handle>,
+    dense_first: Handle,
+    dense_second: Option<Handle>,
+    shape: Shape,
+    direction: Direction,
+    words: usize,
+    total: usize,
+}
+
+impl PreparedTransport {
+    pub(crate) fn prepare(
+        client: &Client,
+        rows: &[Row<'_>],
+        dense_first: &Handle,
+        dense_second: Option<&Handle>,
+        shape: Shape,
+        direction: Direction,
+    ) -> Result<Self> {
+        ensure_capability(client, shape, rows.len())?;
+        let (words, total) = geometry(shape, rows.len())?;
+        let kv = matches!(shape, Shape::Kv { .. });
+        if dense_second.is_some() != kv || rows.iter().any(|row| row.second.is_some() != kv) {
+            return Err(Error::Malformed(
+                "state transport domain buffer coverage".into(),
+            ));
+        }
+        let mut ranges = Vec::with_capacity(if kv {
+            2 * rows.len() + 2
+        } else {
+            rows.len() + 1
+        });
+        for row in rows {
+            ranges.push((row.first, buffer(client, row.first, words)?));
+            if let Some(second) = row.second {
+                ranges.push((second, buffer(client, second, words)?));
+            }
+        }
+        ranges.push((dense_first, buffer(client, dense_first, total)?));
+        if let Some(second) = dense_second {
+            ranges.push((second, buffer(client, second, total)?));
+        }
+        let ranges = ranges
+            .into_iter()
+            .map(|(handle, range)| (handle.clone().binding(), range))
+            .collect::<Vec<_>>();
+        for (i, (a, ar)) in ranges.iter().enumerate() {
+            for (b, br) in &ranges[..i] {
+                if a.memory.id() == b.memory.id() && ar.start < br.end && br.start < ar.end {
+                    return Err(Error::State(
+                        "state transport aliased independent buffers".into(),
+                    ));
+                }
+            }
+        }
+        Ok(Self {
+            service: client.service_id(),
+            first: rows.iter().map(|row| row.first.clone()).collect(),
+            second: rows.iter().filter_map(|row| row.second.cloned()).collect(),
+            dense_first: dense_first.clone(),
+            dense_second: dense_second.cloned(),
+            shape,
+            direction,
+            words,
+            total,
+        })
+    }
+
+    /// One original-client launch per domain/direction, not per row. The
+    /// enclosing physical submission already retains every original lease.
+    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
+        if client.service_id() != self.service {
+            return Err(Error::State(
+                "prepared state transport service changed".into(),
+            ));
+        }
+        // ABI-only inactive arguments reuse row zero. Compile-time row bounds
+        // make them unreachable; no session, model row or state is padded.
+        let first = |row: usize| {
+            BufferArg::from_raw_parts(
+                self.first.get(row).unwrap_or(&self.first[0]).clone(),
+                self.words,
+            )
+        };
+        let second = |row: usize| {
+            BufferArg::from_raw_parts(
+                self.second.get(row).unwrap_or(&self.second[0]).clone(),
+                self.words,
+            )
+        };
+        let dense_first = || BufferArg::from_raw_parts(self.dense_first.clone(), self.total);
+        let dense_second =
+            || BufferArg::from_raw_parts(self.dense_second.as_ref().unwrap().clone(), self.total);
+        let grid = CubeCount::Static((self.total as u32).div_ceil(LANES), 1, 1);
+        let threads = CubeDim::new_1d(LANES);
+        // SAFETY: checked original service, disjoint visible ranges, exact
+        // B×N layouts and all shader/binding/grid bounds were fixed prelaunch.
+        unsafe {
+            match (self.shape, self.direction) {
+                (Shape::Vector { .. }, Direction::Gather) => gather_vector::launch_unchecked(
+                    client,
+                    grid,
+                    threads,
+                    first(0),
+                    first(1),
+                    first(2),
+                    first(3),
+                    first(4),
+                    first(5),
+                    first(6),
+                    first(7),
+                    dense_first(),
+                    self.words,
+                    self.first.len(),
+                ),
+                (Shape::Vector { .. }, Direction::Scatter) => scatter_vector::launch_unchecked(
+                    client,
+                    grid,
+                    threads,
+                    first(0),
+                    first(1),
+                    first(2),
+                    first(3),
+                    first(4),
+                    first(5),
+                    first(6),
+                    first(7),
+                    dense_first(),
+                    self.words,
+                    self.first.len(),
+                ),
+                (Shape::Kv { slab, capacity }, Direction::Gather) => gather_kv::launch_unchecked(
+                    client,
+                    grid,
+                    threads,
+                    first(0),
+                    first(1),
+                    first(2),
+                    first(3),
+                    first(4),
+                    first(5),
+                    first(6),
+                    first(7),
+                    second(0),
+                    second(1),
+                    second(2),
+                    second(3),
+                    second(4),
+                    second(5),
+                    second(6),
+                    second(7),
+                    dense_first(),
+                    dense_second(),
+                    slab,
+                    capacity,
+                    self.first.len(),
+                ),
+                (Shape::Kv { slab, capacity }, Direction::Scatter) => scatter_kv::launch_unchecked(
+                    client,
+                    grid,
+                    threads,
+                    first(0),
+                    first(1),
+                    first(2),
+                    first(3),
+                    first(4),
+                    first(5),
+                    first(6),
+                    first(7),
+                    second(0),
+                    second(1),
+                    second(2),
+                    second(3),
+                    second(4),
+                    second(5),
+                    second(6),
+                    second(7),
+                    dense_first(),
+                    dense_second(),
+                    slab,
+                    capacity,
+                    self.first.len(),
+                ),
+            }
+        }
+        Ok(1)
+    }
+}
+
+#[cube(launch_unchecked)]
+fn gather_vector(
+    p0: &[u32],
+    p1: &[u32],
+    p2: &[u32],
+    p3: &[u32],
+    p4: &[u32],
+    p5: &[u32],
+    p6: &[u32],
+    p7: &[u32],
+    dense: &mut [u32],
+    #[comptime] words: usize,
+    #[comptime] rows: usize,
+) {
+    let at = ABSOLUTE_POS;
+    if at < rows * words {
+        let row = at / words;
+        let local = at % words;
+        let packed = at;
+        if row == 0usize {
+            dense[packed] = p0[local];
+        } else if row == 1usize {
+            dense[packed] = p1[local];
+        } else if row == 2usize {
+            dense[packed] = p2[local];
+        } else if row == 3usize {
+            dense[packed] = p3[local];
+        } else if row == 4usize {
+            dense[packed] = p4[local];
+        } else if row == 5usize {
+            dense[packed] = p5[local];
+        } else if row == 6usize {
+            dense[packed] = p6[local];
+        } else if row == 7usize {
+            dense[packed] = p7[local];
+        }
+    }
+}
+
+#[cube(launch_unchecked)]
+fn scatter_vector(
+    p0: &mut [u32],
+    p1: &mut [u32],
+    p2: &mut [u32],
+    p3: &mut [u32],
+    p4: &mut [u32],
+    p5: &mut [u32],
+    p6: &mut [u32],
+    p7: &mut [u32],
+    dense: &[u32],
+    #[comptime] words: usize,
+    #[comptime] rows: usize,
+) {
+    let at = ABSOLUTE_POS;
+    if at < rows * words {
+        let row = at / words;
+        let local = at % words;
+        let packed = at;
+        if row == 0usize {
+            p0[local] = dense[packed];
+        } else if row == 1usize {
+            p1[local] = dense[packed];
+        } else if row == 2usize {
+            p2[local] = dense[packed];
+        } else if row == 3usize {
+            p3[local] = dense[packed];
+        } else if row == 4usize {
+            p4[local] = dense[packed];
+        } else if row == 5usize {
+            p5[local] = dense[packed];
+        } else if row == 6usize {
+            p6[local] = dense[packed];
+        } else if row == 7usize {
+            p7[local] = dense[packed];
+        }
+    }
+}
+
+#[cube(launch_unchecked)]
+fn gather_kv(
+    p0: &[u32],
+    p1: &[u32],
+    p2: &[u32],
+    p3: &[u32],
+    p4: &[u32],
+    p5: &[u32],
+    p6: &[u32],
+    p7: &[u32],
+    v0: &[u32],
+    v1: &[u32],
+    v2: &[u32],
+    v3: &[u32],
+    v4: &[u32],
+    v5: &[u32],
+    v6: &[u32],
+    v7: &[u32],
+    dense: &mut [u32],
+    dense_v: &mut [u32],
+    #[comptime] slab: usize,
+    #[comptime] capacity: usize,
+    #[comptime] rows: usize,
+) {
+    let at = ABSOLUTE_POS;
+    let words = slab * capacity;
+    if at < rows * words {
+        let row = at / words;
+        let local = at % words;
+        let packed = ((local / slab) * rows + row) * slab + local % slab;
+        if row == 0usize {
+            dense[packed] = p0[local];
+            dense_v[packed] = v0[local];
+        } else if row == 1usize {
+            dense[packed] = p1[local];
+            dense_v[packed] = v1[local];
+        } else if row == 2usize {
+            dense[packed] = p2[local];
+            dense_v[packed] = v2[local];
+        } else if row == 3usize {
+            dense[packed] = p3[local];
+            dense_v[packed] = v3[local];
+        } else if row == 4usize {
+            dense[packed] = p4[local];
+            dense_v[packed] = v4[local];
+        } else if row == 5usize {
+            dense[packed] = p5[local];
+            dense_v[packed] = v5[local];
+        } else if row == 6usize {
+            dense[packed] = p6[local];
+            dense_v[packed] = v6[local];
+        } else if row == 7usize {
+            dense[packed] = p7[local];
+            dense_v[packed] = v7[local];
+        }
+    }
+}
+
+#[cube(launch_unchecked)]
+fn scatter_kv(
+    p0: &mut [u32],
+    p1: &mut [u32],
+    p2: &mut [u32],
+    p3: &mut [u32],
+    p4: &mut [u32],
+    p5: &mut [u32],
+    p6: &mut [u32],
+    p7: &mut [u32],
+    v0: &mut [u32],
+    v1: &mut [u32],
+    v2: &mut [u32],
+    v3: &mut [u32],
+    v4: &mut [u32],
+    v5: &mut [u32],
+    v6: &mut [u32],
+    v7: &mut [u32],
+    dense: &[u32],
+    dense_v: &[u32],
+    #[comptime] slab: usize,
+    #[comptime] capacity: usize,
+    #[comptime] rows: usize,
+) {
+    let at = ABSOLUTE_POS;
+    let words = slab * capacity;
+    if at < rows * words {
+        let row = at / words;
+        let local = at % words;
+        let packed = ((local / slab) * rows + row) * slab + local % slab;
+        if row == 0usize {
+            p0[local] = dense[packed];
+            v0[local] = dense_v[packed];
+        } else if row == 1usize {
+            p1[local] = dense[packed];
+            v1[local] = dense_v[packed];
+        } else if row == 2usize {
+            p2[local] = dense[packed];
+            v2[local] = dense_v[packed];
+        } else if row == 3usize {
+            p3[local] = dense[packed];
+            v3[local] = dense_v[packed];
+        } else if row == 4usize {
+            p4[local] = dense[packed];
+            v4[local] = dense_v[packed];
+        } else if row == 5usize {
+            p5[local] = dense[packed];
+            v5[local] = dense_v[packed];
+        } else if row == 6usize {
+            p6[local] = dense[packed];
+            v6[local] = dense_v[packed];
+        } else if row == 7usize {
+            p7[local] = dense[packed];
+            v7[local] = dense_v[packed];
+        }
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn explicit_native_domain_includes_real_8x262144_state_words() {
+        assert_eq!(
+            geometry(Shape::Vector { words: 262_144 }, 8).unwrap(),
+            (262_144, 2_097_152)
+        );
+        assert_eq!(
+            geometry(
+                Shape::Kv {
+                    slab: 512,
+                    capacity: 512
+                },
+                8
+            )
+            .unwrap(),
+            (262_144, 2_097_152)
+        );
+        assert_eq!(
+            geometry(
+                Shape::Vector {
+                    words: MAX_ROW_WORDS
+                },
+                8
+            )
+            .unwrap()
+            .1,
+            MAX_TOTAL_WORDS
+        );
+        for rows in [0, 1, 3, 5, 16, usize::MAX] {
+            assert!(geometry(Shape::Vector { words: 1 }, rows).is_err());
+        }
+        for words in [0, MAX_ROW_WORDS + 1, usize::MAX] {
+            assert!(geometry(Shape::Vector { words }, 8).is_err());
+        }
+        for (slab, capacity) in [(0, 512), (512, 0), (1, 513), (usize::MAX, 512)] {
+            assert!(geometry(Shape::Kv { slab, capacity }, 8).is_err());
+        }
+    }
+
+    #[test]
+    fn checked_original_canary_views_refuse_underflow_and_wrapped_offsets() {
+        assert_eq!(checked_span(64, 16, 16, 8).unwrap(), 16..48);
+        for (size, start, trim, words) in [
+            (64, 0, 65, 1),
+            (64, 49, 16, 1),
+            (64, u64::MAX, u64::MAX, 1),
+            (64, 1, 0, 1),
+            (64, 16, 16, 9),
+            (64, 0, 0, 0),
+            (64, 0, 0, usize::MAX),
+        ] {
+            assert!(checked_span(size, start, trim, words).is_err());
+        }
+    }
+}

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/ROOT_WIRING.patch

Original UTF-8 bytes: 245; SHA256 `765c8970fafd88c545812024d7a8d7d898dde2c8bd591a511fa87d48132c417d`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -72,6 +72,7 @@
 mod resident_leases;
 mod resident_program;
 mod resident_provider;
+mod resident_state_transport;
 mod rms_norm;
 mod rope_angles;
 mod rope_kv;

````

