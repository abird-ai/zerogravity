# Draft Patches — volume 29

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/frozen-prepared/NEW_DELTA.patch

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

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/frozen-prepared/ROOT_WIRING.patch

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

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/root-compile-fix-v2/ROOT_V2.review.patch

Original UTF-8 bytes: 2602; SHA256 `820a2f81de216db45d9ec62bc1799ddb08efeae1c70e9ea5244dcb953e813d2e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- original/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification/physical.rs
+++ ROOTV2/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification/physical.rs
@@ -1,12 +1,13 @@
 //! Physical original-per-row vs native bit copy, with original leases/Ready.
 use crate::resident_leases::{ResidentLease, ResidentService};
 use crate::resident_state_transport::*;
-use cubecl::{Device, bytes::Bytes};
+use cubecl::{CubeElement, Device, bytes::Bytes, server::Handle};
 use std::{path::Path, sync::Arc};
 use zerogravity_core::device_events::{BufferRole, DeviceAccess};
 use zerogravity_core::resource_arbiter::{
     AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
 };
+use zerogravity_core::{Error, Result};
 use zerogravity_model_qwen35::LoadedQwenModel;
 
 const PAD: usize = 4;
--- original/crates/zerogravity-gpu/src/resident_state_transport.rs
+++ ROOTV2/crates/zerogravity-gpu/src/resident_state_transport.rs
@@ -196,21 +196,26 @@
         }
         // ABI-only inactive arguments reuse row zero. Compile-time row bounds
         // make them unreachable; no session, model row or state is padded.
-        let first = |row: usize| {
+        // SAFETY: prepare checked every original handle and visible word span.
+        let first = |row: usize| unsafe {
             BufferArg::from_raw_parts(
                 self.first.get(row).unwrap_or(&self.first[0]).clone(),
                 self.words,
             )
         };
-        let second = |row: usize| {
+        // SAFETY: only KV branches call this with the checked second handles.
+        let second = |row: usize| unsafe {
             BufferArg::from_raw_parts(
                 self.second.get(row).unwrap_or(&self.second[0]).clone(),
                 self.words,
             )
         };
-        let dense_first = || BufferArg::from_raw_parts(self.dense_first.clone(), self.total);
-        let dense_second =
-            || BufferArg::from_raw_parts(self.dense_second.as_ref().unwrap().clone(), self.total);
+        // SAFETY: the complete dense spans were checked against B times N.
+        let dense_first =
+            || unsafe { BufferArg::from_raw_parts(self.dense_first.clone(), self.total) };
+        let dense_second = || unsafe {
+            BufferArg::from_raw_parts(self.dense_second.as_ref().unwrap().clone(), self.total)
+        };
         let grid = CubeCount::Static((self.total as u32).div_ceil(LANES), 1, 1);
         let threads = CubeDim::new_1d(LANES);
         // SAFETY: checked original service, disjoint visible ranges, exact

````

## Source: .agents/runs/serving-and-speculation-state-scatter-certification-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 36262; SHA256 `091628b52766e4a1f1975a71662c1d336c70e66c2b933f49bba95b2fc02fa40a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -7,7 +7,22 @@
 use super::*;
 use crate::activation_quant::{self, PARAMETER_WORDS};
 use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
-use crate::resident_state_transport::{Direction, PreparedTransport, Row, Shape};
+use crate::resident_state_transport::{
+    Direction, PreparedCertifiedScatter, PreparedTransport, Row, Shape,
+};
+
+enum PreparedStateTransport {
+    Opaque(PreparedTransport),
+    Certified(PreparedCertifiedScatter),
+}
+impl PreparedStateTransport {
+    fn launch(&self, client: &Client) -> Result<usize> {
+        match self {
+            Self::Opaque(transport) => transport.launch(client),
+            Self::Certified(transport) => transport.launch(client),
+        }
+    }
+}
 use crate::source_float::{round_source, widen_source};
 
 fn transport_shape(layout: StateLayout) -> Shape {
@@ -114,7 +129,7 @@
             ));
         }
         for state in &preflight.states {
-            crate::resident_state_transport::ensure_capability(
+            crate::resident_state_transport::ensure_certified_capability(
                 client,
                 transport_shape(state.layout),
                 rows,
@@ -644,9 +659,8 @@
                     .iter()
                     .find(|b| b.slot == version.slot)
                     .expect("validated domain");
-                for value in binding.candidate.buffers() {
-                    dispatches += self.certify(client, &value.handle, value.elements);
-                }
+                // Typed scatter copied and certified EVERY original candidate
+                // word in the same launches; keep the exact all-domain proposal.
                 proposals.push((*version, binding.candidate.clone()));
             }
         }
@@ -704,15 +718,10 @@
         client: &Client,
         bindings: &[Vec<StateBinding>],
         scatter: bool,
-    ) -> Result<Vec<PreparedTransport>> {
+    ) -> Result<Vec<PreparedStateTransport>> {
         if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
             return Err(Error::State("state transport domain/row coverage".into()));
         }
-        let direction = if scatter {
-            Direction::Scatter
-        } else {
-            Direction::Gather
-        };
         let mut prepared = Vec::with_capacity(self.packed.len());
         for packed in &self.packed {
             let dense = if scatter {
@@ -752,14 +761,25 @@
                             _ => Err(Error::State("state transport vector row changed".into())),
                         })
                         .collect::<Result<Vec<_>>>()?;
-                    prepared.push(PreparedTransport::prepare(
-                        client,
-                        &rows,
-                        &dense.handle,
-                        None,
-                        Shape::Vector { words },
-                        direction,
-                    )?);
+                    prepared.push(if scatter {
+                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
+                            client,
+                            &rows,
+                            &dense.handle,
+                            None,
+                            Shape::Vector { words },
+                            &self.inner.status,
+                        )?)
+                    } else {
+                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
+                            client,
+                            &rows,
+                            &dense.handle,
+                            None,
+                            Shape::Vector { words },
+                            Direction::Gather,
+                        )?)
+                    });
                 }
                 StateBuffers::Kv { keys, values } => {
                     let capacity = self.inner.plan.prefix_capacity();
@@ -791,14 +811,25 @@
                             _ => Err(Error::State("state transport KV row changed".into())),
                         })
                         .collect::<Result<Vec<_>>>()?;
-                    prepared.push(PreparedTransport::prepare(
-                        client,
-                        &rows,
-                        &keys.handle,
-                        Some(&values.handle),
-                        Shape::Kv { slab, capacity },
-                        direction,
-                    )?);
+                    prepared.push(if scatter {
+                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
+                            client,
+                            &rows,
+                            &keys.handle,
+                            Some(&values.handle),
+                            Shape::Kv { slab, capacity },
+                            &self.inner.status,
+                        )?)
+                    } else {
+                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
+                            client,
+                            &rows,
+                            &keys.handle,
+                            Some(&values.handle),
+                            Shape::Kv { slab, capacity },
+                            Direction::Gather,
+                        )?)
+                    });
                 }
             }
         }
--- a/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification.rs
@@ -65,3 +65,6 @@
 }
 
 mod physical;
+
+// Typed certification has a separate differential/status/unknown qualifier.
+mod certified;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification/certified.rs
@@ -0,0 +1,606 @@
+//! ROOT-only real CUDA differential: original copy +original finite certificate.
+//! Synthetic state words only; trained all42/fullV/worker gates are separate.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentService};
+use crate::resident_state_transport::{PreparedCertifiedScatter, geometry};
+use cubecl::{CubeElement, Device, bytes::Bytes};
+use std::{path::Path, sync::Arc};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+use zerogravity_model_qwen35::LoadedQwenModel;
+
+const PAD: usize = 4;
+const CANARY: u32 = 0xa5a55a5a;
+const FINITE: [u32; 8] = [
+    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
+];
+const NONFINITE: [u32; 4] = [0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345];
+
+fn word(
+    row: usize,
+    channel: usize,
+    at: usize,
+    words: usize,
+    pattern: usize,
+    channels: usize,
+) -> u32 {
+    if pattern >= 2 && row * channels + channel == pattern - 2 && at + 1 == words {
+        NONFINITE[(pattern - 2) % NONFINITE.len()]
+    } else if at % 16 < FINITE.len() {
+        FINITE[(at + row + channel) % FINITE.len()]
+    } else {
+        0x3f000000 | ((at as u32).wrapping_mul(73) & 0x007fffff)
+    }
+}
+fn visible(lease: &ResidentLease, words: usize) -> Handle {
+    lease
+        .handle()
+        .offset_start((PAD * 4) as u64)
+        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
+}
+fn rows(handles: &[Handle], channels: usize) -> Vec<Row<'_>> {
+    handles
+        .chunks_exact(channels)
+        .map(|h| Row {
+            first: &h[0],
+            second: (channels == 2).then(|| &h[1]),
+        })
+        .collect()
+}
+fn refuses(value: Result<PreparedCertifiedScatter>, text: &str) -> Result<()> {
+    if !matches!(value, Err(ref e) if e.to_string().contains(text)) {
+        return Err(Error::State(format!(
+            "certified scatter expected cold refusal: {text}"
+        )));
+    }
+    Ok(())
+}
+fn check_words(read: &Bytes, words: usize, value: impl Fn(usize) -> u32) -> Result<usize> {
+    if read.len() != (words + 2 * PAD) * 4 {
+        return Err(Error::State(
+            "certified scatter complete canary byte count".into(),
+        ));
+    }
+    for (at, raw) in read.chunks_exact(4).enumerate() {
+        let expected = if at < PAD || at >= PAD + words {
+            CANARY
+        } else {
+            value(at - PAD)
+        };
+        if raw != expected.to_ne_bytes() {
+            return Err(Error::State(format!(
+                "certified scatter raw bit/canary mismatch at{at}"
+            )));
+        }
+    }
+    Ok(words + 2 * PAD)
+}
+
+#[derive(Clone, Copy, PartialEq, Eq)]
+enum Stop {
+    None,
+    Abandon,
+    RejectStatus,
+}
+
+async fn case(
+    service: &Arc<ResidentService>,
+    batch: usize,
+    shape: Shape,
+    pattern: usize,
+    stop: Stop,
+) -> Result<(usize, usize, usize)> {
+    let (words, total) = geometry(shape, batch)?;
+    let channels = 1 + usize::from(matches!(shape, Shape::Kv { .. }));
+    if pattern >= batch * channels + 2 {
+        return Err(Error::Malformed("certified scatter fixture pattern".into()));
+    }
+    let seed = if pattern == 0 {
+        0
+    } else {
+        crate::program_kernels::VIEW_OUT_OF_BOUNDS | crate::program_kernels::INVALID_REDUCTION
+    };
+    let expected_status = seed
+        | if pattern >= 2 {
+            crate::program_kernels::NONFINITE_VALUE
+        } else {
+            0
+        };
+    let bytes = (words + 2 * PAD) * 4 * batch * channels * 3
+        + (total + 2 * PAD) * 4 * channels
+        + (1 + 2 * PAD) * 4 * 2;
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
+        return Err(Error::Limit(
+            "certified scatter fixture physical envelope".into(),
+        ));
+    }
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(0)?;
+    let mut input = Vec::new();
+    for row in 0..batch {
+        for channel in 0..channels {
+            let mut data = vec![CANARY; words + 2 * PAD];
+            for (at, value) in data[PAD..PAD + words].iter_mut().enumerate() {
+                *value = word(row, channel, at, words, pattern, channels);
+            }
+            input.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
+        }
+    }
+    let mut output = Vec::new();
+    for _ in 0..2 * batch * channels {
+        output.push(init.upload_bytes(
+            u32::as_bytes(&vec![CANARY; words + 2 * PAD]),
+            BufferRole::Scratch,
+        )?);
+    }
+    for _ in 0..channels {
+        output.push(init.upload_bytes(
+            u32::as_bytes(&vec![CANARY; total + 2 * PAD]),
+            BufferRole::Scratch,
+        )?);
+    }
+    let mut status = Vec::new();
+    for _ in 0..2 {
+        let mut data = [CANARY; 1 + 2 * PAD];
+        data[PAD] = seed;
+        status.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
+    }
+    let initialized = init.complete_sync().await?;
+    if initialized.outputs().len() != input.len() + output.len() + status.len() {
+        return Err(Error::State(
+            "certified scatter initialized lease coverage".into(),
+        ));
+    }
+    drop(initialized);
+    let client = service.client();
+    let inputs = input.iter().map(|l| visible(l, words)).collect::<Vec<_>>();
+    let old_private = output[..batch * channels]
+        .iter()
+        .map(|l| visible(l, words))
+        .collect::<Vec<_>>();
+    let new_private = output[batch * channels..2 * batch * channels]
+        .iter()
+        .map(|l| visible(l, words))
+        .collect::<Vec<_>>();
+    let dense = output[2 * batch * channels..]
+        .iter()
+        .map(|l| visible(l, total))
+        .collect::<Vec<_>>();
+    let statuses = status.iter().map(|l| visible(l, 1)).collect::<Vec<_>>();
+    let input_rows = rows(&inputs, channels);
+    let old_rows = rows(&old_private, channels);
+    let new_rows = rows(&new_private, channels);
+    let dense_v = (channels == 2).then(|| &dense[1]);
+    // BOTH complete old/new span+status bindings pass before the original flight.
+    PreparedCertifiedScatter::prepare(client, &old_rows, &dense[0], dense_v, shape, &statuses[0])?;
+    let prepared = PreparedCertifiedScatter::prepare(
+        client,
+        &new_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        &statuses[1],
+    )?;
+    PreparedTransport::prepare(
+        client,
+        &input_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        Direction::Gather,
+    )?;
+    let bytes_before = service.resident_bytes()?;
+    let uploads_before = service.uploaded_value_bytes();
+    let mut foreign = statuses[1].clone();
+    foreign.service.service = std::any::TypeId::of::<()>();
+    refuses(
+        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &foreign),
+        "service",
+    )?;
+    refuses(
+        PreparedCertifiedScatter::prepare(
+            client,
+            &new_rows,
+            &dense[0],
+            dense_v,
+            shape,
+            &new_private[0],
+        ),
+        "aliases",
+    )?;
+    refuses(
+        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &dense[0]),
+        "aliases",
+    )?;
+    let mut short = statuses[1].clone();
+    short.offset_end = Some(short.size() - (PAD * 4) as u64);
+    refuses(
+        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &short),
+        "alignment/size",
+    )?;
+    let mut misaligned = statuses[1].clone();
+    misaligned.offset_start = Some((PAD * 4 + 1) as u64);
+    refuses(
+        PreparedCertifiedScatter::prepare(
+            client,
+            &new_rows,
+            &dense[0],
+            dense_v,
+            shape,
+            &misaligned,
+        ),
+        "alignment/size",
+    )?;
+    let mut aliased = rows(&new_private, channels);
+    aliased[batch - 1].first = aliased[0].first;
+    refuses(
+        PreparedCertifiedScatter::prepare(
+            client,
+            &aliased,
+            &dense[0],
+            dense_v,
+            shape,
+            &statuses[1],
+        ),
+        "aliased",
+    )?;
+    if service.resident_bytes()? != bytes_before
+        || service.uploaded_value_bytes() != uploads_before
+        || service.pending_events()? != 0
+    {
+        return Err(Error::State(
+            "certified scatter cold checks allocated/uploaded/submitted".into(),
+        ));
+    }
+    // Initialization has already completed. The actual work uses a fresh original
+    // selected slot; no status is cleared or synthesized after its seed upload.
+    slot.finish()?;
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: 4,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    let mut accesses = input
+        .iter()
+        .cloned()
+        .map(|l| (l, DeviceAccess::Read))
+        .collect::<Vec<_>>();
+    accesses.extend(
+        output
+            .iter()
+            .chain(&status)
+            .cloned()
+            .map(|l| (l, DeviceAccess::Write)),
+    );
+    let promoted = output.iter().chain(&status).cloned().collect::<Vec<_>>();
+    let order = input
+        .iter()
+        .chain(&output)
+        .chain(&status)
+        .collect::<Vec<_>>();
+    let reads = order
+        .iter()
+        .map(|l| l.readback(0, l.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
+    slot.begin_submission()?;
+    let flight = service.submit(0, &accesses)?;
+    let gathered = super::launch_original(
+        client,
+        &input_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        Direction::Gather,
+        &statuses[0],
+    )?;
+    let mut original = super::launch_original(
+        client,
+        &old_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        Direction::Scatter,
+        &statuses[0],
+    )?;
+    for handle in &old_private {
+        for offset in (0..words).step_by(crate::program_kernels::MAX_WORD_OPS) {
+            let count = (words - offset).min(crate::program_kernels::MAX_WORD_OPS);
+            // SAFETY: independent original preflight covers these complete
+            // original views; original flight retains every accessed lease.
+            unsafe {
+                crate::program_kernels::certify_finite_strided::launch_unchecked(
+                    client,
+                    grid(count),
+                    threads(),
+                    arg(handle, words),
+                    arg(&statuses[0], 1),
+                    offset as u32,
+                    count as u32,
+                    1,
+                    count,
+                );
+            }
+            original += 1;
+        }
+    }
+    let native = prepared.launch(client)?;
+    if stop == Stop::Abandon {
+        // Real original commands are enqueued; omit ANY trusted terminal
+        // readback. Unknown completion must retain original handles and charge.
+        drop(flight);
+        drop(slot);
+        let state = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
+        if !state.fenced
+            || state.reserved == 0
+            || service.pending_events()? == 0
+            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
+            || service.ensure_usable().is_ok()
+        {
+            return Err(Error::State(
+                "certified scatter unknown completion not quarantined/fenced".into(),
+            ));
+        }
+        return Ok((0, original + gathered, native));
+    }
+    if stop == Stop::RejectStatus {
+        let mut observed = false;
+        let result = flight
+            .complete_with_readback(&promoted, &reads, |read| {
+                if expected_status & crate::program_kernels::NONFINITE_VALUE == 0
+                    || read.len() != order.len()
+                {
+                    return Err(Error::State(
+                        "certified scatter finite-status fixture coverage".into(),
+                    ));
+                }
+                for bytes in &read[read.len() - 2..] {
+                    check_words(bytes, 1, |_| expected_status)?;
+                }
+                observed = true;
+                // SAME original owner policy: no Ready follows a nonzero final
+                // certificate. Its armed flight conservatively quarantines/fences.
+                Err(Error::State(
+                    "certified scatter original finite-status rejected".into(),
+                ))
+            })
+            .await;
+        drop(slot);
+        if !observed
+            || !matches!(result, Err(ref e) if e.to_string().contains("original finite-status rejected"))
+            || service.ensure_usable().is_ok()
+            || service.pending_events()? == 0
+            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
+        {
+            return Err(Error::State(
+                "certified scatter finite-status produced Ready or escaped quarantine".into(),
+            ));
+        }
+        return Ok((0, original + gathered, native));
+    }
+    let mut checked = 0;
+    let (ready, read) = flight
+        .complete_with_readback(&promoted, &reads, |read| {
+            if read.len() != input.len() + output.len() + 2 {
+                return Err(Error::State(
+                    "certified scatter all readback bindings".into(),
+                ));
+            }
+            let mut index = 0;
+            for _set in 0..3 {
+                for row in 0..batch {
+                    for channel in 0..channels {
+                        checked += check_words(&read[index], words, |at| {
+                            word(row, channel, at, words, pattern, channels)
+                        })?;
+                        index += 1;
+                    }
+                }
+            }
+            for channel in 0..channels {
+                checked += check_words(&read[index], total, |at| {
+                    let (row, local) = match shape {
+                        Shape::Vector { .. } => (at / words, at % words),
+                        Shape::Kv { slab, .. } => (
+                            (at / slab) % batch,
+                            (at / (slab * batch)) * slab + at % slab,
+                        ),
+                    };
+                    word(row, channel, local, words, pattern, channels)
+                })?;
+                index += 1;
+            }
+            for _ in 0..2 {
+                checked += check_words(&read[index], 1, |_| expected_status)?;
+                index += 1;
+            }
+            if index != read.len() {
+                return Err(Error::State("certified scatter omitted readback".into()));
+            }
+            Ok(())
+        })
+        .await?;
+    slot.finish()?;
+    if original != batch * (1 + channels)
+        || gathered != batch
+        || native != channels
+        || ready.outputs().len() != promoted.len()
+        || read.len() != order.len()
+        || service.uploaded_value_bytes() != uploads_before
+    {
+        return Err(Error::State(
+            "certified scatter exact launch/Ready/upload accounting".into(),
+        ));
+    }
+    drop(ready);
+    Ok((checked, original, native))
+}
+
+async fn qualify(device: Device, path: &Path, stop: Stop) -> Result<()> {
+    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "certified scatter qualification requires release CUDA".into(),
+        ));
+    }
+    let parent = path
+        .parent()
+        .ok_or_else(|| Error::Malformed("approved source directory".into()))?;
+    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new(
+            "certified-state-scatter",
+            "synthetic-original-certificate-differential",
+        )?,
+        256 * 1024 * 1024,
+        5120 * 1024 * 1024,
+    )?;
+    let key = service.physical_key();
+    if stop != Stop::None {
+        let pattern = if stop == Stop::RejectStatus { 2 } else { 0 };
+        let shape = if stop == Stop::Abandon {
+            Shape::Kv {
+                slab: 512,
+                capacity: 512,
+            }
+        } else {
+            Shape::Vector { words: 262_144 }
+        };
+        case(&service, 8, shape, pattern, stop).await?;
+        drop(service);
+        let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
+        if !final_state.fenced || final_state.reserved == 0 {
+            return Err(Error::State(
+                "certified scatter unknown original envelope released".into(),
+            ));
+        }
+        if stop == Stop::Abandon {
+            println!(
+                "CUDA_CERTIFIED_SCATTER_UNKNOWN_PASS B8 KV_fields=2 original_commands_enqueued=1 terminal_readback=omitted pending_event=retained handles=quarantined service=fenced original_envelope=retained sync_or_recovery=not_claimed roots_published=0"
+            );
+        } else {
+            println!(
+                "CUDA_CERTIFIED_SCATTER_STATUS_REFUSAL_PASS B8 original_nonfinite_status=observed old_and_new_status_bits=exact trusted_numerical_Ready=refused pending_event=retained handles=quarantined service=fenced original_envelope=retained roots_published=0"
+            );
+        }
+        return Ok(());
+    }
+    let mut counts = (0usize, 0usize, 0usize, 0usize);
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
+            let channels = 1 + usize::from(matches!(shape, Shape::Kv { .. }));
+            for pattern in 0..batch * channels + 2 {
+                let got = case(&service, batch, shape, pattern, Stop::None).await?;
+                counts.0 += 1;
+                counts.1 += got.0;
+                counts.2 += got.1;
+                counts.3 += got.2;
+                if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+                    return Err(Error::State(
+                        "certified scatter private lease/event not reclaimed".into(),
+                    ));
+                }
+            }
+        }
+    }
+    drop(service);
+    let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_state.reserved != 0 || final_state.fenced {
+        return Err(Error::State(
+            "certified scatter final physical original envelope".into(),
+        ));
+    }
+    println!(
+        "CUDA_CERTIFIED_SCATTER_PASS B=2,4,8 cases={} words={} original_scatter_plus_certificate_launchsites={} fused_field_launchsites={} status=byte_exact_OR finite_signedzero_subnormal_Inf_NaN=preserved all_rows_fields=checked prefix_suffix_status_canaries=unchanged original_queue_leases_Ready=1 warm_uploads=0 final_reserved=0 roots_published=0 all42_fullmodel_worker_graph_speed=separate_gates",
+        counts.0, counts.1, counts.2, counts.3
+    );
+    Ok(())
+}
+
+#[test]
+#[ignore = "release original CUDA operator only; isolated GPU coordinator"]
+fn original_qwen35_08b_certified_state_scatter() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
+        Stop::None,
+    ))
+}
+#[test]
+#[ignore = "release original CUDA operator only; isolated GPU coordinator"]
+fn original_qwen35_2b_certified_state_scatter() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
+        Stop::None,
+    ))
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "release unknown completion fences device; ONE fresh process only"]
+fn original_qwen35_08b_certified_state_scatter_abandon() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
+        Stop::Abandon,
+    ))
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "release unknown completion fences device; ONE fresh process only"]
+fn original_qwen35_2b_certified_state_scatter_abandon() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
+        Stop::Abandon,
+    ))
+}
+
+#[test]
+#[ignore = "release nonfinite status refuses Ready and fences device; ONE fresh process"]
+fn original_qwen35_08b_certified_state_scatter_status_refusal() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
+        Stop::RejectStatus,
+    ))
+}
+#[test]
+#[ignore = "release nonfinite status refuses Ready and fences device; ONE fresh process"]
+fn original_qwen35_2b_certified_state_scatter_status_refusal() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
+        Stop::RejectStatus,
+    ))
+}
--- a/crates/zerogravity-gpu/src/resident_state_transport.rs
+++ b/crates/zerogravity-gpu/src/resident_state_transport.rs
@@ -7,6 +7,9 @@
 use cubecl_common::device::ServiceId;
 use std::ops::Range;
 use zerogravity_core::{Error, Result};
+
+mod certified;
+pub(crate) use certified::{PreparedCertifiedScatter, ensure_certified_capability};
 
 const MAX_ROW_WORDS: usize = 1_000_000;
 const MAX_TOTAL_WORDS: usize = 8_000_000;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_state_transport/certified.rs
@@ -0,0 +1,205 @@
+//! Typed final-state scatter + original finite-status certificate, no FP math.
+//! Opaque U32 transport stays separate; only the model owner selects this API.
+use super::*;
+use crate::program_kernels::NONFINITE_VALUE;
+
+pub(crate) fn ensure_certified_capability(
+    client: &Client,
+    shape: Shape,
+    rows: usize,
+) -> Result<()> {
+    ensure_capability(client, shape, rows)?;
+    if client.properties().hardware.max_bindings < 10 {
+        return Err(Error::Unsupported(
+            "certified state scatter CUDA binding limits".into(),
+        ));
+    }
+    Ok(())
+}
+
+/// Checked original handles only. Caller retains the existing status and ALL
+/// state leases in its original physical flight before the first launch.
+pub(crate) struct PreparedCertifiedScatter {
+    transport: PreparedTransport,
+    status: Handle,
+}
+
+impl PreparedCertifiedScatter {
+    pub(crate) fn prepare(
+        client: &Client,
+        rows: &[Row<'_>],
+        dense_first: &Handle,
+        dense_second: Option<&Handle>,
+        shape: Shape,
+        status: &Handle,
+    ) -> Result<Self> {
+        ensure_certified_capability(client, shape, rows.len())?;
+        let transport = PreparedTransport::prepare(
+            client,
+            rows,
+            dense_first,
+            dense_second,
+            shape,
+            Direction::Scatter,
+        )?;
+        // Each typed field uses8 row bindings +dense +status. KV fields are
+        // separate launches, retaining the accepted <=18 binding ceiling.
+        let status_range = buffer(client, status, 1)?;
+        let status_binding = status.clone().binding();
+        for handle in transport
+            .first
+            .iter()
+            .chain(&transport.second)
+            .chain(std::iter::once(&transport.dense_first))
+            .chain(transport.dense_second.iter())
+        {
+            let range = checked_span(
+                handle.size(),
+                handle.offset_start.unwrap_or(0),
+                handle.offset_end.unwrap_or(0),
+                1,
+            )?;
+            if handle.clone().binding().memory.id() == status_binding.memory.id()
+                && overlaps(&range, &status_range)
+            {
+                return Err(Error::State(
+                    "certified state scatter status aliases state".into(),
+                ));
+            }
+        }
+        Ok(Self {
+            transport,
+            status: status.clone(),
+        })
+    }
+
+    /// Vector1/KV2 actual source launches. Never clears existing status bits.
+    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
+        let transport = &self.transport;
+        if client.service_id() != transport.service {
+            return Err(Error::State(
+                "certified state scatter service changed".into(),
+            ));
+        }
+        let slab = match transport.shape {
+            Shape::Vector { .. } => 1,
+            Shape::Kv { slab, .. } => slab,
+        };
+        let interleaved = matches!(transport.shape, Shape::Kv { .. });
+        let channels = 1 + usize::from(interleaved);
+        for channel in 0..channels {
+            let (rows, dense) = if channel == 0 {
+                (&transport.first, &transport.dense_first)
+            } else {
+                (
+                    &transport.second,
+                    transport.dense_second.as_ref().expect("checked original V"),
+                )
+            };
+            // SAFETY: full service/range/alias/geometry preflight precedes
+            // enclosing submission. Inactive ABI aliases cannot be addressed.
+            let row = |index: usize| unsafe {
+                BufferArg::from_raw_parts(
+                    rows.get(index).unwrap_or(&rows[0]).clone(),
+                    transport.words,
+                )
+            };
+            // SAFETY: the original flight retains every checked state/status
+            // lease; all visible spans and CUDA launch bounds are prevalidated.
+            unsafe {
+                scatter_field::launch_unchecked(
+                    client,
+                    CubeCount::Static((transport.total as u32).div_ceil(LANES), 1, 1),
+                    CubeDim::new_1d(LANES),
+                    row(0),
+                    row(1),
+                    row(2),
+                    row(3),
+                    row(4),
+                    row(5),
+                    row(6),
+                    row(7),
+                    BufferArg::from_raw_parts(dense.clone(), transport.total),
+                    BufferArg::from_raw_parts(self.status.clone(), 1),
+                    transport.words,
+                    slab,
+                    rows.len(),
+                    interleaved,
+                );
+            }
+        }
+        Ok(channels)
+    }
+}
+
+fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
+    a.start < b.end && b.start < a.end
+}
+
+#[cube(launch_unchecked)]
+fn scatter_field(
+    p0: &mut [u32],
+    p1: &mut [u32],
+    p2: &mut [u32],
+    p3: &mut [u32],
+    p4: &mut [u32],
+    p5: &mut [u32],
+    p6: &mut [u32],
+    p7: &mut [u32],
+    dense: &[u32],
+    status: &[Atomic<u32>],
+    #[comptime] words: usize,
+    #[comptime] slab: usize,
+    #[comptime] rows: usize,
+    #[comptime] interleaved: bool,
+) {
+    let at = ABSOLUTE_POS;
+    if at < rows * words {
+        let row = at / words;
+        let local = at % words;
+        let mut packed = at;
+        if interleaved {
+            packed = ((local / slab) * rows + row) * slab + local % slab;
+        }
+        let bits = dense[packed];
+        if row == 0usize {
+            p0[local] = bits;
+        } else if row == 1usize {
+            p1[local] = bits;
+        } else if row == 2usize {
+            p2[local] = bits;
+        } else if row == 3usize {
+            p3[local] = bits;
+        } else if row == 4usize {
+            p4[local] = bits;
+        } else if row == 5usize {
+            p5[local] = bits;
+        } else if row == 6usize {
+            p6[local] = bits;
+        } else if row == 7usize {
+            p7[local] = bits;
+        }
+        // EXACT original certify_finite_strided predicate/status bit. Reads
+        // the same stored U32 value; no cast, FP operation or zero exception.
+        if (bits & 0x7f800000u32) == 0x7f800000u32 {
+            status[0].fetch_or(NONFINITE_VALUE);
+        }
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn exact_status_word_span_and_disjoint_view_boundaries() {
+        assert_eq!(checked_span(64, 16, 44, 1).unwrap(), 16..20);
+        assert!(!overlaps(&(0..16), &(16..20)));
+        assert!(!overlaps(&(20..64), &(16..20)));
+        assert!(overlaps(&(0..64), &(16..20)));
+        assert!(overlaps(&(16..20), &(16..20)));
+        for (size, start, trim) in [(3, 0, 0), (64, 1, 0), (64, 64, 0), (64, 0, 65)] {
+            assert!(checked_span(size, start, trim, 1).is_err());
+        }
+    }
+}

````

## Source: .agents/runs/serving-and-speculation-state-scatter-certification-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 36262; SHA256 `091628b52766e4a1f1975a71662c1d336c70e66c2b933f49bba95b2fc02fa40a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -7,7 +7,22 @@
 use super::*;
 use crate::activation_quant::{self, PARAMETER_WORDS};
 use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
-use crate::resident_state_transport::{Direction, PreparedTransport, Row, Shape};
+use crate::resident_state_transport::{
+    Direction, PreparedCertifiedScatter, PreparedTransport, Row, Shape,
+};
+
+enum PreparedStateTransport {
+    Opaque(PreparedTransport),
+    Certified(PreparedCertifiedScatter),
+}
+impl PreparedStateTransport {
+    fn launch(&self, client: &Client) -> Result<usize> {
+        match self {
+            Self::Opaque(transport) => transport.launch(client),
+            Self::Certified(transport) => transport.launch(client),
+        }
+    }
+}
 use crate::source_float::{round_source, widen_source};
 
 fn transport_shape(layout: StateLayout) -> Shape {
@@ -114,7 +129,7 @@
             ));
         }
         for state in &preflight.states {
-            crate::resident_state_transport::ensure_capability(
+            crate::resident_state_transport::ensure_certified_capability(
                 client,
                 transport_shape(state.layout),
                 rows,
@@ -644,9 +659,8 @@
                     .iter()
                     .find(|b| b.slot == version.slot)
                     .expect("validated domain");
-                for value in binding.candidate.buffers() {
-                    dispatches += self.certify(client, &value.handle, value.elements);
-                }
+                // Typed scatter copied and certified EVERY original candidate
+                // word in the same launches; keep the exact all-domain proposal.
                 proposals.push((*version, binding.candidate.clone()));
             }
         }
@@ -704,15 +718,10 @@
         client: &Client,
         bindings: &[Vec<StateBinding>],
         scatter: bool,
-    ) -> Result<Vec<PreparedTransport>> {
+    ) -> Result<Vec<PreparedStateTransport>> {
         if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
             return Err(Error::State("state transport domain/row coverage".into()));
         }
-        let direction = if scatter {
-            Direction::Scatter
-        } else {
-            Direction::Gather
-        };
         let mut prepared = Vec::with_capacity(self.packed.len());
         for packed in &self.packed {
             let dense = if scatter {
@@ -752,14 +761,25 @@
                             _ => Err(Error::State("state transport vector row changed".into())),
                         })
                         .collect::<Result<Vec<_>>>()?;
-                    prepared.push(PreparedTransport::prepare(
-                        client,
-                        &rows,
-                        &dense.handle,
-                        None,
-                        Shape::Vector { words },
-                        direction,
-                    )?);
+                    prepared.push(if scatter {
+                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
+                            client,
+                            &rows,
+                            &dense.handle,
+                            None,
+                            Shape::Vector { words },
+                            &self.inner.status,
+                        )?)
+                    } else {
+                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
+                            client,
+                            &rows,
+                            &dense.handle,
+                            None,
+                            Shape::Vector { words },
+                            Direction::Gather,
+                        )?)
+                    });
                 }
                 StateBuffers::Kv { keys, values } => {
                     let capacity = self.inner.plan.prefix_capacity();
@@ -791,14 +811,25 @@
                             _ => Err(Error::State("state transport KV row changed".into())),
                         })
                         .collect::<Result<Vec<_>>>()?;
-                    prepared.push(PreparedTransport::prepare(
-                        client,
-                        &rows,
-                        &keys.handle,
-                        Some(&values.handle),
-                        Shape::Kv { slab, capacity },
-                        direction,
-                    )?);
+                    prepared.push(if scatter {
+                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
+                            client,
+                            &rows,
+                            &keys.handle,
+                            Some(&values.handle),
+                            Shape::Kv { slab, capacity },
+                            &self.inner.status,
+                        )?)
+                    } else {
+                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
+                            client,
+                            &rows,
+                            &keys.handle,
+                            Some(&values.handle),
+                            Shape::Kv { slab, capacity },
+                            Direction::Gather,
+                        )?)
+                    });
                 }
             }
         }
--- a/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification.rs
@@ -65,3 +65,6 @@
 }
 
 mod physical;
+
+// Typed certification has a separate differential/status/unknown qualifier.
+mod certified;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/state_transport_qualification/certified.rs
@@ -0,0 +1,606 @@
+//! ROOT-only real CUDA differential: original copy +original finite certificate.
+//! Synthetic state words only; trained all42/fullV/worker gates are separate.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentService};
+use crate::resident_state_transport::{PreparedCertifiedScatter, geometry};
+use cubecl::{CubeElement, Device, bytes::Bytes};
+use std::{path::Path, sync::Arc};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+use zerogravity_model_qwen35::LoadedQwenModel;
+
+const PAD: usize = 4;
+const CANARY: u32 = 0xa5a55a5a;
+const FINITE: [u32; 8] = [
+    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
+];
+const NONFINITE: [u32; 4] = [0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345];
+
+fn word(
+    row: usize,
+    channel: usize,
+    at: usize,
+    words: usize,
+    pattern: usize,
+    channels: usize,
+) -> u32 {
+    if pattern >= 2 && row * channels + channel == pattern - 2 && at + 1 == words {
+        NONFINITE[(pattern - 2) % NONFINITE.len()]
+    } else if at % 16 < FINITE.len() {
+        FINITE[(at + row + channel) % FINITE.len()]
+    } else {
+        0x3f000000 | ((at as u32).wrapping_mul(73) & 0x007fffff)
+    }
+}
+fn visible(lease: &ResidentLease, words: usize) -> Handle {
+    lease
+        .handle()
+        .offset_start((PAD * 4) as u64)
+        .offset_end((lease.bytes() - (PAD + words) * 4) as u64)
+}
+fn rows(handles: &[Handle], channels: usize) -> Vec<Row<'_>> {
+    handles
+        .chunks_exact(channels)
+        .map(|h| Row {
+            first: &h[0],
+            second: (channels == 2).then(|| &h[1]),
+        })
+        .collect()
+}
+fn refuses(value: Result<PreparedCertifiedScatter>, text: &str) -> Result<()> {
+    if !matches!(value, Err(ref e) if e.to_string().contains(text)) {
+        return Err(Error::State(format!(
+            "certified scatter expected cold refusal: {text}"
+        )));
+    }
+    Ok(())
+}
+fn check_words(read: &Bytes, words: usize, value: impl Fn(usize) -> u32) -> Result<usize> {
+    if read.len() != (words + 2 * PAD) * 4 {
+        return Err(Error::State(
+            "certified scatter complete canary byte count".into(),
+        ));
+    }
+    for (at, raw) in read.chunks_exact(4).enumerate() {
+        let expected = if at < PAD || at >= PAD + words {
+            CANARY
+        } else {
+            value(at - PAD)
+        };
+        if raw != expected.to_ne_bytes() {
+            return Err(Error::State(format!(
+                "certified scatter raw bit/canary mismatch at{at}"
+            )));
+        }
+    }
+    Ok(words + 2 * PAD)
+}
+
+#[derive(Clone, Copy, PartialEq, Eq)]
+enum Stop {
+    None,
+    Abandon,
+    RejectStatus,
+}
+
+async fn case(
+    service: &Arc<ResidentService>,
+    batch: usize,
+    shape: Shape,
+    pattern: usize,
+    stop: Stop,
+) -> Result<(usize, usize, usize)> {
+    let (words, total) = geometry(shape, batch)?;
+    let channels = 1 + usize::from(matches!(shape, Shape::Kv { .. }));
+    if pattern >= batch * channels + 2 {
+        return Err(Error::Malformed("certified scatter fixture pattern".into()));
+    }
+    let seed = if pattern == 0 {
+        0
+    } else {
+        crate::program_kernels::VIEW_OUT_OF_BOUNDS | crate::program_kernels::INVALID_REDUCTION
+    };
+    let expected_status = seed
+        | if pattern >= 2 {
+            crate::program_kernels::NONFINITE_VALUE
+        } else {
+            0
+        };
+    let bytes = (words + 2 * PAD) * 4 * batch * channels * 3
+        + (total + 2 * PAD) * 4 * channels
+        + (1 + 2 * PAD) * 4 * 2;
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
+        return Err(Error::Limit(
+            "certified scatter fixture physical envelope".into(),
+        ));
+    }
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(0)?;
+    let mut input = Vec::new();
+    for row in 0..batch {
+        for channel in 0..channels {
+            let mut data = vec![CANARY; words + 2 * PAD];
+            for (at, value) in data[PAD..PAD + words].iter_mut().enumerate() {
+                *value = word(row, channel, at, words, pattern, channels);
+            }
+            input.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
+        }
+    }
+    let mut output = Vec::new();
+    for _ in 0..2 * batch * channels {
+        output.push(init.upload_bytes(
+            u32::as_bytes(&vec![CANARY; words + 2 * PAD]),
+            BufferRole::Scratch,
+        )?);
+    }
+    for _ in 0..channels {
+        output.push(init.upload_bytes(
+            u32::as_bytes(&vec![CANARY; total + 2 * PAD]),
+            BufferRole::Scratch,
+        )?);
+    }
+    let mut status = Vec::new();
+    for _ in 0..2 {
+        let mut data = [CANARY; 1 + 2 * PAD];
+        data[PAD] = seed;
+        status.push(init.upload_bytes(u32::as_bytes(&data), BufferRole::Scratch)?);
+    }
+    let initialized = init.complete_sync().await?;
+    if initialized.outputs().len() != input.len() + output.len() + status.len() {
+        return Err(Error::State(
+            "certified scatter initialized lease coverage".into(),
+        ));
+    }
+    drop(initialized);
+    let client = service.client();
+    let inputs = input.iter().map(|l| visible(l, words)).collect::<Vec<_>>();
+    let old_private = output[..batch * channels]
+        .iter()
+        .map(|l| visible(l, words))
+        .collect::<Vec<_>>();
+    let new_private = output[batch * channels..2 * batch * channels]
+        .iter()
+        .map(|l| visible(l, words))
+        .collect::<Vec<_>>();
+    let dense = output[2 * batch * channels..]
+        .iter()
+        .map(|l| visible(l, total))
+        .collect::<Vec<_>>();
+    let statuses = status.iter().map(|l| visible(l, 1)).collect::<Vec<_>>();
+    let input_rows = rows(&inputs, channels);
+    let old_rows = rows(&old_private, channels);
+    let new_rows = rows(&new_private, channels);
+    let dense_v = (channels == 2).then(|| &dense[1]);
+    // BOTH complete old/new span+status bindings pass before the original flight.
+    PreparedCertifiedScatter::prepare(client, &old_rows, &dense[0], dense_v, shape, &statuses[0])?;
+    let prepared = PreparedCertifiedScatter::prepare(
+        client,
+        &new_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        &statuses[1],
+    )?;
+    PreparedTransport::prepare(
+        client,
+        &input_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        Direction::Gather,
+    )?;
+    let bytes_before = service.resident_bytes()?;
+    let uploads_before = service.uploaded_value_bytes();
+    let mut foreign = statuses[1].clone();
+    foreign.service.service = std::any::TypeId::of::<()>();
+    refuses(
+        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &foreign),
+        "service",
+    )?;
+    refuses(
+        PreparedCertifiedScatter::prepare(
+            client,
+            &new_rows,
+            &dense[0],
+            dense_v,
+            shape,
+            &new_private[0],
+        ),
+        "aliases",
+    )?;
+    refuses(
+        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &dense[0]),
+        "aliases",
+    )?;
+    let mut short = statuses[1].clone();
+    short.offset_end = Some(short.size() - (PAD * 4) as u64);
+    refuses(
+        PreparedCertifiedScatter::prepare(client, &new_rows, &dense[0], dense_v, shape, &short),
+        "alignment/size",
+    )?;
+    let mut misaligned = statuses[1].clone();
+    misaligned.offset_start = Some((PAD * 4 + 1) as u64);
+    refuses(
+        PreparedCertifiedScatter::prepare(
+            client,
+            &new_rows,
+            &dense[0],
+            dense_v,
+            shape,
+            &misaligned,
+        ),
+        "alignment/size",
+    )?;
+    let mut aliased = rows(&new_private, channels);
+    aliased[batch - 1].first = aliased[0].first;
+    refuses(
+        PreparedCertifiedScatter::prepare(
+            client,
+            &aliased,
+            &dense[0],
+            dense_v,
+            shape,
+            &statuses[1],
+        ),
+        "aliased",
+    )?;
+    if service.resident_bytes()? != bytes_before
+        || service.uploaded_value_bytes() != uploads_before
+        || service.pending_events()? != 0
+    {
+        return Err(Error::State(
+            "certified scatter cold checks allocated/uploaded/submitted".into(),
+        ));
+    }
+    // Initialization has already completed. The actual work uses a fresh original
+    // selected slot; no status is cleared or synthesized after its seed upload.
+    slot.finish()?;
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: 4,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    let mut accesses = input
+        .iter()
+        .cloned()
+        .map(|l| (l, DeviceAccess::Read))
+        .collect::<Vec<_>>();
+    accesses.extend(
+        output
+            .iter()
+            .chain(&status)
+            .cloned()
+            .map(|l| (l, DeviceAccess::Write)),
+    );
+    let promoted = output.iter().chain(&status).cloned().collect::<Vec<_>>();
+    let order = input
+        .iter()
+        .chain(&output)
+        .chain(&status)
+        .collect::<Vec<_>>();
+    let reads = order
+        .iter()
+        .map(|l| l.readback(0, l.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
+    slot.begin_submission()?;
+    let flight = service.submit(0, &accesses)?;
+    let gathered = super::launch_original(
+        client,
+        &input_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        Direction::Gather,
+        &statuses[0],
+    )?;
+    let mut original = super::launch_original(
+        client,
+        &old_rows,
+        &dense[0],
+        dense_v,
+        shape,
+        Direction::Scatter,
+        &statuses[0],
+    )?;
+    for handle in &old_private {
+        for offset in (0..words).step_by(crate::program_kernels::MAX_WORD_OPS) {
+            let count = (words - offset).min(crate::program_kernels::MAX_WORD_OPS);
+            // SAFETY: independent original preflight covers these complete
+            // original views; original flight retains every accessed lease.
+            unsafe {
+                crate::program_kernels::certify_finite_strided::launch_unchecked(
+                    client,
+                    grid(count),
+                    threads(),
+                    arg(handle, words),
+                    arg(&statuses[0], 1),
+                    offset as u32,
+                    count as u32,
+                    1,
+                    count,
+                );
+            }
+            original += 1;
+        }
+    }
+    let native = prepared.launch(client)?;
+    if stop == Stop::Abandon {
+        // Real original commands are enqueued; omit ANY trusted terminal
+        // readback. Unknown completion must retain original handles and charge.
+        drop(flight);
+        drop(slot);
+        let state = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
+        if !state.fenced
+            || state.reserved == 0
+            || service.pending_events()? == 0
+            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
+            || service.ensure_usable().is_ok()
+        {
+            return Err(Error::State(
+                "certified scatter unknown completion not quarantined/fenced".into(),
+            ));
+        }
+        return Ok((0, original + gathered, native));
+    }
+    if stop == Stop::RejectStatus {
+        let mut observed = false;
+        let result = flight
+            .complete_with_readback(&promoted, &reads, |read| {
+                if expected_status & crate::program_kernels::NONFINITE_VALUE == 0
+                    || read.len() != order.len()
+                {
+                    return Err(Error::State(
+                        "certified scatter finite-status fixture coverage".into(),
+                    ));
+                }
+                for bytes in &read[read.len() - 2..] {
+                    check_words(bytes, 1, |_| expected_status)?;
+                }
+                observed = true;
+                // SAME original owner policy: no Ready follows a nonzero final
+                // certificate. Its armed flight conservatively quarantines/fences.
+                Err(Error::State(
+                    "certified scatter original finite-status rejected".into(),
+                ))
+            })
+            .await;
+        drop(slot);
+        if !observed
+            || !matches!(result, Err(ref e) if e.to_string().contains("original finite-status rejected"))
+            || service.ensure_usable().is_ok()
+            || service.pending_events()? == 0
+            || crate::physical_quarantine::quarantined_handles_for(client)? <= quarantine_before
+        {
+            return Err(Error::State(
+                "certified scatter finite-status produced Ready or escaped quarantine".into(),
+            ));
+        }
+        return Ok((0, original + gathered, native));
+    }
+    let mut checked = 0;
+    let (ready, read) = flight
+        .complete_with_readback(&promoted, &reads, |read| {
+            if read.len() != input.len() + output.len() + 2 {
+                return Err(Error::State(
+                    "certified scatter all readback bindings".into(),
+                ));
+            }
+            let mut index = 0;
+            for _set in 0..3 {
+                for row in 0..batch {
+                    for channel in 0..channels {
+                        checked += check_words(&read[index], words, |at| {
+                            word(row, channel, at, words, pattern, channels)
+                        })?;
+                        index += 1;
+                    }
+                }
+            }
+            for channel in 0..channels {
+                checked += check_words(&read[index], total, |at| {
+                    let (row, local) = match shape {
+                        Shape::Vector { .. } => (at / words, at % words),
+                        Shape::Kv { slab, .. } => (
+                            (at / slab) % batch,
+                            (at / (slab * batch)) * slab + at % slab,
+                        ),
+                    };
+                    word(row, channel, local, words, pattern, channels)
+                })?;
+                index += 1;
+            }
+            for _ in 0..2 {
+                checked += check_words(&read[index], 1, |_| expected_status)?;
+                index += 1;
+            }
+            if index != read.len() {
+                return Err(Error::State("certified scatter omitted readback".into()));
+            }
+            Ok(())
+        })
+        .await?;
+    slot.finish()?;
+    if original != batch * (1 + channels)
+        || gathered != batch
+        || native != channels
+        || ready.outputs().len() != promoted.len()
+        || read.len() != order.len()
+        || service.uploaded_value_bytes() != uploads_before
+    {
+        return Err(Error::State(
+            "certified scatter exact launch/Ready/upload accounting".into(),
+        ));
+    }
+    drop(ready);
+    Ok((checked, original, native))
+}
+
+async fn qualify(device: Device, path: &Path, stop: Stop) -> Result<()> {
+    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "certified scatter qualification requires release CUDA".into(),
+        ));
+    }
+    let parent = path
+        .parent()
+        .ok_or_else(|| Error::Malformed("approved source directory".into()))?;
+    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new(
+            "certified-state-scatter",
+            "synthetic-original-certificate-differential",
+        )?,
+        256 * 1024 * 1024,
+        5120 * 1024 * 1024,
+    )?;
+    let key = service.physical_key();
+    if stop != Stop::None {
+        let pattern = if stop == Stop::RejectStatus { 2 } else { 0 };
+        let shape = if stop == Stop::Abandon {
+            Shape::Kv {
+                slab: 512,
+                capacity: 512,
+            }
+        } else {
+            Shape::Vector { words: 262_144 }
+        };
+        case(&service, 8, shape, pattern, stop).await?;
+        drop(service);
+        let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
+        if !final_state.fenced || final_state.reserved == 0 {
+            return Err(Error::State(
+                "certified scatter unknown original envelope released".into(),
+            ));
+        }
+        if stop == Stop::Abandon {
+            println!(
+                "CUDA_CERTIFIED_SCATTER_UNKNOWN_PASS B8 KV_fields=2 original_commands_enqueued=1 terminal_readback=omitted pending_event=retained handles=quarantined service=fenced original_envelope=retained sync_or_recovery=not_claimed roots_published=0"
+            );
+        } else {
+            println!(
+                "CUDA_CERTIFIED_SCATTER_STATUS_REFUSAL_PASS B8 original_nonfinite_status=observed old_and_new_status_bits=exact trusted_numerical_Ready=refused pending_event=retained handles=quarantined service=fenced original_envelope=retained roots_published=0"
+            );
+        }
+        return Ok(());
+    }
+    let mut counts = (0usize, 0usize, 0usize, 0usize);
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
+            let channels = 1 + usize::from(matches!(shape, Shape::Kv { .. }));
+            for pattern in 0..batch * channels + 2 {
+                let got = case(&service, batch, shape, pattern, Stop::None).await?;
+                counts.0 += 1;
+                counts.1 += got.0;
+                counts.2 += got.1;
+                counts.3 += got.2;
+                if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+                    return Err(Error::State(
+                        "certified scatter private lease/event not reclaimed".into(),
+                    ));
+                }
+            }
+        }
+    }
+    drop(service);
+    let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_state.reserved != 0 || final_state.fenced {
+        return Err(Error::State(
+            "certified scatter final physical original envelope".into(),
+        ));
+    }
+    println!(
+        "CUDA_CERTIFIED_SCATTER_PASS B=2,4,8 cases={} words={} original_scatter_plus_certificate_launchsites={} fused_field_launchsites={} status=byte_exact_OR finite_signedzero_subnormal_Inf_NaN=preserved all_rows_fields=checked prefix_suffix_status_canaries=unchanged original_queue_leases_Ready=1 warm_uploads=0 final_reserved=0 roots_published=0 all42_fullmodel_worker_graph_speed=separate_gates",
+        counts.0, counts.1, counts.2, counts.3
+    );
+    Ok(())
+}
+
+#[test]
+#[ignore = "release original CUDA operator only; isolated GPU coordinator"]
+fn original_qwen35_08b_certified_state_scatter() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
+        Stop::None,
+    ))
+}
+#[test]
+#[ignore = "release original CUDA operator only; isolated GPU coordinator"]
+fn original_qwen35_2b_certified_state_scatter() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
+        Stop::None,
+    ))
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "release unknown completion fences device; ONE fresh process only"]
+fn original_qwen35_08b_certified_state_scatter_abandon() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
+        Stop::Abandon,
+    ))
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "release unknown completion fences device; ONE fresh process only"]
+fn original_qwen35_2b_certified_state_scatter_abandon() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
+        Stop::Abandon,
+    ))
+}
+
+#[test]
+#[ignore = "release nonfinite status refuses Ready and fences device; ONE fresh process"]
+fn original_qwen35_08b_certified_state_scatter_status_refusal() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b/model.gguf"),
+        Stop::RejectStatus,
+    ))
+}
+#[test]
+#[ignore = "release nonfinite status refuses Ready and fences device; ONE fresh process"]
+fn original_qwen35_2b_certified_state_scatter_status_refusal() -> Result<()> {
+    futures_lite::future::block_on(qualify(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-2b/model.gguf"),
+        Stop::RejectStatus,
+    ))
+}
--- a/crates/zerogravity-gpu/src/resident_state_transport.rs
+++ b/crates/zerogravity-gpu/src/resident_state_transport.rs
@@ -7,6 +7,9 @@
 use cubecl_common::device::ServiceId;
 use std::ops::Range;
 use zerogravity_core::{Error, Result};
+
+mod certified;
+pub(crate) use certified::{PreparedCertifiedScatter, ensure_certified_capability};
 
 const MAX_ROW_WORDS: usize = 1_000_000;
 const MAX_TOTAL_WORDS: usize = 8_000_000;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_state_transport/certified.rs
@@ -0,0 +1,205 @@
+//! Typed final-state scatter + original finite-status certificate, no FP math.
+//! Opaque U32 transport stays separate; only the model owner selects this API.
+use super::*;
+use crate::program_kernels::NONFINITE_VALUE;
+
+pub(crate) fn ensure_certified_capability(
+    client: &Client,
+    shape: Shape,
+    rows: usize,
+) -> Result<()> {
+    ensure_capability(client, shape, rows)?;
+    if client.properties().hardware.max_bindings < 10 {
+        return Err(Error::Unsupported(
+            "certified state scatter CUDA binding limits".into(),
+        ));
+    }
+    Ok(())
+}
+
+/// Checked original handles only. Caller retains the existing status and ALL
+/// state leases in its original physical flight before the first launch.
+pub(crate) struct PreparedCertifiedScatter {
+    transport: PreparedTransport,
+    status: Handle,
+}
+
+impl PreparedCertifiedScatter {
+    pub(crate) fn prepare(
+        client: &Client,
+        rows: &[Row<'_>],
+        dense_first: &Handle,
+        dense_second: Option<&Handle>,
+        shape: Shape,
+        status: &Handle,
+    ) -> Result<Self> {
+        ensure_certified_capability(client, shape, rows.len())?;
+        let transport = PreparedTransport::prepare(
+            client,
+            rows,
+            dense_first,
+            dense_second,
+            shape,
+            Direction::Scatter,
+        )?;
+        // Each typed field uses8 row bindings +dense +status. KV fields are
+        // separate launches, retaining the accepted <=18 binding ceiling.
+        let status_range = buffer(client, status, 1)?;
+        let status_binding = status.clone().binding();
+        for handle in transport
+            .first
+            .iter()
+            .chain(&transport.second)
+            .chain(std::iter::once(&transport.dense_first))
+            .chain(transport.dense_second.iter())
+        {
+            let range = checked_span(
+                handle.size(),
+                handle.offset_start.unwrap_or(0),
+                handle.offset_end.unwrap_or(0),
+                1,
+            )?;
+            if handle.clone().binding().memory.id() == status_binding.memory.id()
+                && overlaps(&range, &status_range)
+            {
+                return Err(Error::State(
+                    "certified state scatter status aliases state".into(),
+                ));
+            }
+        }
+        Ok(Self {
+            transport,
+            status: status.clone(),
+        })
+    }
+
+    /// Vector1/KV2 actual source launches. Never clears existing status bits.
+    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
+        let transport = &self.transport;
+        if client.service_id() != transport.service {
+            return Err(Error::State(
+                "certified state scatter service changed".into(),
+            ));
+        }
+        let slab = match transport.shape {
+            Shape::Vector { .. } => 1,
+            Shape::Kv { slab, .. } => slab,
+        };
+        let interleaved = matches!(transport.shape, Shape::Kv { .. });
+        let channels = 1 + usize::from(interleaved);
+        for channel in 0..channels {
+            let (rows, dense) = if channel == 0 {
+                (&transport.first, &transport.dense_first)
+            } else {
+                (
+                    &transport.second,
+                    transport.dense_second.as_ref().expect("checked original V"),
+                )
+            };
+            // SAFETY: full service/range/alias/geometry preflight precedes
+            // enclosing submission. Inactive ABI aliases cannot be addressed.
+            let row = |index: usize| unsafe {
+                BufferArg::from_raw_parts(
+                    rows.get(index).unwrap_or(&rows[0]).clone(),
+                    transport.words,
+                )
+            };
+            // SAFETY: the original flight retains every checked state/status
+            // lease; all visible spans and CUDA launch bounds are prevalidated.
+            unsafe {
+                scatter_field::launch_unchecked(
+                    client,
+                    CubeCount::Static((transport.total as u32).div_ceil(LANES), 1, 1),
+                    CubeDim::new_1d(LANES),
+                    row(0),
+                    row(1),
+                    row(2),
+                    row(3),
+                    row(4),
+                    row(5),
+                    row(6),
+                    row(7),
+                    BufferArg::from_raw_parts(dense.clone(), transport.total),
+                    BufferArg::from_raw_parts(self.status.clone(), 1),
+                    transport.words,
+                    slab,
+                    rows.len(),
+                    interleaved,
+                );
+            }
+        }
+        Ok(channels)
+    }
+}
+
+fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
+    a.start < b.end && b.start < a.end
+}
+
+#[cube(launch_unchecked)]
+fn scatter_field(
+    p0: &mut [u32],
+    p1: &mut [u32],
+    p2: &mut [u32],
+    p3: &mut [u32],
+    p4: &mut [u32],
+    p5: &mut [u32],
+    p6: &mut [u32],
+    p7: &mut [u32],
+    dense: &[u32],
+    status: &[Atomic<u32>],
+    #[comptime] words: usize,
+    #[comptime] slab: usize,
+    #[comptime] rows: usize,
+    #[comptime] interleaved: bool,
+) {
+    let at = ABSOLUTE_POS;
+    if at < rows * words {
+        let row = at / words;
+        let local = at % words;
+        let mut packed = at;
+        if interleaved {
+            packed = ((local / slab) * rows + row) * slab + local % slab;
+        }
+        let bits = dense[packed];
+        if row == 0usize {
+            p0[local] = bits;
+        } else if row == 1usize {
+            p1[local] = bits;
+        } else if row == 2usize {
+            p2[local] = bits;
+        } else if row == 3usize {
+            p3[local] = bits;
+        } else if row == 4usize {
+            p4[local] = bits;
+        } else if row == 5usize {
+            p5[local] = bits;
+        } else if row == 6usize {
+            p6[local] = bits;
+        } else if row == 7usize {
+            p7[local] = bits;
+        }
+        // EXACT original certify_finite_strided predicate/status bit. Reads
+        // the same stored U32 value; no cast, FP operation or zero exception.
+        if (bits & 0x7f800000u32) == 0x7f800000u32 {
+            status[0].fetch_or(NONFINITE_VALUE);
+        }
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn exact_status_word_span_and_disjoint_view_boundaries() {
+        assert_eq!(checked_span(64, 16, 44, 1).unwrap(), 16..20);
+        assert!(!overlaps(&(0..16), &(16..20)));
+        assert!(!overlaps(&(20..64), &(16..20)));
+        assert!(overlaps(&(0..64), &(16..20)));
+        assert!(overlaps(&(16..20), &(16..20)));
+        for (size, start, trim) in [(3, 0, 0), (64, 1, 0), (64, 64, 0), (64, 0, 65)] {
+            assert!(checked_span(size, start, trim, 1).is_err());
+        }
+    }
+}

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-baseline-20261020/BUILD_AFFINITY_V2.patch

Original UTF-8 bytes: 2690; SHA256 `b6e2776bdda200ab2878d5948574dfc660f38a19bb1382ceddde8cfc2c1a4c86`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- root_build.py
+++ root_build_v2.py
@@ -19,7 +19,7 @@
         assert path.is_file() and not path.is_symlink()
         assert path.stat().st_size == row['bytes'] and owned.sha(path) == row['sha256']
 check_source()
-output = HERE / 'root-build-1'
+output = HERE / 'root-build-2'
 output.mkdir(mode=0o700)
 nonce = secrets.token_hex(16)
 name = 'zg-gguf-plugin-build-' + nonce
@@ -32,7 +32,7 @@
 argv = owned.PODMAN + ['run', '--name', name, '--label', owned.LABEL+'='+nonce,
     '--cidfile', str(cidfile), '--rm', '--pull=never', '--network=none', '--read-only',
     '--cap-drop=all', '--security-opt=no-new-privileges', '--memory=8g', '--memory-swap=8g',
-    '--cpuset-cpus=0-3', '--cpus=4', '--pids-limit=256', '--shm-size=64m',
+    '--cpus=4', '--pids-limit=256', '--shm-size=64m',
     '--ulimit', 'fsize=536870912:536870912', '--tmpfs', '/tmp:rw,nosuid,nodev,size=4g']
 for key, value in dict(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
     TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='2', OMP_NUM_THREADS='4',
@@ -41,13 +41,13 @@
     VLLM_GGUF_PLUGIN_LOCAL_VERSION_SUFFIX='g'+manifest['revision'][:12]).items():
     argv += ['-e', key+'='+value]
 argv += ['-v', str(HERE)+':/probe:ro', '-v', str(HERE/'source')+':/plugin:ro',
-         '-v', str(output)+':/output:rw', '--entrypoint', 'python3', IMAGE, '/probe/build_inside.py']
+         '-v', str(output)+':/output:rw', '--entrypoint', 'python3', IMAGE, '/probe/build_inside_v2.py']
 receipt = dict(status='UNQUALIFIED', argv=argv, owned_name=name, owned_nonce=nonce,
     image_id=IMAGE, upstream_revision=manifest['revision'],
     source_manifest_sha256=owned.sha(HERE/'upstream-source-files.json'),
     helper_sha256=owned.sha(HERE/'owned_container_v4.py'),
-    root_builder_sha256=owned.sha(HERE/'root_build.py'),
-    inside_builder_sha256=owned.sha(HERE/'build_inside.py'), benchmark=False)
+    root_builder_sha256=owned.sha(HERE/'root_build_v2.py'),
+    inside_builder_sha256=owned.sha(HERE/'build_inside_v2.py'), benchmark=False)
 def write():
     (output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
 write()
--- build_inside.py
+++ build_inside_v2.py
@@ -12,7 +12,11 @@
 assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
 assert os.environ['TORCH_CUDA_ARCH_LIST'] == '8.6'
 assert os.environ['MAX_JOBS'] == '2'
-inventory = {}
+cpus = [0, 1, 2, 3]
+assert set(cpus) <= os.sched_getaffinity(0)
+os.sched_setaffinity(0, cpus)
+assert sorted(os.sched_getaffinity(0)) == cpus
+inventory = dict(actual_cpu_affinity=cpus)
 for name in ['torch', 'vllm', 'gguf', 'huggingface-hub', 'setuptools', 'wheel', 'ninja']:
     try:
         inventory[name] = importlib.metadata.version(name)

````

