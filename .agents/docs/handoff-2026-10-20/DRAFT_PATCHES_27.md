# Draft Patches — volume 27

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 90014; SHA256 `482b3e7aae5c582c097c9dcd07c60bb8dcf8076753cb8615e0a1db22eb42163e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/fused.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/fused.rs
@@ -5,6 +5,8 @@
     Native,
     #[cfg(all(feature = "qualification", feature = "cuda"))]
     NativeChunk(crate::resident_batch::ResidentBatchChunk),
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    Temporal(crate::resident_batch::ResidentBatchChunk),
     #[cfg(all(feature = "qualification", feature = "cuda"))]
     Graph(Arc<super::graph::CapturedResidentGraph>),
 }
@@ -27,6 +29,49 @@
         return Err(Error::Unsupported(
             "graph requires native host-token band and matching typed body/readout".into(),
         ));
+    }
+    Ok(())
+}
+
+fn allocate_batch_buffer(
+    initialization: &mut crate::resident_leases::ResidentInitialization<'_>,
+    leases: &mut Vec<(ResidentLease, DeviceAccess)>,
+    role: ResidentBufferRole,
+    count: usize,
+    data: AllocationInit<'_>,
+) -> Result<Handle> {
+    let role = match role {
+        ResidentBufferRole::Source => BufferRole::ImmutableWeights,
+        ResidentBufferRole::Value | ResidentBufferRole::Parameter => BufferRole::Activation,
+        ResidentBufferRole::Workspace | ResidentBufferRole::Status => BufferRole::Scratch,
+    };
+    let lease = match data {
+        AllocationInit::Empty => initialization.allocate_empty(count, role)?,
+        AllocationInit::Upload(data) => {
+            if data.len() != count {
+                return Err(Error::State("native initializer length".into()));
+            }
+            initialization.upload_bytes(data, role)?
+        }
+    };
+    let handle = lease.handle();
+    leases.push((lease, DeviceAccess::Write));
+    Ok(handle)
+}
+fn verify_batch_allocations(
+    bytes: usize,
+    handles: Vec<Handle>,
+    leases: &[(ResidentLease, DeviceAccess)],
+) -> Result<()> {
+    for handle in handles {
+        lease_for(leases, &handle)?;
+    }
+    let actual = leases.iter().try_fold(0usize, |n, (lease, _)| {
+        n.checked_add(lease.bytes())
+            .ok_or_else(|| Error::Limit("native allocation coverage overflow".into()))
+    })?;
+    if actual != bytes {
+        return Err(Error::State("native allocation coverage".into()));
     }
     Ok(())
 }
@@ -93,6 +138,39 @@
         .await
     }
 
+    /// Closed B2/T2 native dependency grouping; physical B4 activations are
+    /// never four sessions. Original atomic publisher advances two roots by2.
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub async fn prepare_state_temporal_b2_t2_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, &[u32])],
+    ) -> Result<Option<PreparedResidentCohort>> {
+        self.service.pinned_cuda_queue()?;
+        if rows.len() != 2 || rows.iter().any(|(_, tokens)| tokens.len() != 2) {
+            return Err(Error::Unsupported(
+                "temporal grouping requires B2/T2".into(),
+            ));
+        }
+        let bases = rows
+            .iter()
+            .map(|(session, _)| session.binding())
+            .collect::<Vec<_>>();
+        let teachers = rows.iter().map(|(_, tokens)| *tokens).collect::<Vec<_>>();
+        let chunk = crate::resident_batch::ResidentBatchChunk::new(
+            &bases,
+            &teachers,
+            self.vocab,
+            self.capacity,
+        )?;
+        let first = rows
+            .iter()
+            .enumerate()
+            .map(|(row, (session, _))| Ok((*session, chunk.token(row, 0)?)))
+            .collect::<Result<Vec<_>>>()?;
+        self.prepare_fused_execution(&first, None, Readout::None, FusedExecution::Temporal(chunk))
+            .await
+    }
+
     /// One native full-model batch and source-exact device greedy selection.
     /// Each independent output packet gains its producer only at cohort commit.
     pub async fn prepare_greedy_fused_cohort(
@@ -178,19 +256,38 @@
             graph_readout(rows.len(), graph.output(), readout, device_tokens.is_some())?;
         }
         #[cfg(all(feature = "qualification", feature = "cuda"))]
-        if matches!(&execution, FusedExecution::NativeChunk(_))
-            && (!matches!(readout, Readout::None) || device_tokens.is_some())
+        if matches!(
+            &execution,
+            FusedExecution::NativeChunk(_) | FusedExecution::Temporal(_)
+        ) && (!matches!(readout, Readout::None) || device_tokens.is_some())
         {
             return Err(Error::Unsupported(
                 "teacher chunk requires state-only host tokens".into(),
             ));
         }
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let temporal = matches!(&execution, FusedExecution::Temporal(_));
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if let FusedExecution::Temporal(chunk) = &execution {
+            self.service.pinned_cuda_queue()?;
+            if rows.len() != 2
+                || chunk.rows() != 2
+                || chunk.tokens_per_row() != 2
+                || self.capacity != 512
+            {
+                return Err(Error::Unsupported(
+                    "temporal closed B2/T2/cap512 only".into(),
+                ));
+            }
+        }
         let tokens_per_row = match &execution {
             FusedExecution::Native => 1,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
             FusedExecution::Graph(_) => 1,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
-            FusedExecution::NativeChunk(chunk) => chunk.tokens_per_row(),
+            FusedExecution::NativeChunk(chunk) | FusedExecution::Temporal(chunk) => {
+                chunk.tokens_per_row()
+            }
         };
         let output = if matches!(readout, Readout::None) {
             ResidentOutput::StateOnly
@@ -205,7 +302,7 @@
         let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
         let roots = ResidentBatchRows::new(&bases, 1)?;
         #[cfg(all(feature = "qualification", feature = "cuda"))]
-        if let FusedExecution::NativeChunk(chunk) = &execution {
+        if let FusedExecution::NativeChunk(chunk) | FusedExecution::Temporal(chunk) = &execution {
             chunk.require_current(&bases)?;
             for (row, (_, token)) in rows.iter().enumerate() {
                 if chunk.token(row, 0)? != *token {
@@ -295,11 +392,13 @@
             let mut workspace = self.workspace.lock().await;
             self.service.ensure_usable()?;
             #[cfg(all(feature = "qualification", feature = "cuda"))]
-            if matches!(&execution, FusedExecution::NativeChunk(_))
-                && workspace
-                    .batch
-                    .as_ref()
-                    .is_some_and(|batch| batch.program.graph_pinned())
+            if matches!(
+                &execution,
+                FusedExecution::NativeChunk(_) | FusedExecution::Temporal(_)
+            ) && workspace
+                .batch
+                .as_ref()
+                .is_some_and(|batch| batch.program.graph_pinned())
             {
                 return Err(Error::Unsupported(
                     "teacher chunk refuses a graph-pinned workspace before allocation".into(),
@@ -345,7 +444,7 @@
                 workspace
                     .batch
                     .as_ref()
-                    .map(|b| b.program.budget().owned_bytes())
+                    .map(BatchWorkspace::owned_bytes)
                     .transpose()?
                     .unwrap_or(0)
             };
@@ -353,6 +452,23 @@
                 0
             } else {
                 BatchProgram::preflight(&workspace.program, rows.len())?.owned_bytes()?
+            };
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let workspace_bytes = if temporal
+                && (!same_band
+                    || workspace
+                        .batch
+                        .as_ref()
+                        .is_none_or(|batch| batch.temporal.is_none()))
+            {
+                workspace_bytes
+                    .checked_add(
+                        crate::resident_program::TemporalProgram::preflight(&workspace.program)?
+                            .owned_bytes()?,
+                    )
+                    .ok_or_else(|| Error::Limit("temporal combined admission".into()))?
+            } else {
+                workspace_bytes
             };
             let required = staging_bytes
                 .checked_add(packet_bytes)
@@ -392,7 +508,9 @@
             })
             .collect::<Vec<_>>();
         #[cfg(all(feature = "qualification", feature = "cuda"))]
-        let chunk_parameters = if let FusedExecution::NativeChunk(chunk) = &execution {
+        let chunk_parameters = if let FusedExecution::NativeChunk(chunk)
+        | FusedExecution::Temporal(chunk) = &execution
+        {
             Some(
                 (0..tokens_per_row)
                     .map(|step| {
@@ -454,72 +572,121 @@
             workspace.batch = None;
         }
         let batch_budget = BatchProgram::preflight(&workspace.program, rows.len())?;
-        if workspace.batch.is_none()
-            && batch_budget.owned_bytes()? > self.service.remaining_capacity()?
-        {
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let needs_temporal = temporal
+            && workspace
+                .batch
+                .as_ref()
+                .is_none_or(|batch| batch.temporal.is_none());
+        #[cfg(not(all(feature = "qualification", feature = "cuda")))]
+        let needs_temporal = false;
+        let base_bytes = if workspace.batch.is_none() {
+            batch_budget.owned_bytes()?
+        } else {
+            0
+        };
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let cold_bytes = base_bytes
+            .checked_add(if needs_temporal {
+                crate::resident_program::TemporalProgram::preflight(&workspace.program)?
+                    .owned_bytes()?
+            } else {
+                0
+            })
+            .ok_or_else(|| Error::Limit("temporal complete cold admission".into()))?;
+        #[cfg(not(all(feature = "qualification", feature = "cuda")))]
+        let cold_bytes = base_bytes;
+        if cold_bytes > self.service.remaining_capacity()? {
             return Err(Error::Limit(
-                "native resident batch workspace admission exhausted".into(),
+                "native combined workspace admission exhausted".into(),
             ));
         }
-        if workspace.batch.is_none() {
+        if workspace.batch.is_none() || needs_temporal {
             slot.begin_submission()?;
             let mut initialization = self.service.begin_initialization(bases[0].root())?;
             let mut leases = Vec::new();
-            let program = BatchProgram::prepare(
-                self.service.client(),
-                &workspace.program,
-                rows.len(),
-                |role, count, data| {
-                    let role = match role {
-                        ResidentBufferRole::Source => BufferRole::ImmutableWeights,
-                        ResidentBufferRole::Value | ResidentBufferRole::Parameter => {
-                            BufferRole::Activation
-                        }
-                        ResidentBufferRole::Workspace | ResidentBufferRole::Status => {
-                            BufferRole::Scratch
-                        }
-                    };
-                    let lease = match data {
-                        AllocationInit::Empty => initialization.allocate_empty(count, role)?,
-                        AllocationInit::Upload(data) => {
-                            if data.len() != count {
-                                return Err(Error::State("native batch initializer length".into()));
-                            }
-                            initialization.upload_bytes(data, role)?
-                        }
-                    };
-                    let handle = lease.handle();
-                    leases.push((lease, DeviceAccess::Write));
-                    Ok(handle)
-                },
-            )?;
-            if program.rows() != rows.len()
-                || program.budget().owned_bytes()? != batch_budget.owned_bytes()?
-            {
+            let program = if workspace.batch.is_none() {
+                Some(BatchProgram::prepare(
+                    self.service.client(),
+                    &workspace.program,
+                    rows.len(),
+                    |role, count, data| {
+                        allocate_batch_buffer(&mut initialization, &mut leases, role, count, data)
+                    },
+                )?)
+            } else {
+                None
+            };
+            if let Some(program) = &program {
+                if program.rows() != rows.len()
+                    || program.budget().owned_bytes()? != batch_budget.owned_bytes()?
+                {
+                    return Err(Error::State(
+                        "native prepared geometry/budget changed".into(),
+                    ));
+                }
+                verify_batch_allocations(
+                    batch_budget.owned_bytes()?,
+                    program.owned_handles(),
+                    &leases,
+                )?;
+            }
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let mut temporal_leases = Vec::new();
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let companion = if needs_temporal {
+                let causal = program
+                    .as_ref()
+                    .unwrap_or_else(|| &workspace.batch.as_ref().unwrap().program);
+                let companion = crate::resident_program::TemporalProgram::prepare(
+                    self.service.client(),
+                    &workspace.program,
+                    causal,
+                    |role, count, data| {
+                        allocate_batch_buffer(
+                            &mut initialization,
+                            &mut temporal_leases,
+                            role,
+                            count,
+                            data,
+                        )
+                    },
+                )?;
+                verify_batch_allocations(
+                    crate::resident_program::TemporalProgram::preflight(&workspace.program)?
+                        .owned_bytes()?,
+                    companion.owned_handles(),
+                    &temporal_leases,
+                )?;
+                Some(companion)
+            } else {
+                None
+            };
+            let allocated = leases.len();
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let allocated = allocated + temporal_leases.len();
+            let ready = initialization.complete_sync().await?;
+            if ready.outputs().len() != allocated {
                 return Err(Error::State(
-                    "native batch prepared geometry/budget changed".into(),
+                    "native combined initialization incomplete".into(),
                 ));
             }
-            for handle in program.owned_handles() {
-                lease_for(&leases, &handle)?;
-            }
-            if leases.iter().map(|(l, _)| l.bytes()).sum::<usize>() != batch_budget.owned_bytes()? {
-                return Err(Error::State("native batch allocation coverage".into()));
-            }
-            let ready = initialization.complete_sync().await?;
-            if ready.outputs().len() != leases.len() {
-                return Err(Error::State(
-                    "native batch initialization incomplete".into(),
-                ));
-            }
-            workspace.batch = Some(BatchWorkspace {
-                program,
-                leases,
-                rows: rows.len(),
-            });
-            // Initialization reached a real terminal sync. Retire its slot,
-            // then release the workspace before waiting for execution: a B1
-            // worker selected meanwhile may need that same workspace lock.
+            if let Some(program) = program {
+                workspace.batch = Some(BatchWorkspace {
+                    program,
+                    leases,
+                    rows: rows.len(),
+                    #[cfg(all(feature = "qualification", feature = "cuda"))]
+                    temporal: None,
+                });
+            }
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            if let Some(program) = companion {
+                workspace.batch.as_mut().unwrap().temporal = Some(TemporalWorkspace {
+                    program,
+                    leases: temporal_leases,
+                });
+            }
             slot.finish()?;
             drop(workspace);
             let ticket = self.service.queue().submit(
@@ -540,9 +707,18 @@
             .validate_runtime(self.service.client(), &parameters, &bindings)?;
         #[cfg(all(feature = "qualification", feature = "cuda"))]
         if let Some(steps) = &chunk_parameters {
-            batch
-                .program
-                .validate_state_chunk(self.service.client(), steps, &bindings)?;
+            if temporal {
+                batch
+                    .temporal
+                    .as_ref()
+                    .ok_or_else(|| Error::State("temporal workspace absent".into()))?
+                    .program
+                    .validate(self.service.client(), &batch.program, steps, &bindings)?;
+            } else {
+                batch
+                    .program
+                    .validate_state_chunk(self.service.client(), steps, &bindings)?;
+            }
         }
         let device_buffers = device_tokens.map(|tokens| {
             tokens
@@ -578,6 +754,21 @@
             )?;
         }
         let handles = batch.program.execution_handles(output);
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let handles = if temporal {
+            let mut handles = handles;
+            handles.extend(
+                batch
+                    .temporal
+                    .as_ref()
+                    .ok_or_else(|| Error::State("temporal workspace absent".into()))?
+                    .program
+                    .execution_handles(),
+            );
+            handles
+        } else {
+            handles
+        };
         let mut accesses = workspace
             .leases
             .iter()
@@ -585,6 +776,19 @@
             .filter(|(l, _)| handles.iter().any(|h| same_handle(h, &l.handle())))
             .cloned()
             .collect::<Vec<_>>();
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if temporal {
+            accesses.extend(
+                batch
+                    .temporal
+                    .as_ref()
+                    .unwrap()
+                    .leases
+                    .iter()
+                    .filter(|(lease, _)| handles.iter().any(|h| same_handle(h, &lease.handle())))
+                    .cloned(),
+            );
+        }
         for handle in handles {
             lease_for(&accesses, &handle)?;
         }
@@ -631,8 +835,23 @@
             }
             flight
         };
-        let program = &mut workspace.batch.as_mut().unwrap().program;
+        let batch = workspace.batch.as_mut().unwrap();
+        let program = &mut batch.program;
         let mut submitted = match &execution {
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::Temporal(_) => batch
+                .temporal
+                .as_ref()
+                .ok_or_else(|| Error::State("temporal workspace retired".into()))?
+                .program
+                .launch(
+                    self.service.client(),
+                    program,
+                    chunk_parameters
+                        .as_ref()
+                        .ok_or_else(|| Error::State("temporal parameters absent".into()))?,
+                    &bindings,
+                )?,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
             FusedExecution::Graph(graph) => graph.launch(program, &parameters, &bindings)?,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk/temporal.rs
@@ -0,0 +1,619 @@
+//! Closed ROOT-only actual B2/T2 stateless grouping; no serving path.
+use super::*;
+use crate::resident_program::TemporalProgram;
+
+async fn loaded(name: &str) -> Result<(Arc<LoadedQwenModel>, Arc<ResidentModel>, u32)> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "temporal physical qualifier requires release".into(),
+        ));
+    }
+    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
+        .join("../../target/models")
+        .join(name);
+    let source = LoadedQwenModel::open(&path.join("model.gguf"), &path.join("config.json"))?;
+    let eos = QwenTokenizer::from_file(&path.join("tokenizer.json"))?.eos();
+    let plan = source.forward_plan(512, true, true)?;
+    let model = ResidentModel::load_inner(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        source.clone(),
+        plan,
+        PROVIDER,
+        DEVICE_CAP,
+        true,
+    )
+    .await?;
+    assert_eq!(model.states.len(), 42);
+    assert_eq!(model.vocab, 248320);
+    model.service.pinned_cuda_queue()?;
+    Ok((source, model, eos))
+}
+
+async fn temporal(
+    model: &Arc<ResidentModel>,
+    sessions: &[ResidentSession],
+    teachers: &[Vec<u32>],
+) -> Result<Option<PreparedResidentCohort>> {
+    let rows = sessions
+        .iter()
+        .zip(teachers)
+        .map(|(session, tokens)| (session, tokens.as_slice()))
+        .collect::<Vec<_>>();
+    model.prepare_state_temporal_b2_t2_admitted(&rows).await
+}
+
+/// Private packed intermediate read AFTER actual terminal Ready, under original
+/// leases/queue/transfer admission. It issues no session root or future Ready.
+async fn first_state(
+    model: &Arc<ResidentModel>,
+    root: u64,
+    expected: &[Vec<(StateSlot, Vec<f32>)>],
+    positions: &[usize],
+) -> Result<()> {
+    assert_eq!(expected.len(), 2);
+    assert_eq!(positions.len(), 2);
+    let _allocation = model.allocation.lock().await;
+    let workspace = model.workspace.lock().await;
+    let batch = workspace.batch.as_ref().unwrap();
+    let fields = TemporalProgram::first_state_buffers(&batch.program);
+    assert_eq!(fields.len(), 42);
+    let handles = fields
+        .iter()
+        .flat_map(|(_, state)| state.handles())
+        .collect::<Vec<_>>();
+    let leases = handles
+        .iter()
+        .map(|handle| lease_for(&batch.leases, handle).cloned())
+        .collect::<Result<Vec<_>>>()?;
+    let accesses = leases
+        .iter()
+        .cloned()
+        .map(|lease| (lease, DeviceAccess::Read))
+        .collect::<Vec<_>>();
+    let reads = leases
+        .iter()
+        .map(|lease| lease.readback(0, lease.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let bytes = leases.iter().try_fold(0u64, |n, lease| {
+        n.checked_add(lease.bytes() as u64)
+            .ok_or_else(|| Error::Limit("temporal intermediate staging".into()))
+    })?;
+    drop(workspace);
+    let ticket = model.service.queue().submit(
+        model.service.budget(),
+        model.service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::TransferStaging,
+            bytes,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    let _workspace = model.workspace.lock().await;
+    slot.begin_submission()?;
+    let flight = model.service.submit(root, &accesses)?;
+    let data = flight
+        .complete_read_only_readback(&reads, |data| {
+            for raw in data {
+                if words(raw)?.iter().any(|x| !x.is_finite()) {
+                    return Err(Error::State("temporal intermediate nonfinite".into()));
+                }
+            }
+            Ok(())
+        })
+        .await?;
+    assert_eq!(data.len(), handles.len());
+    let mut cursor = 0;
+    for (domain, state) in fields {
+        let mut actual = vec![Vec::new(), Vec::new()];
+        for value in state.buffers() {
+            let flat = words(&data[cursor])?;
+            cursor += 1;
+            assert_eq!(flat.len(), value.elements);
+            match state {
+                StateBuffers::Vector(_) => {
+                    let count = flat.len() / 2;
+                    assert_eq!(count * 2, flat.len());
+                    for row in 0..2 {
+                        actual[row].extend_from_slice(&flat[row * count..(row + 1) * count]);
+                    }
+                }
+                StateBuffers::Kv { .. } => {
+                    let slab = flat.len() / (2 * 512);
+                    assert_eq!(slab * 2 * 512, flat.len());
+                    for row in 0..2 {
+                        for token in 0..512 {
+                            let span =
+                                &flat[(token * 2 + row) * slab..(token * 2 + row + 1) * slab];
+                            if token < positions[row] {
+                                actual[row].extend_from_slice(span);
+                            } else {
+                                assert!(
+                                    span.iter().all(|x| x.to_bits() == 0),
+                                    "actual T0 canonical KV tail"
+                                );
+                            }
+                        }
+                    }
+                }
+            }
+        }
+        for row in 0..2 {
+            let wanted = expected[row]
+                .iter()
+                .find(|(slot, _)| *slot == domain)
+                .unwrap();
+            exact(
+                &wanted.1,
+                &actual[row],
+                "temporal native T0 all42 original packed state",
+            )?;
+        }
+    }
+    assert_eq!(cursor, data.len());
+    drop(data);
+    slot.finish()?;
+    assert_eq!(model.service.pending_events()?, 0);
+    Ok(())
+}
+async fn references(
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+    teachers: &[Vec<u32>],
+) -> Result<Vec<Vec<(StateSlot, Vec<f32>)>>> {
+    let mut first = Vec::new();
+    for step in 0..2 {
+        for row in 0..2 {
+            let expected = cpu[row].forward(teachers[row][step])?;
+            exact(
+                &expected,
+                &single[row]
+                    .forward(&[teachers[row][step]], false)
+                    .await?
+                    .logits,
+                "temporal each teacher CPU/B1 fullV",
+            )?;
+        }
+        state(single, cpu).await?;
+        if step == 0 {
+            first = cpu
+                .iter()
+                .map(|cpu| cpu.snapshot().domain_values())
+                .collect();
+        }
+    }
+    Ok(first)
+}
+async fn run(name: &str) -> Result<()> {
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![55, eos], vec![101, eos]];
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let queue = model.service.queue().stats()?;
+    for bad in [
+        vec![vec![55], vec![101, eos]],
+        vec![vec![55, eos], vec![101, u32::MAX]],
+        vec![vec![55, eos, 78, 135]; 2],
+    ] {
+        assert!(temporal(&model, &native, &bad).await.is_err());
+    }
+    assert!(
+        model
+            .prepare_state_temporal_b2_t2_admitted(&[
+                (&native[0], teachers[0].as_slice()),
+                (&native[0], teachers[1].as_slice())
+            ])
+            .await
+            .is_err()
+    );
+    assert_eq!(model.service.queue().stats()?, queue);
+    unchanged(&native, &before)?;
+    let uploads = model.service.uploaded_value_bytes();
+    let abandoned = temporal(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&abandoned, &before);
+    drop(abandoned);
+    unchanged_domains(&native, &domains).await?;
+    let warm = model.service.resident_bytes()?;
+    drop(temporal(&model, &native, &teachers).await?.unwrap());
+    idle(&model, warm)?;
+    let saved = native[1].snapshot();
+    let stale = temporal(&model, &native, &teachers).await?.unwrap();
+    native[1].restore(&saved)?;
+    let restored = bindings(&native);
+    assert!(
+        stale
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    unchanged(&native, &restored)?;
+    unchanged_domains(&native, &domains).await?;
+    drop(saved);
+    let before = bindings(&native);
+    let candidate = temporal(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&candidate, &before);
+    let dispatches = candidate.dispatches();
+    let projections = candidate.fused_dispatches();
+    {
+        let workspace = model.workspace.lock().await;
+        let batch = workspace.batch.as_ref().unwrap();
+        assert_eq!(batch.rows, 2);
+        let companion = batch.temporal.as_ref().unwrap();
+        assert_eq!(projections, companion.program.expected_native_projections());
+        assert!(projections > 0);
+        assert_eq!(
+            batch.owned_bytes()?,
+            batch.program.budget().owned_bytes()? + companion.program.budget().owned_bytes()?
+        );
+    }
+    let first = references(&mut single, &mut cpu, &teachers).await?;
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    chunk_reads(&reads, 2, 2, dispatches);
+    for (session, old) in native.iter().zip(&before) {
+        assert_eq!(session.binding().root(), old.root() + 1);
+        assert_eq!(session.binding().frontier(), old.frontier() + 2);
+    }
+    first_state(
+        &model,
+        native[0].binding().root(),
+        &first,
+        &before
+            .iter()
+            .map(|b| b.frontier() as usize + 1)
+            .collect::<Vec<_>>(),
+    )
+    .await?;
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    assert_eq!(model.service.uploaded_value_bytes(), uploads);
+    drop(reads);
+    future(&model, &mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(first);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_TEMPORAL_B2_T2_PASS source={name} roots=2 physical_stateless_rows=4 native_T0_T1_all42_exact=true CPU_B1_each_teacher_fullV=true future_fullV_greedy_exact=true EOS_teacher_no_output=true transaction_status_bytes=4 source_launch_sites={dispatches} source_projection_dot_sites={projections} final_reserved=0"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "ROOT release actual original temporal B2/T2; --test-threads=1"]
+fn qwen08_temporal_b2_t2() {
+    futures_lite::future::block_on(run("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "ROOT release actual original temporal B2/T2; --test-threads=1"]
+fn qwen2_temporal_b2_t2() {
+    futures_lite::future::block_on(run("qwen35-2b")).unwrap();
+}
+
+#[cfg(feature = "hardware-abort-probe")]
+async fn temporal_abort(name: &str) -> Result<()> {
+    use std::sync::atomic::Ordering;
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![11, eos], vec![353, eos]];
+    let before = bindings(&native);
+    let saved = native
+        .iter()
+        .map(ResidentSession::snapshot)
+        .collect::<Vec<_>>();
+    let domains = domain_values(&native).await?;
+    let ready = temporal(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&ready, &before);
+    let client = model.service.client().clone();
+    let old_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    model.launch_seen.store(0, Ordering::Release);
+    model.abort_after_launch.store(1, Ordering::Release);
+    let mut pending = Box::pin(temporal(&model, &native, &teachers));
+    loop {
+        if let Some(result) = futures_lite::future::poll_once(pending.as_mut()).await {
+            return Err(Error::State(format!(
+                "chunk abort completed before original hook: success={}",
+                result.is_ok()
+            )));
+        }
+        if model.launch_seen.load(Ordering::Acquire) != 0 {
+            break;
+        }
+        futures_lite::future::yield_now().await;
+    }
+    drop(pending);
+    let handles = model.launch_seen.load(Ordering::Acquire);
+    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(handles > 0 && quarantined >= old_quarantine + handles);
+    assert!(model.service.ensure_usable().is_err());
+    assert!(
+        ready
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    unchanged(&native, &before)?;
+    assert!(temporal(&model, &native, &teachers).await.is_err());
+    assert!(native[0].prepare(&[11], false).await.is_err());
+    assert!(
+        ResidentService::new(
+            Device::Cuda(cubecl::device::CudaDevice::new(0)),
+            source.source_identity(),
+            ResourceOwner::new("other-model", "after-chunk-abort")?,
+            1024 * 1024,
+            DEVICE_CAP
+        )
+        .is_err()
+    );
+    client.sync().await.map_err(|error| {
+        Error::State(format!(
+            "chunk abort independent prior readback barrier:{error}"
+        ))
+    })?;
+    for (snapshot, expected) in saved.iter().zip(&domains) {
+        physical_prior_after_abort(&client, snapshot, expected).await?;
+    }
+    unchanged(&native, &before)?;
+    let service = Arc::clone(&model.service);
+    let key = service.physical_key();
+    drop(saved);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    assert!(budget.fenced);
+    assert_eq!(budget.reserved, PROVIDER as u64);
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantined);
+    println!(
+        "P12_TEMPORAL_B2_T2_ABORT_PASS source={name} B=2 T=2 actual_original_queued_chunk=true one_terminal_Ready_not_reached=true all42_prior_bits_exact=true original_handles={handles} quarantine_retained=true service_fenced=true recovery=false"
+    );
+    Ok(())
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "unknown completion fences actual card; ROOT separate --release process"]
+fn qwen08_temporal_abort() {
+    futures_lite::future::block_on(temporal_abort("qwen35-0.8b")).unwrap();
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "unknown completion fences actual card; ROOT separate --release process"]
+fn qwen2_temporal_abort() {
+    futures_lite::future::block_on(temporal_abort("qwen35-2b")).unwrap();
+}
+
+#[cfg(feature = "portable-checkpoint")]
+async fn temporal_boundary(name: &str) -> Result<()> {
+    let steps = 2;
+    let (source, model, _) = loaded(name).await?;
+    for (scope, positions) in [
+        ("cross256", [257 - steps, 258 - steps]),
+        ("capacity512", [512 - steps, 512 - steps]),
+    ] {
+        let mut cpu = Vec::new();
+        let mut single = Vec::new();
+        let mut native = Vec::new();
+        let mut teachers = Vec::new();
+        for (row, frontier) in positions.into_iter().enumerate() {
+            let stream = (0..512)
+                .map(|position| IDS[(position + row * 3) % IDS.len()])
+                .collect::<Vec<_>>();
+            let mut reference = QwenDenseCpu::from_shared(Arc::clone(&source))?;
+            reference.prefill_chunk_without_logits(&stream[..frontier])?;
+            let checkpoint = reference.checkpoint()?;
+            let domains = reference.snapshot().domain_values();
+            let rebuilt = source.checkpoint_from_domains(frontier, &domains)?;
+            compare_domains(&domains, &checkpoint.domain_values())?;
+            compare_domains(&domains, &rebuilt.domain_values())?;
+            let mut batch = model.session().await?;
+            let mut b1 = model.session().await?;
+            batch.import_checkpoint(&rebuilt).await?;
+            b1.import_checkpoint(&checkpoint).await?;
+            compare_domains(&domains, &batch.snapshot().read_domains().await?)?;
+            compare_domains(&domains, &b1.snapshot().read_domains().await?)?;
+            teachers.push(stream[frontier..frontier + steps].to_vec());
+            cpu.push(reference);
+            single.push(b1);
+            native.push(batch);
+        }
+        let before = bindings(&native);
+        let uploads = model.service.uploaded_value_bytes();
+        advance_oracle(&mut single, &mut cpu, &teachers).await?;
+        let candidate = temporal(&model, &native, &teachers).await?.unwrap();
+        state_receipt(&candidate, &before);
+        let dispatches = candidate.dispatches();
+        let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+        chunk_reads(&reads, 2, steps, dispatches);
+        state(&native, &cpu).await?;
+        state(&single, &cpu).await?;
+        for (session, old) in native.iter().zip(&before) {
+            assert_eq!(session.binding().frontier(), old.frontier() + steps as u64);
+            assert_eq!(session.binding().root(), old.root() + 1);
+        }
+        assert_eq!(model.service.uploaded_value_bytes(), uploads);
+        if scope == "cross256" {
+            future(&model, &mut native, &mut single, &mut cpu).await?;
+        } else {
+            assert!(
+                native
+                    .iter()
+                    .all(|session| session.binding().frontier() == 512)
+            );
+            let before = bindings(&native);
+            let domains = domain_values(&native).await?;
+            let queue = model.service.queue().stats()?;
+            let bytes = model.service.resident_bytes()?;
+            assert!(temporal(&model, &native, &teachers).await.is_err());
+            unchanged(&native, &before)?;
+            unchanged_domains(&native, &domains).await?;
+            assert_eq!(model.service.queue().stats()?, queue);
+            assert_eq!(model.service.resident_bytes()?, bytes);
+        }
+        println!(
+            "P12_TEMPORAL_B2_T2_BOUNDARY_SCOPE source={name} B=2 T={steps} scope={scope} actual_CPU_hydrated_prefix=true GPU_complete_prefill_claim=false each_teacher_CPU_B1_fullV_all42_exact=true native_terminal_all42_exact=true status_bytes=4"
+        );
+    }
+    reclaim(model).await?;
+    Ok(())
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen08_temporal_boundary() {
+    futures_lite::future::block_on(temporal_boundary("qwen35-0.8b")).unwrap();
+}
+
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen2_temporal_boundary() {
+    futures_lite::future::block_on(temporal_boundary("qwen35-2b")).unwrap();
+}
+
+async fn temporal_pressure(name: &str) -> Result<()> {
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![55, eos], vec![101, eos]];
+    let need = {
+        let workspace = model.workspace.lock().await;
+        assert!(workspace.batch.is_none());
+        BatchProgram::preflight(&workspace.program, 2)?
+            .owned_bytes()?
+            .checked_add(TemporalProgram::preflight(&workspace.program)?.owned_bytes()?)
+            .and_then(|n| n.checked_add(model.state_bytes * 2))
+            .ok_or_else(|| Error::Limit("temporal pressure combined bound".into()))?
+    };
+    let available = model.service.remaining_capacity()?;
+    assert!(need > 0 && available >= need);
+    let pressure = model
+        .service
+        .allocate_uninitialized(available - (need - 1), BufferRole::Scratch)?;
+    let roots = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let bytes = model.service.resident_bytes()?;
+    let queue = model.service.queue().stats()?;
+    assert!(temporal(&model, &native, &teachers).await?.is_none());
+    assert_eq!(model.service.resident_bytes()?, bytes);
+    assert_eq!(model.service.queue().stats()?, queue);
+    assert!(model.workspace.lock().await.batch.is_none());
+    unchanged(&native, &roots)?;
+    unchanged_domains(&native, &domains).await?;
+    drop(pressure);
+    let candidate = temporal(&model, &native, &teachers).await?.unwrap();
+    let dispatches = candidate.dispatches();
+    references(&mut single, &mut cpu, &teachers).await?;
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    chunk_reads(&reads, 2, 2, dispatches);
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    drop(reads);
+    future(&model, &mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_TEMPORAL_B2_T2_PRESSURE_PASS source={name} true_combined_P2_P4_budget_None_before_queue=true no_hidden_fallback=true CPU_B1_final42_future_fullV_greedy=true final_reserved=0"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "ROOT release physical combined B2+B4 pressure; --test-threads=1"]
+fn qwen08_temporal_pressure() {
+    futures_lite::future::block_on(temporal_pressure("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "ROOT release physical combined B2+B4 pressure; --test-threads=1"]
+fn qwen2_temporal_pressure() {
+    futures_lite::future::block_on(temporal_pressure("qwen35-2b")).unwrap();
+}
+
+async fn temporal_certificate_refusal(name: &str) -> Result<()> {
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![11, eos], vec![353, eos]];
+    let before = bindings(&native);
+    let saved = native
+        .iter()
+        .map(ResidentSession::snapshot)
+        .collect::<Vec<_>>();
+    let domains = domain_values(&native).await?;
+    let ready = temporal(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&ready, &before);
+    let client = model.service.client().clone();
+    let old_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    // Synthetic one-shot NaN in intermediate inactive KV only. The following
+    // teacher erases it; an omitted intermediate certificate/reset OR would pass.
+    TemporalProgram::arm_inactive_nan_probe();
+    assert!(
+        matches!(temporal(&model,&native,&teachers).await,Err(Error::State(ref message))if message.contains("status refused"))
+    );
+    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(quarantined > old_quarantine);
+    assert!(model.service.ensure_usable().is_err());
+    unchanged(&native, &before)?;
+    assert!(
+        ready
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    // Read only the original retained handles after the actual failed readback.
+    // This diagnostic does not promote an event, make Ready, recover or publish.
+    let (status, padding) = {
+        let workspace = model.workspace.lock().await;
+        let program = &workspace.batch.as_ref().unwrap().program;
+        (program.status(), program.chunk_even_final_padding_view()?)
+    };
+    let raw = client
+        .read_async(vec![status, padding])
+        .await
+        .map_err(|error| Error::State(format!("chunk certificate raw proof:{error}")))?;
+    assert_eq!(raw.len(), 2);
+    assert_eq!(raw[0].len(), 4);
+    assert_eq!(raw[1].len(), 4);
+    assert_eq!(
+        u32::from_ne_bytes(raw[0][..].try_into().unwrap()),
+        crate::program_kernels::NONFINITE_VALUE
+    );
+    assert_eq!(
+        u32::from_ne_bytes(raw[1][..].try_into().unwrap()),
+        0,
+        "later append must heal final padding; rejection proves earlier certificate latch"
+    );
+    for (snapshot, expected) in saved.iter().zip(&domains) {
+        physical_prior_after_abort(&client, snapshot, expected).await?;
+    }
+    assert!(temporal(&model, &native, &teachers).await.is_err());
+    unchanged(&native, &before)?;
+    let service = Arc::clone(&model.service);
+    let key = service.physical_key();
+    drop(raw);
+    drop(saved);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    assert!(budget.fenced);
+    assert_eq!(budget.reserved, PROVIDER as u64);
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantined);
+    println!(
+        "P12_TEMPORAL_B2_T2_CERTIFICATE_REFUSAL source={name} B=2 T=2 synthetic_intermediate_inactive_KV_nan=true final_padding_raw_positive_zero=true actual_latched_status=2 original_roots_all42_unchanged=true quarantine_retained=true service_fenced=true recovery=false"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "actual numerical-status refusal fences card; ROOT fresh --release process"]
+fn qwen08_temporal_certificate_refusal() {
+    futures_lite::future::block_on(temporal_certificate_refusal("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "actual numerical-status refusal fences card; ROOT fresh --release process"]
+fn qwen2_temporal_certificate_refusal() {
+    futures_lite::future::block_on(temporal_certificate_refusal("qwen35-2b")).unwrap();
+}
--- a/crates/zerogravity-gpu/src/resident_inference.rs
+++ b/crates/zerogravity-gpu/src/resident_inference.rs
@@ -110,6 +110,29 @@
     program: BatchProgram,
     leases: Vec<(ResidentLease, DeviceAccess)>,
     rows: usize,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    temporal: Option<TemporalWorkspace>,
+}
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+struct TemporalWorkspace {
+    program: crate::resident_program::TemporalProgram,
+    leases: Vec<(ResidentLease, DeviceAccess)>,
+}
+impl BatchWorkspace {
+    fn owned_bytes(&self) -> Result<usize> {
+        let bytes = self.program.budget().owned_bytes()?;
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let bytes = bytes
+            .checked_add(
+                self.temporal
+                    .as_ref()
+                    .map(|t| t.program.budget().owned_bytes())
+                    .transpose()?
+                    .unwrap_or(0),
+            )
+            .ok_or_else(|| Error::Limit("temporal complete workspace budget".into()))?;
+        Ok(bytes)
+    }
 }
 /// Shared immutable source and serialized physical workspace, independent roots.
 pub struct ResidentModel {
--- a/crates/zerogravity-gpu/src/resident_program/batch/chunk.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch/chunk.rs
@@ -262,7 +262,11 @@
 
     /// Same two scalar-token/position kernels as original launch_inputs; no
     /// clear, gather, upload, device-token feedback, sampler or graph enqueue.
-    fn stage_teacher_parameters(&self, client: &Client, parameters: &[RuntimeParameters]) -> usize {
+    pub(super) fn stage_teacher_parameters(
+        &self,
+        client: &Client,
+        parameters: &[RuntimeParameters],
+    ) -> usize {
         let mut tokens = [0u32; 8];
         let mut positions = [0u32; 8];
         for ((token, position), parameter) in tokens.iter_mut().zip(&mut positions).zip(parameters)
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/temporal.rs
@@ -0,0 +1,911 @@
+//! Closed B2/T2 dependency lowering. Physical activation rows are NOT roots.
+use super::*;
+use std::collections::BTreeSet;
+use std::ops::Range;
+
+#[cfg(test)]
+std::thread_local! {
+    static INACTIVE_NAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
+}
+
+#[derive(Clone, Debug, PartialEq, Eq)]
+enum Stage {
+    Flat(usize),
+    Causal {
+        nodes: Range<usize>,
+        slots: Vec<StateSlot>,
+    },
+}
+
+/// Derived from the SAME immutable plan; no token/layer schedule is copied.
+#[derive(Clone, Debug)]
+struct Dependencies {
+    stages: Vec<Stage>,
+    flat: BTreeSet<usize>,
+    causal: BTreeSet<usize>,
+    count: usize,
+}
+impl Dependencies {
+    fn new(plan: &ForwardPlan) -> Result<Self> {
+        ResidentProgram::require_legacy_contract(plan)?;
+        let result = Self::nodes(plan.nodes(), plan.state_nodes().len(), plan.values().len())?;
+        let actual = result
+            .causal
+            .iter()
+            .filter_map(|&index| plan.nodes()[index].state.map(|effect| effect.write))
+            .collect::<Vec<_>>();
+        if actual.len() != plan.proposed_domains().len()
+            || plan
+                .proposed_domains()
+                .iter()
+                .any(|wanted| actual.iter().filter(|actual| *actual == wanted).count() != 1)
+        {
+            return Err(Error::State("temporal all-domain effect closure".into()));
+        }
+        Ok(result)
+    }
+
+    fn nodes(nodes: &[PlanNode], count: usize, values: usize) -> Result<Self> {
+        if count == 0 || count > nodes.len() || nodes.len() > 32768 || values > 65536 {
+            return Err(Error::Limit("temporal plan metadata bound".into()));
+        }
+        let mut stages = Vec::new();
+        let mut flat = BTreeSet::new();
+        let mut causal = BTreeSet::new();
+        let mut produced = BTreeSet::new();
+        let mut advanced = BTreeSet::new();
+        let mut index = 0;
+        while index < count {
+            let node = &nodes[index];
+            let stage = match &node.operation {
+                PlanOperation::KvPrefixGuard { state, .. } => {
+                    if state.ordinal != 0 || advanced.contains(&state.slot) {
+                        return Err(Error::State("temporal KV prior version".into()));
+                    }
+                    // Find the actual full-plan region, then clip at the
+                    // original StateOnly cut (which can end just after append).
+                    let end = nodes[index + 1..].iter().position(|node| {
+                        matches!(node.operation, PlanOperation::AttentionValue { state: read, .. }
+                            if read.slot == state.slot && read.ordinal == 1)
+                    }).map(|offset| index + offset + 2)
+                        .ok_or_else(|| Error::Unsupported("temporal whole attention region".into()))?;
+                    let mut writes = 0;
+                    for entry in &nodes[index..end] {
+                        Self::known(entry)?;
+                        if let Some(effect) = entry.state {
+                            if !matches!(entry.operation, PlanOperation::AppendKv)
+                                || effect.read.slot != state.slot
+                            {
+                                return Err(Error::Unsupported(
+                                    "temporal interleaved attention effect".into(),
+                                ));
+                            }
+                            Self::effect(entry, &mut advanced)?;
+                            writes += 1;
+                        }
+                        if matches!(
+                            entry.operation,
+                            PlanOperation::AttentionQk { .. }
+                                | PlanOperation::AttentionValue { .. }
+                        ) && (writes != 1
+                            || Self::reads(entry).iter().any(|read| read.ordinal != 1))
+                        {
+                            return Err(Error::State(
+                                "temporal attention must read actual proposed snapshot".into(),
+                            ));
+                        }
+                        if matches!(entry.operation, PlanOperation::KvPrefixGuard { .. })
+                            && !std::ptr::eq(entry, &nodes[index])
+                        {
+                            return Err(Error::Unsupported(
+                                "temporal nested attention guard".into(),
+                            ));
+                        }
+                        if Self::reads(entry)
+                            .iter()
+                            .any(|read| read.slot != state.slot || read.ordinal > 1)
+                        {
+                            return Err(Error::Unsupported(
+                                "temporal foreign attention state read".into(),
+                            ));
+                        }
+                    }
+                    if writes != 1 {
+                        return Err(Error::State("temporal attention append coverage".into()));
+                    }
+                    Stage::Causal {
+                        nodes: index..end.min(count),
+                        slots: vec![state.slot],
+                    }
+                }
+                PlanOperation::Conv { .. } | PlanOperation::Recurrent { .. } => {
+                    Self::known(node)?;
+                    let slot = Self::effect(node, &mut advanced)?;
+                    Stage::Causal {
+                        nodes: index..index + 1,
+                        slots: vec![slot],
+                    }
+                }
+                PlanOperation::ObserveState { state, .. } => {
+                    if node.state.is_some() || state.ordinal != 1 || !advanced.contains(&state.slot)
+                    {
+                        return Err(Error::Unsupported(
+                            "temporal observation requires retained proposed version".into(),
+                        ));
+                    }
+                    Stage::Causal {
+                        nodes: index..index + 1,
+                        slots: vec![state.slot],
+                    }
+                }
+                _ => {
+                    Self::known(node)?;
+                    if node.state.is_some()
+                        || !Self::reads(node).is_empty()
+                        || matches!(
+                            node.operation,
+                            PlanOperation::AppendKv | PlanOperation::AttentionSoftmax { .. }
+                        )
+                    {
+                        return Err(Error::Unsupported(
+                            "temporal hidden causal dependency".into(),
+                        ));
+                    }
+                    Stage::Flat(index)
+                }
+            };
+            let range = match &stage {
+                Stage::Flat(i) => *i..*i + 1,
+                Stage::Causal { nodes, .. } => nodes.clone(),
+            };
+            // SSA closure is checked independently of the state/effect tags.
+            for (i, node) in nodes[range.clone()].iter().enumerate() {
+                if node
+                    .inputs
+                    .iter()
+                    .any(|id| id.0 as usize >= values || !produced.contains(&id.0))
+                    || node
+                        .outputs
+                        .iter()
+                        .any(|id| id.0 as usize >= values || produced.contains(&id.0))
+                {
+                    return Err(Error::State(
+                        "temporal value dependency/producer closure".into(),
+                    ));
+                }
+                for id in &node.outputs {
+                    produced.insert(id.0);
+                }
+                if matches!(stage, Stage::Flat(_)) {
+                    flat.insert(range.start + i);
+                } else {
+                    causal.insert(range.start + i);
+                }
+            }
+            index = range.end;
+            stages.push(stage);
+        }
+        Ok(Self {
+            stages,
+            flat,
+            causal,
+            count,
+        })
+    }
+    fn effect(node: &PlanNode, advanced: &mut BTreeSet<StateSlot>) -> Result<StateSlot> {
+        let effect = node
+            .state
+            .ok_or_else(|| Error::State("temporal effect absent".into()))?;
+        if effect.read.slot != effect.write.slot
+            || effect.read.ordinal != 0
+            || effect.write.ordinal != 1
+            || !advanced.insert(effect.read.slot)
+        {
+            return Err(Error::State(
+                "temporal repeated/noncanonical state effect".into(),
+            ));
+        }
+        if !matches!(
+            (&node.operation, effect.read.slot),
+            (PlanOperation::Conv { .. }, StateSlot::Conv(_))
+                | (PlanOperation::Recurrent { .. }, StateSlot::Recurrent(_))
+                | (PlanOperation::AppendKv, StateSlot::Kv(_))
+        ) {
+            return Err(Error::State("temporal operation/domain mismatch".into()));
+        }
+        Ok(effect.read.slot)
+    }
+    fn reads(node: &PlanNode) -> Vec<StateVersion> {
+        let mut reads = node
+            .state
+            .map(|effect| vec![effect.read])
+            .unwrap_or_default();
+        match node.operation {
+            PlanOperation::KvPrefixGuard { state, .. }
+            | PlanOperation::AttentionQk { state, .. }
+            | PlanOperation::AttentionValue { state, .. }
+            | PlanOperation::ObserveState { state, .. } => reads.push(state),
+            _ => {}
+        }
+        reads
+    }
+    fn known(node: &PlanNode) -> Result<()> {
+        match node.operation {
+            PlanOperation::Embed { .. }
+            | PlanOperation::Project { .. }
+            | PlanOperation::View(_)
+            | PlanOperation::Contiguous
+            | PlanOperation::Rms { .. }
+            | PlanOperation::Silu { .. }
+            | PlanOperation::Add
+            | PlanOperation::SigmoidGate
+            | PlanOperation::Conv { .. }
+            | PlanOperation::Coefficients { .. }
+            | PlanOperation::Recurrent { .. }
+            | PlanOperation::KvPrefixGuard { .. }
+            | PlanOperation::Rope { .. }
+            | PlanOperation::AppendKv
+            | PlanOperation::AttentionQk { .. }
+            | PlanOperation::AttentionSoftmax { .. }
+            | PlanOperation::AttentionValue { .. }
+            | PlanOperation::CheckFinite { .. }
+            | PlanOperation::Observe { .. }
+            | PlanOperation::ObserveState { .. } => Ok(()),
+            _ => Err(Error::Unsupported(
+                "temporal unqualified/new operation".into(),
+            )),
+        }
+    }
+}
+
+/// Fixed original B4 buffers/prebound quantizers. The enclosing workspace owns
+/// every lease, including unused B4 state/greedy allocations charged at cold load.
+pub(crate) struct TemporalProgram {
+    flat: BatchProgram,
+    dependencies: Dependencies,
+}
+struct Banks<'a> {
+    program: &'a mut BatchProgram,
+    slots: &'a [StateSlot],
+    flipped: bool,
+}
+impl Banks<'_> {
+    fn flip(&mut self) {
+        for binding in &mut self.program.packed {
+            if self.slots.contains(&binding.slot) {
+                std::mem::swap(&mut binding.prior, &mut binding.candidate);
+            }
+        }
+        self.flipped = !self.flipped;
+    }
+}
+impl Drop for Banks<'_> {
+    fn drop(&mut self) {
+        if self.flipped {
+            self.flip();
+        }
+    }
+}
+
+impl TemporalProgram {
+    #[cfg(test)]
+    pub(crate) fn arm_inactive_nan_probe() {
+        INACTIVE_NAN.with(|flag| assert!(!flag.replace(true), "temporal probe already armed"));
+    }
+    #[cfg(test)]
+    fn poison_intermediate(
+        client: &Client,
+        program: &BatchProgram,
+        slot: StateSlot,
+    ) -> Result<usize> {
+        let binding = program.packed.iter().find(|b| b.slot == slot).unwrap();
+        let StateBuffers::Kv { keys, .. } = &binding.candidate else {
+            return Ok(0);
+        };
+        let offset = keys
+            .elements
+            .checked_sub(1)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("temporal NaN offset".into()))?;
+        let start = keys
+            .handle
+            .offset_start
+            .unwrap_or(0)
+            .checked_add(offset as u64)
+            .ok_or_else(|| Error::Limit("temporal NaN start".into()))?;
+        let end = start
+            .checked_add(4)
+            .ok_or_else(|| Error::Limit("temporal NaN end".into()))?;
+        let mut view = keys.handle.clone();
+        view.offset_start = Some(start);
+        view.offset_end = Some(
+            view.size()
+                .checked_sub(end)
+                .ok_or_else(|| Error::State("temporal NaN extent".into()))?,
+        );
+        BufferRange::new(&view)?;
+        unsafe {
+            batch_tokens::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&view, 1),
+                0x7fc00001u32,
+                0,
+                0,
+                0,
+                0,
+                0,
+                0,
+                0,
+                1,
+            );
+        }
+        Ok(1)
+    }
+    pub(crate) fn preflight(original: &ResidentProgram) -> Result<ResidentPreflight> {
+        Dependencies::new(&original.plan)?;
+        if original.plan.prefix_capacity() != 512 {
+            return Err(Error::Unsupported(
+                "temporal original capacity512 only".into(),
+            ));
+        }
+        BatchProgram::preflight(original, 4)
+    }
+    pub(crate) fn prepare(
+        client: &Client,
+        original: &ResidentProgram,
+        causal: &BatchProgram,
+        allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
+    ) -> Result<Self> {
+        if causal.rows != 2
+            || causal.graph_pinned()
+            || !Arc::ptr_eq(&causal.inner.plan_lineage, &original.plan_lineage)
+        {
+            return Err(Error::State(
+                "temporal same original plan/B2 lineage".into(),
+            ));
+        }
+        Self::preflight(original)?;
+        let flat = BatchProgram::prepare(client, original, 4, allocate)?;
+        Ok(Self {
+            flat,
+            dependencies: Dependencies::new(&original.plan)?,
+        })
+    }
+    pub(crate) fn budget(&self) -> &ResidentPreflight {
+        self.flat.budget()
+    }
+    #[cfg(test)]
+    pub(crate) fn expected_native_projections(&self) -> usize {
+        self.dependencies
+            .flat
+            .iter()
+            .filter(|&&i| matches!(self.flat.nodes[i].operation, PlanOperation::Project { .. }))
+            .count()
+            + 2 * self
+                .dependencies
+                .causal
+                .iter()
+                .filter(|&&i| matches!(self.flat.nodes[i].operation, PlanOperation::Project { .. }))
+                .count()
+    }
+    /// Diagnostic metadata only; caller must retain actual leases and use the
+    /// original queue/read-only completion before inspecting T0's packed bank.
+    #[cfg(test)]
+    pub(crate) fn first_state_buffers(causal: &BatchProgram) -> Vec<(StateSlot, StateBuffers)> {
+        causal
+            .packed
+            .iter()
+            .map(|binding| (binding.slot, binding.candidate.clone()))
+            .collect()
+    }
+    pub(crate) fn owned_handles(&self) -> Vec<Handle> {
+        self.flat.owned_handles()
+    }
+    pub(crate) fn execution_handles(&self) -> Vec<Handle> {
+        self.flat.execution_handles(ResidentOutput::StateOnly)
+    }
+    pub(crate) fn validate(
+        &self,
+        client: &Client,
+        causal: &BatchProgram,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<()> {
+        if causal.rows != 2
+            || self.flat.rows != 4
+            || steps.len() != 2
+            || causal.graph_pinned()
+            || self.flat.graph_pinned()
+            || !Arc::ptr_eq(&self.flat.inner.plan_lineage, &causal.inner.plan_lineage)
+            || !Arc::ptr_eq(&self.flat.inner._source_owner, &causal.inner._source_owner)
+            || self.dependencies.count != causal.inner.plan.state_nodes().len()
+            || self.dependencies.flat.is_empty()
+            || self.dependencies.causal.is_empty()
+        {
+            return Err(Error::State("temporal closed B2/T2 layout/lineage".into()));
+        }
+        causal.validate_state_chunk(client, steps, bindings)?;
+        let causal_handles = causal
+            .execution_handles(ResidentOutput::StateOnly)
+            .into_iter()
+            .map(|h| BufferRange::new(&h))
+            .collect::<Result<Vec<_>>>()?;
+        for handle in self.flat.owned_handles() {
+            let range = BufferRange::new(&handle)?;
+            if range.service != client.service_id()
+                || causal_handles.iter().any(|other| range.overlaps(other))
+            {
+                return Err(Error::State(
+                    "temporal private layouts alias/foreign service".into(),
+                ));
+            }
+        }
+        DeviceBuffer::new(self.flat.inner.status.clone(), 1).check(client, 1)?;
+        DeviceBuffer::new(self.flat.inner.token.clone(), 4).check(client, 4)?;
+        DeviceBuffer::new(self.flat.positions.clone(), 4).check(client, 4)?;
+        for (index, shape) in causal.inner.plan.values().iter().enumerate() {
+            let words = shape.capacity_elements()?;
+            let small = &causal.inner.values[index];
+            let large = &self.flat.inner.values[index];
+            small.check(
+                client,
+                words
+                    .checked_mul(2)
+                    .ok_or_else(|| Error::Limit("temporal B2 value bound".into()))?,
+            )?;
+            large.check(
+                client,
+                words
+                    .checked_mul(4)
+                    .ok_or_else(|| Error::Limit("temporal B4 value bound".into()))?,
+            )?;
+            for teacher in 0..2 {
+                Self::bridge_span(small, large, teacher)?;
+            }
+        }
+        // Every actual stateless node is checked against the sealed partition.
+        // No fake B4 RootBindings, StateBindings or future Ready are constructed.
+        for &index in &self.dependencies.flat {
+            let node = &self.flat.nodes[index];
+            if node.state.is_some() || !Dependencies::reads(node).is_empty() {
+                return Err(Error::State(
+                    "temporal flat state dependency changed".into(),
+                ));
+            }
+        }
+        Ok(())
+    }
+    fn bridge_span(small: &DeviceBuffer, large: &DeviceBuffer, teacher: usize) -> Result<usize> {
+        if teacher >= 2 || small.elements.checked_mul(2) != Some(large.elements) {
+            return Err(Error::State("temporal bridge true row geometry".into()));
+        }
+        let offset = teacher
+            .checked_mul(small.elements)
+            .ok_or_else(|| Error::Limit("temporal bridge offset".into()))?;
+        if offset
+            .checked_add(small.elements)
+            .is_none_or(|end| end > large.elements)
+        {
+            return Err(Error::Limit("temporal bridge range".into()));
+        }
+        let a = BufferRange::new(&small.handle)?;
+        let b = BufferRange::new(&large.handle)?;
+        if a.service != b.service
+            || a.overlaps(&b)
+            || a.bytes.end - a.bytes.start < bytes(small.elements, 4)? as u64
+            || b.bytes.end - b.bytes.start < bytes(large.elements, 4)? as u64
+        {
+            return Err(Error::State("temporal bridge binding/alias".into()));
+        }
+        Ok(offset)
+    }
+    fn bridge(
+        &self,
+        client: &Client,
+        causal: &BatchProgram,
+        id: ValueId,
+        teacher: usize,
+        to_flat: bool,
+    ) -> Result<usize> {
+        let small = &causal.inner.values[id.0 as usize];
+        let large = &self.flat.inner.values[id.0 as usize];
+        let offset = Self::bridge_span(small, large, teacher)?;
+        Ok(if to_flat {
+            causal.copy_dense(
+                client,
+                &small.handle,
+                small.elements,
+                0,
+                &large.handle,
+                large.elements,
+                offset,
+                small.elements,
+            )
+        } else {
+            causal.copy_dense(
+                client,
+                &large.handle,
+                large.elements,
+                offset,
+                &small.handle,
+                small.elements,
+                0,
+                small.elements,
+            )
+        })
+    }
+    fn certify_candidate(
+        client: &Client,
+        program: &BatchProgram,
+        slot: StateSlot,
+    ) -> Result<usize> {
+        let binding = program
+            .packed
+            .iter()
+            .find(|binding| binding.slot == slot)
+            .ok_or_else(|| Error::State("temporal effect packed domain".into()))?;
+        Ok(binding
+            .candidate
+            .buffers()
+            .iter()
+            .map(|value| program.certify(client, &value.handle, value.elements))
+            .sum())
+    }
+    fn selected_status(&self, causal: &BatchProgram, submission: &mut ResidentSubmission) {
+        submission.quant_status.clear();
+        submission.embedding_status.clear();
+        for (program, selected) in [
+            (&self.flat, &self.dependencies.flat),
+            (causal, &self.dependencies.causal),
+        ] {
+            for &index in selected {
+                match &program.nodes[index].operation {
+                    PlanOperation::Project { tensor } => {
+                        if let Some(layout) = program.inner.tensors[&tensor.name].layout {
+                            let flags = &program.inner.node_status[index];
+                            submission.quant_status.push((
+                                flags.block.as_ref().unwrap().clone(),
+                                program.rows * layout.blocks,
+                                flags.row.as_ref().unwrap().clone(),
+                                program.rows * layout.rows,
+                            ));
+                        }
+                    }
+                    PlanOperation::Embed { tensor, .. } => submission.embedding_status.push((
+                        program.inner.node_status[index]
+                            .block
+                            .as_ref()
+                            .unwrap()
+                            .clone(),
+                        program.rows * tensor.columns,
+                    )),
+                    _ => {}
+                }
+            }
+        }
+    }
+    pub(crate) fn launch(
+        &self,
+        client: &Client,
+        causal: &mut BatchProgram,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<ResidentSubmission> {
+        self.validate(client, causal, steps, bindings)?;
+        let prefix = causal.launch_inputs(client, &steps[0], bindings, None)?;
+        let flat_parameters = steps.iter().flatten().copied().collect::<Vec<_>>();
+        unsafe {
+            crate::program_kernels::clear_step_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&self.flat.inner.status, 1),
+            );
+        }
+        let mut body = BatchBodyReceipt {
+            dispatches: 1 + self.flat.stage_teacher_parameters(client, &flat_parameters),
+            fused_dispatches: 0,
+        };
+        for stage in &self.dependencies.stages {
+            match stage {
+                Stage::Flat(index) => {
+                    let receipt =
+                        self.flat
+                            .launch_checked_node(client, *index, &flat_parameters)?;
+                    body.dispatches += receipt.dispatches;
+                    body.fused_dispatches += receipt.fused_dispatches;
+                }
+                Stage::Causal { nodes, slots } => {
+                    let mut banks = Banks {
+                        program: causal,
+                        slots,
+                        flipped: false,
+                    };
+                    for (teacher, parameters) in steps.iter().enumerate() {
+                        if teacher == 1 {
+                            banks.flip();
+                        }
+                        body.dispatches +=
+                            banks.program.stage_teacher_parameters(client, parameters);
+                        for index in nodes.clone() {
+                            let node = &self.flat.inner.plan.nodes()[index];
+                            for id in &node.inputs {
+                                body.dispatches +=
+                                    self.bridge(client, banks.program, *id, teacher, false)?;
+                            }
+                            let receipt = banks
+                                .program
+                                .launch_checked_node(client, index, parameters)?;
+                            body.dispatches += receipt.dispatches;
+                            body.fused_dispatches += receipt.fused_dispatches;
+                            // Retain every transient finite/status decision before
+                            // its actual bank/value can be reused or overwritten.
+                            if let Some(effect) = node.state {
+                                #[cfg(test)]
+                                if teacher == 0
+                                    && matches!(effect.write.slot, StateSlot::Kv(_))
+                                    && INACTIVE_NAN.with(|flag| flag.replace(false))
+                                {
+                                    if parameters.iter().any(|p| p.valid_tokens >= 512) {
+                                        return Err(Error::State(
+                                            "temporal NaN probe requires inactive KV".into(),
+                                        ));
+                                    }
+                                    body.dispatches += Self::poison_intermediate(
+                                        client,
+                                        banks.program,
+                                        effect.write.slot,
+                                    )?;
+                                }
+                                body.dispatches += Self::certify_candidate(
+                                    client,
+                                    banks.program,
+                                    effect.write.slot,
+                                )?;
+                            }
+                            for id in &node.outputs {
+                                body.dispatches +=
+                                    self.bridge(client, banks.program, *id, teacher, true)?;
+                            }
+                        }
+                    }
+                }
+            }
+        }
+        // TWO persistent OR statuses; one real terminal status/readback. No
+        // per-node/per-teacher host synchronization or status reset occurred.
+        unsafe {
+            crate::resident_bindings::merge_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&self.flat.inner.status, 1),
+                arg(&causal.inner.status, 1),
+                1,
+            );
+        }
+        body.dispatches += 1;
+        let slots = causal
+            .packed
+            .iter()
+            .map(|state| state.slot)
+            .collect::<Vec<_>>();
+        let mut final_banks = Banks {
+            program: causal,
+            slots: &slots,
+            flipped: false,
+        };
+        final_banks.flip(); // T1 lives in original prior A; candidate now A.
+        let mut submission = final_banks.program.launch_outputs(
+            client,
+            bindings,
+            ResidentOutput::StateOnly,
+            prefix,
+            body,
+        )?;
+        self.selected_status(final_banks.program, &mut submission);
+        Ok(submission)
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use zerogravity_core::accelerator::program::{
+        AttentionGeometry, ConvDomain, KvDomain, ParameterRef, ParameterSlot, StateEffect,
+        StaticVectorId,
+    };
+    fn node(
+        operation: PlanOperation,
+        inputs: &[u32],
+        outputs: &[u32],
+        state: Option<StateEffect>,
+    ) -> PlanNode {
+        PlanNode {
+            name: "synthetic descriptor".into(),
+            operation,
+            inputs: inputs.iter().copied().map(ValueId).collect(),
+            outputs: outputs.iter().copied().map(ValueId).collect(),
+            state,
+        }
+    }
+    #[test]
+    fn hidden_reads_are_not_flat_and_unproduced_inputs_refuse() {
+        let state = StateVersion {
+            slot: StateSlot::Kv(KvDomain(0)),
+            ordinal: 1,
+        };
+        let qk = node(
+            PlanOperation::AttentionQk {
+                state,
+                geometry: AttentionGeometry {
+                    width: 128,
+                    query_heads: 2,
+                    kv_heads: 1,
+                },
+                valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                scale: 1.0,
+            },
+            &[],
+            &[],
+            None,
+        );
+        assert_eq!(Dependencies::reads(&qk), vec![state]);
+        assert!(Dependencies::nodes(&[qk], 1, 0).is_err());
+        let add = node(PlanOperation::Add, &[0, 1], &[2], None);
+        assert!(Dependencies::nodes(&[add], 1, 3).is_err());
+    }
+    #[test]
+    fn duplicate_effect_and_reading_overwritten_initial_version_refuse() {
+        let slot = StateSlot::Conv(ConvDomain(0));
+        let effect = StateEffect {
+            read: StateVersion { slot, ordinal: 0 },
+            write: StateVersion { slot, ordinal: 1 },
+        };
+        let conv = node(
+            PlanOperation::Conv {
+                weights: StaticVectorId(0),
+            },
+            &[],
+            &[],
+            Some(effect),
+        );
+        let observe = node(
+            PlanOperation::ObserveState {
+                layer: 0,
+                stage: "synthetic",
+                state: effect.read,
+            },
+            &[],
+            &[],
+            None,
+        );
+        assert!(Dependencies::nodes(&[conv.clone(), conv.clone()], 2, 0).is_err());
+        assert!(Dependencies::nodes(&[conv, observe], 2, 0).is_err());
+    }
+    #[test]
+    fn genuine_teacher_axis_is_not_a_state_or_strided_kv_axis() {
+        assert_eq!(
+            (0..2)
+                .flat_map(|teacher| (0..2).map(move |row| teacher * 2 + row))
+                .collect::<Vec<_>>(),
+            vec![0, 1, 2, 3]
+        );
+        let slot = StateSlot::Conv(ConvDomain(0));
+        let effect = StateEffect {
+            read: StateVersion { slot, ordinal: 0 },
+            write: StateVersion { slot, ordinal: 1 },
+        };
+        let nodes = vec![
+            node(PlanOperation::Contiguous, &[], &[0], None),
+            node(
+                PlanOperation::Conv {
+                    weights: StaticVectorId(0),
+                },
+                &[0],
+                &[1],
+                Some(effect),
+            ),
+            node(PlanOperation::Silu { multiplied: false }, &[1], &[2], None),
+            node(
+                PlanOperation::ObserveState {
+                    layer: 0,
+                    stage: "synthetic",
+                    state: effect.write,
+                },
+                &[],
+                &[],
+                None,
+            ),
+        ];
+        let descriptor = Dependencies::nodes(&nodes, 4, 3).unwrap();
+        assert_eq!(descriptor.flat, BTreeSet::from([0, 2]));
+        assert_eq!(descriptor.causal, BTreeSet::from([1, 3]));
+        assert_eq!(descriptor.stages.len(), 4);
+    }
+    #[test]
+    fn attention_keeps_hidden_reads_and_softmax_in_one_causal_region_and_clips_original_cut() {
+        let slot = StateSlot::Kv(KvDomain(0));
+        let prior = StateVersion { slot, ordinal: 0 };
+        let proposed = StateVersion { slot, ordinal: 1 };
+        let geometry = AttentionGeometry {
+            width: 128,
+            query_heads: 2,
+            kv_heads: 1,
+        };
+        let nodes = vec![
+            node(PlanOperation::Contiguous, &[], &[0], None),
+            node(
+                PlanOperation::KvPrefixGuard {
+                    state: prior,
+                    position: ParameterRef(ParameterSlot::Position),
+                },
+                &[],
+                &[],
+                None,
+            ),
+            node(
+                PlanOperation::AppendKv,
+                &[0, 0],
+                &[],
+                Some(StateEffect {
+                    read: prior,
+                    write: proposed,
+                }),
+            ),
+            node(
+                PlanOperation::AttentionQk {
+                    state: proposed,
+                    geometry,
+                    valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                    scale: 1.0,
+                },
+                &[0],
+                &[1, 2],
+                None,
+            ),
+            node(
+                PlanOperation::AttentionSoftmax {
+                    heads: 2,
+                    valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                },
+                &[2],
+                &[3],
+                None,
+            ),
+            node(
+                PlanOperation::AttentionValue {
+                    state: proposed,
+                    geometry,
+                    valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                    padding: 256,
+                },
+                &[3],
+                &[4],
+                None,
+            ),
+        ];
+        let full = Dependencies::nodes(&nodes, 6, 5).unwrap();
+        assert_eq!(
+            full.stages[1],
+            Stage::Causal {
+                nodes: 1..6,
+                slots: vec![slot]
+            }
+        );
+        let cut = Dependencies::nodes(&nodes, 3, 5).unwrap();
+        assert_eq!(
+            cut.stages[1],
+            Stage::Causal {
+                nodes: 1..3,
+                slots: vec![slot]
+            }
+        );
+        assert_eq!(cut.causal, BTreeSet::from([1, 2]));
+        let mut wrong = nodes.clone();
+        wrong[3].inputs = vec![ValueId(4)];
+        assert!(Dependencies::nodes(&wrong, 6, 5).is_err());
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -64,6 +64,10 @@
 
 #[cfg(all(feature = "qualification", feature = "cuda"))]
 mod chunk;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod temporal;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+pub(crate) use temporal::TemporalProgram;
 
 impl BatchProgram {
     /// New allocations only: the original source handles remain shared.
@@ -261,6 +265,8 @@
         let positions = allocator.empty(ResidentBufferRole::Parameter, bytes(rows, 4)?)?;
         let mut inner = ResidentProgram {
             _source_owner: Arc::clone(&original._source_owner),
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            plan_lineage: Arc::clone(&original.plan_lineage),
             plan: original.plan.clone(),
             preflight,
             tensors,
@@ -599,48 +605,61 @@
         parameters: &[RuntimeParameters],
         output: ResidentOutput,
     ) -> Result<BatchBodyReceipt> {
-        let mut dispatches = 0;
-        // One actual native projection-dot launch per Project. Activation
-        // preparation/status stages remain in the total dispatch count only.
-        let mut fused_dispatches = 0;
-        let count = output.nodes(&self.inner.plan).len();
-        for (index, node) in self.nodes[..count].iter().enumerate() {
-            dispatches += self.launch_node(client, index, node, parameters)?;
-            if matches!(node.operation, PlanOperation::Project { .. }) {
-                fused_dispatches += 1;
-            }
-            if matches!(node.operation, PlanOperation::View(_)) {
-                continue;
-            }
-            for (out_index, id) in node.outputs.iter().enumerate() {
-                let value = &self.inner.values[id.0 as usize];
-                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
-                    && out_index == 1
-                {
-                    unsafe {
-                        crate::resident_bindings::certify_causal_rows::launch_unchecked(
-                            client,
-                            grid(value.elements),
-                            threads(),
-                            arg(&value.handle, value.elements),
-                            arg(&self.inner.status, 1),
-                            arg(&self.positions, self.rows),
-                            geometry.query_heads / self.rows,
-                            self.rows,
-                            self.inner.plan.prefix_capacity(),
-                        );
-                    }
-                } else {
-                    dispatches += self.certify(client, &value.handle, value.elements);
-                    continue;
-                }
-                dispatches += 1;
-            }
-        }
-        Ok(BatchBodyReceipt {
-            dispatches,
-            fused_dispatches,
-        })
+        let mut receipt = BatchBodyReceipt {
+            dispatches: 0,
+            fused_dispatches: 0,
+        };
+        for index in 0..output.nodes(&self.inner.plan).len() {
+            let node = self.launch_checked_node(client, index, parameters)?;
+            receipt.dispatches += node.dispatches;
+            receipt.fused_dispatches += node.fused_dispatches;
+        }
+        Ok(receipt)
+    }
+
+    /// The original single node and its original output certifications. Shared
+    /// by the unchanged ordered body and the closed dependency lowering.
+    fn launch_checked_node(
+        &self,
+        client: &Client,
+        index: usize,
+        parameters: &[RuntimeParameters],
+    ) -> Result<BatchBodyReceipt> {
+        let node = self
+            .nodes
+            .get(index)
+            .ok_or_else(|| Error::State("native node index".into()))?;
+        let mut receipt = BatchBodyReceipt {
+            dispatches: self.launch_node(client, index, node, parameters)?,
+            fused_dispatches: usize::from(matches!(node.operation, PlanOperation::Project { .. })),
+        };
+        if matches!(node.operation, PlanOperation::View(_)) {
+            return Ok(receipt);
+        }
+        for (out_index, id) in node.outputs.iter().enumerate() {
+            let value = &self.inner.values[id.0 as usize];
+            if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
+                && out_index == 1
+            {
+                unsafe {
+                    crate::resident_bindings::certify_causal_rows::launch_unchecked(
+                        client,
+                        grid(value.elements),
+                        threads(),
+                        arg(&value.handle, value.elements),
+                        arg(&self.inner.status, 1),
+                        arg(&self.positions, self.rows),
+                        geometry.query_heads / self.rows,
+                        self.rows,
+                        self.inner.plan.prefix_capacity(),
+                    );
+                }
+                receipt.dispatches += 1;
+            } else {
+                receipt.dispatches += self.certify(client, &value.handle, value.elements);
+            }
+        }
+        Ok(receipt)
     }
 
     fn launch_outputs(
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -40,7 +40,7 @@
 mod batch;
 pub(crate) use batch::BatchProgram;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
-pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};
+pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay, TemporalProgram};
 
 #[cfg(feature = "qualification")]
 #[path = "resident_projection_qualification.rs"]
@@ -302,6 +302,8 @@
 pub(crate) struct ResidentProgram {
     // The immutable original allocation remains alive for every resident handle.
     _source_owner: Arc<dyn TensorSource>,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    plan_lineage: Arc<()>,
     plan: ForwardPlan,
     preflight: ResidentPreflight,
     tensors: BTreeMap<String, ResidentTensor>,
@@ -977,6 +979,8 @@
             .collect::<Result<Vec<_>>>()?;
         let mut program = Self {
             _source_owner: model,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            plan_lineage: Arc::new(()),
             plan,
             preflight,
             tensors,

````

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/PRODUCTION.review.patch

Original UTF-8 bytes: 64407; SHA256 `26eda1d2a4e50c6f392f69be374cc88b6380772381faf126b9ed5b61bc2badca`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/fused.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/fused.rs
@@ -5,6 +5,8 @@
     Native,
     #[cfg(all(feature = "qualification", feature = "cuda"))]
     NativeChunk(crate::resident_batch::ResidentBatchChunk),
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    Temporal(crate::resident_batch::ResidentBatchChunk),
     #[cfg(all(feature = "qualification", feature = "cuda"))]
     Graph(Arc<super::graph::CapturedResidentGraph>),
 }
@@ -27,6 +29,49 @@
         return Err(Error::Unsupported(
             "graph requires native host-token band and matching typed body/readout".into(),
         ));
+    }
+    Ok(())
+}
+
+fn allocate_batch_buffer(
+    initialization: &mut crate::resident_leases::ResidentInitialization<'_>,
+    leases: &mut Vec<(ResidentLease, DeviceAccess)>,
+    role: ResidentBufferRole,
+    count: usize,
+    data: AllocationInit<'_>,
+) -> Result<Handle> {
+    let role = match role {
+        ResidentBufferRole::Source => BufferRole::ImmutableWeights,
+        ResidentBufferRole::Value | ResidentBufferRole::Parameter => BufferRole::Activation,
+        ResidentBufferRole::Workspace | ResidentBufferRole::Status => BufferRole::Scratch,
+    };
+    let lease = match data {
+        AllocationInit::Empty => initialization.allocate_empty(count, role)?,
+        AllocationInit::Upload(data) => {
+            if data.len() != count {
+                return Err(Error::State("native initializer length".into()));
+            }
+            initialization.upload_bytes(data, role)?
+        }
+    };
+    let handle = lease.handle();
+    leases.push((lease, DeviceAccess::Write));
+    Ok(handle)
+}
+fn verify_batch_allocations(
+    bytes: usize,
+    handles: Vec<Handle>,
+    leases: &[(ResidentLease, DeviceAccess)],
+) -> Result<()> {
+    for handle in handles {
+        lease_for(leases, &handle)?;
+    }
+    let actual = leases.iter().try_fold(0usize, |n, (lease, _)| {
+        n.checked_add(lease.bytes())
+            .ok_or_else(|| Error::Limit("native allocation coverage overflow".into()))
+    })?;
+    if actual != bytes {
+        return Err(Error::State("native allocation coverage".into()));
     }
     Ok(())
 }
@@ -93,6 +138,39 @@
         .await
     }
 
+    /// Closed B2/T2 native dependency grouping; physical B4 activations are
+    /// never four sessions. Original atomic publisher advances two roots by2.
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub async fn prepare_state_temporal_b2_t2_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, &[u32])],
+    ) -> Result<Option<PreparedResidentCohort>> {
+        self.service.pinned_cuda_queue()?;
+        if rows.len() != 2 || rows.iter().any(|(_, tokens)| tokens.len() != 2) {
+            return Err(Error::Unsupported(
+                "temporal grouping requires B2/T2".into(),
+            ));
+        }
+        let bases = rows
+            .iter()
+            .map(|(session, _)| session.binding())
+            .collect::<Vec<_>>();
+        let teachers = rows.iter().map(|(_, tokens)| *tokens).collect::<Vec<_>>();
+        let chunk = crate::resident_batch::ResidentBatchChunk::new(
+            &bases,
+            &teachers,
+            self.vocab,
+            self.capacity,
+        )?;
+        let first = rows
+            .iter()
+            .enumerate()
+            .map(|(row, (session, _))| Ok((*session, chunk.token(row, 0)?)))
+            .collect::<Result<Vec<_>>>()?;
+        self.prepare_fused_execution(&first, None, Readout::None, FusedExecution::Temporal(chunk))
+            .await
+    }
+
     /// One native full-model batch and source-exact device greedy selection.
     /// Each independent output packet gains its producer only at cohort commit.
     pub async fn prepare_greedy_fused_cohort(
@@ -178,19 +256,38 @@
             graph_readout(rows.len(), graph.output(), readout, device_tokens.is_some())?;
         }
         #[cfg(all(feature = "qualification", feature = "cuda"))]
-        if matches!(&execution, FusedExecution::NativeChunk(_))
-            && (!matches!(readout, Readout::None) || device_tokens.is_some())
+        if matches!(
+            &execution,
+            FusedExecution::NativeChunk(_) | FusedExecution::Temporal(_)
+        ) && (!matches!(readout, Readout::None) || device_tokens.is_some())
         {
             return Err(Error::Unsupported(
                 "teacher chunk requires state-only host tokens".into(),
             ));
         }
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let temporal = matches!(&execution, FusedExecution::Temporal(_));
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if let FusedExecution::Temporal(chunk) = &execution {
+            self.service.pinned_cuda_queue()?;
+            if rows.len() != 2
+                || chunk.rows() != 2
+                || chunk.tokens_per_row() != 2
+                || self.capacity != 512
+            {
+                return Err(Error::Unsupported(
+                    "temporal closed B2/T2/cap512 only".into(),
+                ));
+            }
+        }
         let tokens_per_row = match &execution {
             FusedExecution::Native => 1,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
             FusedExecution::Graph(_) => 1,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
-            FusedExecution::NativeChunk(chunk) => chunk.tokens_per_row(),
+            FusedExecution::NativeChunk(chunk) | FusedExecution::Temporal(chunk) => {
+                chunk.tokens_per_row()
+            }
         };
         let output = if matches!(readout, Readout::None) {
             ResidentOutput::StateOnly
@@ -205,7 +302,7 @@
         let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
         let roots = ResidentBatchRows::new(&bases, 1)?;
         #[cfg(all(feature = "qualification", feature = "cuda"))]
-        if let FusedExecution::NativeChunk(chunk) = &execution {
+        if let FusedExecution::NativeChunk(chunk) | FusedExecution::Temporal(chunk) = &execution {
             chunk.require_current(&bases)?;
             for (row, (_, token)) in rows.iter().enumerate() {
                 if chunk.token(row, 0)? != *token {
@@ -295,11 +392,13 @@
             let mut workspace = self.workspace.lock().await;
             self.service.ensure_usable()?;
             #[cfg(all(feature = "qualification", feature = "cuda"))]
-            if matches!(&execution, FusedExecution::NativeChunk(_))
-                && workspace
-                    .batch
-                    .as_ref()
-                    .is_some_and(|batch| batch.program.graph_pinned())
+            if matches!(
+                &execution,
+                FusedExecution::NativeChunk(_) | FusedExecution::Temporal(_)
+            ) && workspace
+                .batch
+                .as_ref()
+                .is_some_and(|batch| batch.program.graph_pinned())
             {
                 return Err(Error::Unsupported(
                     "teacher chunk refuses a graph-pinned workspace before allocation".into(),
@@ -345,7 +444,7 @@
                 workspace
                     .batch
                     .as_ref()
-                    .map(|b| b.program.budget().owned_bytes())
+                    .map(BatchWorkspace::owned_bytes)
                     .transpose()?
                     .unwrap_or(0)
             };
@@ -353,6 +452,23 @@
                 0
             } else {
                 BatchProgram::preflight(&workspace.program, rows.len())?.owned_bytes()?
+            };
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let workspace_bytes = if temporal
+                && (!same_band
+                    || workspace
+                        .batch
+                        .as_ref()
+                        .is_none_or(|batch| batch.temporal.is_none()))
+            {
+                workspace_bytes
+                    .checked_add(
+                        crate::resident_program::TemporalProgram::preflight(&workspace.program)?
+                            .owned_bytes()?,
+                    )
+                    .ok_or_else(|| Error::Limit("temporal combined admission".into()))?
+            } else {
+                workspace_bytes
             };
             let required = staging_bytes
                 .checked_add(packet_bytes)
@@ -392,7 +508,9 @@
             })
             .collect::<Vec<_>>();
         #[cfg(all(feature = "qualification", feature = "cuda"))]
-        let chunk_parameters = if let FusedExecution::NativeChunk(chunk) = &execution {
+        let chunk_parameters = if let FusedExecution::NativeChunk(chunk)
+        | FusedExecution::Temporal(chunk) = &execution
+        {
             Some(
                 (0..tokens_per_row)
                     .map(|step| {
@@ -454,72 +572,121 @@
             workspace.batch = None;
         }
         let batch_budget = BatchProgram::preflight(&workspace.program, rows.len())?;
-        if workspace.batch.is_none()
-            && batch_budget.owned_bytes()? > self.service.remaining_capacity()?
-        {
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let needs_temporal = temporal
+            && workspace
+                .batch
+                .as_ref()
+                .is_none_or(|batch| batch.temporal.is_none());
+        #[cfg(not(all(feature = "qualification", feature = "cuda")))]
+        let needs_temporal = false;
+        let base_bytes = if workspace.batch.is_none() {
+            batch_budget.owned_bytes()?
+        } else {
+            0
+        };
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let cold_bytes = base_bytes
+            .checked_add(if needs_temporal {
+                crate::resident_program::TemporalProgram::preflight(&workspace.program)?
+                    .owned_bytes()?
+            } else {
+                0
+            })
+            .ok_or_else(|| Error::Limit("temporal complete cold admission".into()))?;
+        #[cfg(not(all(feature = "qualification", feature = "cuda")))]
+        let cold_bytes = base_bytes;
+        if cold_bytes > self.service.remaining_capacity()? {
             return Err(Error::Limit(
-                "native resident batch workspace admission exhausted".into(),
+                "native combined workspace admission exhausted".into(),
             ));
         }
-        if workspace.batch.is_none() {
+        if workspace.batch.is_none() || needs_temporal {
             slot.begin_submission()?;
             let mut initialization = self.service.begin_initialization(bases[0].root())?;
             let mut leases = Vec::new();
-            let program = BatchProgram::prepare(
-                self.service.client(),
-                &workspace.program,
-                rows.len(),
-                |role, count, data| {
-                    let role = match role {
-                        ResidentBufferRole::Source => BufferRole::ImmutableWeights,
-                        ResidentBufferRole::Value | ResidentBufferRole::Parameter => {
-                            BufferRole::Activation
-                        }
-                        ResidentBufferRole::Workspace | ResidentBufferRole::Status => {
-                            BufferRole::Scratch
-                        }
-                    };
-                    let lease = match data {
-                        AllocationInit::Empty => initialization.allocate_empty(count, role)?,
-                        AllocationInit::Upload(data) => {
-                            if data.len() != count {
-                                return Err(Error::State("native batch initializer length".into()));
-                            }
-                            initialization.upload_bytes(data, role)?
-                        }
-                    };
-                    let handle = lease.handle();
-                    leases.push((lease, DeviceAccess::Write));
-                    Ok(handle)
-                },
-            )?;
-            if program.rows() != rows.len()
-                || program.budget().owned_bytes()? != batch_budget.owned_bytes()?
-            {
+            let program = if workspace.batch.is_none() {
+                Some(BatchProgram::prepare(
+                    self.service.client(),
+                    &workspace.program,
+                    rows.len(),
+                    |role, count, data| {
+                        allocate_batch_buffer(&mut initialization, &mut leases, role, count, data)
+                    },
+                )?)
+            } else {
+                None
+            };
+            if let Some(program) = &program {
+                if program.rows() != rows.len()
+                    || program.budget().owned_bytes()? != batch_budget.owned_bytes()?
+                {
+                    return Err(Error::State(
+                        "native prepared geometry/budget changed".into(),
+                    ));
+                }
+                verify_batch_allocations(
+                    batch_budget.owned_bytes()?,
+                    program.owned_handles(),
+                    &leases,
+                )?;
+            }
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let mut temporal_leases = Vec::new();
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let companion = if needs_temporal {
+                let causal = program
+                    .as_ref()
+                    .unwrap_or_else(|| &workspace.batch.as_ref().unwrap().program);
+                let companion = crate::resident_program::TemporalProgram::prepare(
+                    self.service.client(),
+                    &workspace.program,
+                    causal,
+                    |role, count, data| {
+                        allocate_batch_buffer(
+                            &mut initialization,
+                            &mut temporal_leases,
+                            role,
+                            count,
+                            data,
+                        )
+                    },
+                )?;
+                verify_batch_allocations(
+                    crate::resident_program::TemporalProgram::preflight(&workspace.program)?
+                        .owned_bytes()?,
+                    companion.owned_handles(),
+                    &temporal_leases,
+                )?;
+                Some(companion)
+            } else {
+                None
+            };
+            let allocated = leases.len();
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            let allocated = allocated + temporal_leases.len();
+            let ready = initialization.complete_sync().await?;
+            if ready.outputs().len() != allocated {
                 return Err(Error::State(
-                    "native batch prepared geometry/budget changed".into(),
+                    "native combined initialization incomplete".into(),
                 ));
             }
-            for handle in program.owned_handles() {
-                lease_for(&leases, &handle)?;
-            }
-            if leases.iter().map(|(l, _)| l.bytes()).sum::<usize>() != batch_budget.owned_bytes()? {
-                return Err(Error::State("native batch allocation coverage".into()));
-            }
-            let ready = initialization.complete_sync().await?;
-            if ready.outputs().len() != leases.len() {
-                return Err(Error::State(
-                    "native batch initialization incomplete".into(),
-                ));
-            }
-            workspace.batch = Some(BatchWorkspace {
-                program,
-                leases,
-                rows: rows.len(),
-            });
-            // Initialization reached a real terminal sync. Retire its slot,
-            // then release the workspace before waiting for execution: a B1
-            // worker selected meanwhile may need that same workspace lock.
+            if let Some(program) = program {
+                workspace.batch = Some(BatchWorkspace {
+                    program,
+                    leases,
+                    rows: rows.len(),
+                    #[cfg(all(feature = "qualification", feature = "cuda"))]
+                    temporal: None,
+                });
+            }
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            if let Some(program) = companion {
+                workspace.batch.as_mut().unwrap().temporal = Some(TemporalWorkspace {
+                    program,
+                    leases: temporal_leases,
+                });
+            }
             slot.finish()?;
             drop(workspace);
             let ticket = self.service.queue().submit(
@@ -540,9 +707,18 @@
             .validate_runtime(self.service.client(), &parameters, &bindings)?;
         #[cfg(all(feature = "qualification", feature = "cuda"))]
         if let Some(steps) = &chunk_parameters {
-            batch
-                .program
-                .validate_state_chunk(self.service.client(), steps, &bindings)?;
+            if temporal {
+                batch
+                    .temporal
+                    .as_ref()
+                    .ok_or_else(|| Error::State("temporal workspace absent".into()))?
+                    .program
+                    .validate(self.service.client(), &batch.program, steps, &bindings)?;
+            } else {
+                batch
+                    .program
+                    .validate_state_chunk(self.service.client(), steps, &bindings)?;
+            }
         }
         let device_buffers = device_tokens.map(|tokens| {
             tokens
@@ -578,6 +754,21 @@
             )?;
         }
         let handles = batch.program.execution_handles(output);
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let handles = if temporal {
+            let mut handles = handles;
+            handles.extend(
+                batch
+                    .temporal
+                    .as_ref()
+                    .ok_or_else(|| Error::State("temporal workspace absent".into()))?
+                    .program
+                    .execution_handles(),
+            );
+            handles
+        } else {
+            handles
+        };
         let mut accesses = workspace
             .leases
             .iter()
@@ -585,6 +776,19 @@
             .filter(|(l, _)| handles.iter().any(|h| same_handle(h, &l.handle())))
             .cloned()
             .collect::<Vec<_>>();
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if temporal {
+            accesses.extend(
+                batch
+                    .temporal
+                    .as_ref()
+                    .unwrap()
+                    .leases
+                    .iter()
+                    .filter(|(lease, _)| handles.iter().any(|h| same_handle(h, &lease.handle())))
+                    .cloned(),
+            );
+        }
         for handle in handles {
             lease_for(&accesses, &handle)?;
         }
@@ -631,8 +835,23 @@
             }
             flight
         };
-        let program = &mut workspace.batch.as_mut().unwrap().program;
+        let batch = workspace.batch.as_mut().unwrap();
+        let program = &mut batch.program;
         let mut submitted = match &execution {
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::Temporal(_) => batch
+                .temporal
+                .as_ref()
+                .ok_or_else(|| Error::State("temporal workspace retired".into()))?
+                .program
+                .launch(
+                    self.service.client(),
+                    program,
+                    chunk_parameters
+                        .as_ref()
+                        .ok_or_else(|| Error::State("temporal parameters absent".into()))?,
+                    &bindings,
+                )?,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
             FusedExecution::Graph(graph) => graph.launch(program, &parameters, &bindings)?,
             #[cfg(all(feature = "qualification", feature = "cuda"))]
--- a/crates/zerogravity-gpu/src/resident_inference.rs
+++ b/crates/zerogravity-gpu/src/resident_inference.rs
@@ -110,6 +110,29 @@
     program: BatchProgram,
     leases: Vec<(ResidentLease, DeviceAccess)>,
     rows: usize,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    temporal: Option<TemporalWorkspace>,
+}
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+struct TemporalWorkspace {
+    program: crate::resident_program::TemporalProgram,
+    leases: Vec<(ResidentLease, DeviceAccess)>,
+}
+impl BatchWorkspace {
+    fn owned_bytes(&self) -> Result<usize> {
+        let bytes = self.program.budget().owned_bytes()?;
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let bytes = bytes
+            .checked_add(
+                self.temporal
+                    .as_ref()
+                    .map(|t| t.program.budget().owned_bytes())
+                    .transpose()?
+                    .unwrap_or(0),
+            )
+            .ok_or_else(|| Error::Limit("temporal complete workspace budget".into()))?;
+        Ok(bytes)
+    }
 }
 /// Shared immutable source and serialized physical workspace, independent roots.
 pub struct ResidentModel {
--- a/crates/zerogravity-gpu/src/resident_program/batch/chunk.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch/chunk.rs
@@ -262,7 +262,11 @@
 
     /// Same two scalar-token/position kernels as original launch_inputs; no
     /// clear, gather, upload, device-token feedback, sampler or graph enqueue.
-    fn stage_teacher_parameters(&self, client: &Client, parameters: &[RuntimeParameters]) -> usize {
+    pub(super) fn stage_teacher_parameters(
+        &self,
+        client: &Client,
+        parameters: &[RuntimeParameters],
+    ) -> usize {
         let mut tokens = [0u32; 8];
         let mut positions = [0u32; 8];
         for ((token, position), parameter) in tokens.iter_mut().zip(&mut positions).zip(parameters)
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/temporal.rs
@@ -0,0 +1,911 @@
+//! Closed B2/T2 dependency lowering. Physical activation rows are NOT roots.
+use super::*;
+use std::collections::BTreeSet;
+use std::ops::Range;
+
+#[cfg(test)]
+std::thread_local! {
+    static INACTIVE_NAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
+}
+
+#[derive(Clone, Debug, PartialEq, Eq)]
+enum Stage {
+    Flat(usize),
+    Causal {
+        nodes: Range<usize>,
+        slots: Vec<StateSlot>,
+    },
+}
+
+/// Derived from the SAME immutable plan; no token/layer schedule is copied.
+#[derive(Clone, Debug)]
+struct Dependencies {
+    stages: Vec<Stage>,
+    flat: BTreeSet<usize>,
+    causal: BTreeSet<usize>,
+    count: usize,
+}
+impl Dependencies {
+    fn new(plan: &ForwardPlan) -> Result<Self> {
+        ResidentProgram::require_legacy_contract(plan)?;
+        let result = Self::nodes(plan.nodes(), plan.state_nodes().len(), plan.values().len())?;
+        let actual = result
+            .causal
+            .iter()
+            .filter_map(|&index| plan.nodes()[index].state.map(|effect| effect.write))
+            .collect::<Vec<_>>();
+        if actual.len() != plan.proposed_domains().len()
+            || plan
+                .proposed_domains()
+                .iter()
+                .any(|wanted| actual.iter().filter(|actual| *actual == wanted).count() != 1)
+        {
+            return Err(Error::State("temporal all-domain effect closure".into()));
+        }
+        Ok(result)
+    }
+
+    fn nodes(nodes: &[PlanNode], count: usize, values: usize) -> Result<Self> {
+        if count == 0 || count > nodes.len() || nodes.len() > 32768 || values > 65536 {
+            return Err(Error::Limit("temporal plan metadata bound".into()));
+        }
+        let mut stages = Vec::new();
+        let mut flat = BTreeSet::new();
+        let mut causal = BTreeSet::new();
+        let mut produced = BTreeSet::new();
+        let mut advanced = BTreeSet::new();
+        let mut index = 0;
+        while index < count {
+            let node = &nodes[index];
+            let stage = match &node.operation {
+                PlanOperation::KvPrefixGuard { state, .. } => {
+                    if state.ordinal != 0 || advanced.contains(&state.slot) {
+                        return Err(Error::State("temporal KV prior version".into()));
+                    }
+                    // Find the actual full-plan region, then clip at the
+                    // original StateOnly cut (which can end just after append).
+                    let end = nodes[index + 1..].iter().position(|node| {
+                        matches!(node.operation, PlanOperation::AttentionValue { state: read, .. }
+                            if read.slot == state.slot && read.ordinal == 1)
+                    }).map(|offset| index + offset + 2)
+                        .ok_or_else(|| Error::Unsupported("temporal whole attention region".into()))?;
+                    let mut writes = 0;
+                    for entry in &nodes[index..end] {
+                        Self::known(entry)?;
+                        if let Some(effect) = entry.state {
+                            if !matches!(entry.operation, PlanOperation::AppendKv)
+                                || effect.read.slot != state.slot
+                            {
+                                return Err(Error::Unsupported(
+                                    "temporal interleaved attention effect".into(),
+                                ));
+                            }
+                            Self::effect(entry, &mut advanced)?;
+                            writes += 1;
+                        }
+                        if matches!(
+                            entry.operation,
+                            PlanOperation::AttentionQk { .. }
+                                | PlanOperation::AttentionValue { .. }
+                        ) && (writes != 1
+                            || Self::reads(entry).iter().any(|read| read.ordinal != 1))
+                        {
+                            return Err(Error::State(
+                                "temporal attention must read actual proposed snapshot".into(),
+                            ));
+                        }
+                        if matches!(entry.operation, PlanOperation::KvPrefixGuard { .. })
+                            && !std::ptr::eq(entry, &nodes[index])
+                        {
+                            return Err(Error::Unsupported(
+                                "temporal nested attention guard".into(),
+                            ));
+                        }
+                        if Self::reads(entry)
+                            .iter()
+                            .any(|read| read.slot != state.slot || read.ordinal > 1)
+                        {
+                            return Err(Error::Unsupported(
+                                "temporal foreign attention state read".into(),
+                            ));
+                        }
+                    }
+                    if writes != 1 {
+                        return Err(Error::State("temporal attention append coverage".into()));
+                    }
+                    Stage::Causal {
+                        nodes: index..end.min(count),
+                        slots: vec![state.slot],
+                    }
+                }
+                PlanOperation::Conv { .. } | PlanOperation::Recurrent { .. } => {
+                    Self::known(node)?;
+                    let slot = Self::effect(node, &mut advanced)?;
+                    Stage::Causal {
+                        nodes: index..index + 1,
+                        slots: vec![slot],
+                    }
+                }
+                PlanOperation::ObserveState { state, .. } => {
+                    if node.state.is_some() || state.ordinal != 1 || !advanced.contains(&state.slot)
+                    {
+                        return Err(Error::Unsupported(
+                            "temporal observation requires retained proposed version".into(),
+                        ));
+                    }
+                    Stage::Causal {
+                        nodes: index..index + 1,
+                        slots: vec![state.slot],
+                    }
+                }
+                _ => {
+                    Self::known(node)?;
+                    if node.state.is_some()
+                        || !Self::reads(node).is_empty()
+                        || matches!(
+                            node.operation,
+                            PlanOperation::AppendKv | PlanOperation::AttentionSoftmax { .. }
+                        )
+                    {
+                        return Err(Error::Unsupported(
+                            "temporal hidden causal dependency".into(),
+                        ));
+                    }
+                    Stage::Flat(index)
+                }
+            };
+            let range = match &stage {
+                Stage::Flat(i) => *i..*i + 1,
+                Stage::Causal { nodes, .. } => nodes.clone(),
+            };
+            // SSA closure is checked independently of the state/effect tags.
+            for (i, node) in nodes[range.clone()].iter().enumerate() {
+                if node
+                    .inputs
+                    .iter()
+                    .any(|id| id.0 as usize >= values || !produced.contains(&id.0))
+                    || node
+                        .outputs
+                        .iter()
+                        .any(|id| id.0 as usize >= values || produced.contains(&id.0))
+                {
+                    return Err(Error::State(
+                        "temporal value dependency/producer closure".into(),
+                    ));
+                }
+                for id in &node.outputs {
+                    produced.insert(id.0);
+                }
+                if matches!(stage, Stage::Flat(_)) {
+                    flat.insert(range.start + i);
+                } else {
+                    causal.insert(range.start + i);
+                }
+            }
+            index = range.end;
+            stages.push(stage);
+        }
+        Ok(Self {
+            stages,
+            flat,
+            causal,
+            count,
+        })
+    }
+    fn effect(node: &PlanNode, advanced: &mut BTreeSet<StateSlot>) -> Result<StateSlot> {
+        let effect = node
+            .state
+            .ok_or_else(|| Error::State("temporal effect absent".into()))?;
+        if effect.read.slot != effect.write.slot
+            || effect.read.ordinal != 0
+            || effect.write.ordinal != 1
+            || !advanced.insert(effect.read.slot)
+        {
+            return Err(Error::State(
+                "temporal repeated/noncanonical state effect".into(),
+            ));
+        }
+        if !matches!(
+            (&node.operation, effect.read.slot),
+            (PlanOperation::Conv { .. }, StateSlot::Conv(_))
+                | (PlanOperation::Recurrent { .. }, StateSlot::Recurrent(_))
+                | (PlanOperation::AppendKv, StateSlot::Kv(_))
+        ) {
+            return Err(Error::State("temporal operation/domain mismatch".into()));
+        }
+        Ok(effect.read.slot)
+    }
+    fn reads(node: &PlanNode) -> Vec<StateVersion> {
+        let mut reads = node
+            .state
+            .map(|effect| vec![effect.read])
+            .unwrap_or_default();
+        match node.operation {
+            PlanOperation::KvPrefixGuard { state, .. }
+            | PlanOperation::AttentionQk { state, .. }
+            | PlanOperation::AttentionValue { state, .. }
+            | PlanOperation::ObserveState { state, .. } => reads.push(state),
+            _ => {}
+        }
+        reads
+    }
+    fn known(node: &PlanNode) -> Result<()> {
+        match node.operation {
+            PlanOperation::Embed { .. }
+            | PlanOperation::Project { .. }
+            | PlanOperation::View(_)
+            | PlanOperation::Contiguous
+            | PlanOperation::Rms { .. }
+            | PlanOperation::Silu { .. }
+            | PlanOperation::Add
+            | PlanOperation::SigmoidGate
+            | PlanOperation::Conv { .. }
+            | PlanOperation::Coefficients { .. }
+            | PlanOperation::Recurrent { .. }
+            | PlanOperation::KvPrefixGuard { .. }
+            | PlanOperation::Rope { .. }
+            | PlanOperation::AppendKv
+            | PlanOperation::AttentionQk { .. }
+            | PlanOperation::AttentionSoftmax { .. }
+            | PlanOperation::AttentionValue { .. }
+            | PlanOperation::CheckFinite { .. }
+            | PlanOperation::Observe { .. }
+            | PlanOperation::ObserveState { .. } => Ok(()),
+            _ => Err(Error::Unsupported(
+                "temporal unqualified/new operation".into(),
+            )),
+        }
+    }
+}
+
+/// Fixed original B4 buffers/prebound quantizers. The enclosing workspace owns
+/// every lease, including unused B4 state/greedy allocations charged at cold load.
+pub(crate) struct TemporalProgram {
+    flat: BatchProgram,
+    dependencies: Dependencies,
+}
+struct Banks<'a> {
+    program: &'a mut BatchProgram,
+    slots: &'a [StateSlot],
+    flipped: bool,
+}
+impl Banks<'_> {
+    fn flip(&mut self) {
+        for binding in &mut self.program.packed {
+            if self.slots.contains(&binding.slot) {
+                std::mem::swap(&mut binding.prior, &mut binding.candidate);
+            }
+        }
+        self.flipped = !self.flipped;
+    }
+}
+impl Drop for Banks<'_> {
+    fn drop(&mut self) {
+        if self.flipped {
+            self.flip();
+        }
+    }
+}
+
+impl TemporalProgram {
+    #[cfg(test)]
+    pub(crate) fn arm_inactive_nan_probe() {
+        INACTIVE_NAN.with(|flag| assert!(!flag.replace(true), "temporal probe already armed"));
+    }
+    #[cfg(test)]
+    fn poison_intermediate(
+        client: &Client,
+        program: &BatchProgram,
+        slot: StateSlot,
+    ) -> Result<usize> {
+        let binding = program.packed.iter().find(|b| b.slot == slot).unwrap();
+        let StateBuffers::Kv { keys, .. } = &binding.candidate else {
+            return Ok(0);
+        };
+        let offset = keys
+            .elements
+            .checked_sub(1)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("temporal NaN offset".into()))?;
+        let start = keys
+            .handle
+            .offset_start
+            .unwrap_or(0)
+            .checked_add(offset as u64)
+            .ok_or_else(|| Error::Limit("temporal NaN start".into()))?;
+        let end = start
+            .checked_add(4)
+            .ok_or_else(|| Error::Limit("temporal NaN end".into()))?;
+        let mut view = keys.handle.clone();
+        view.offset_start = Some(start);
+        view.offset_end = Some(
+            view.size()
+                .checked_sub(end)
+                .ok_or_else(|| Error::State("temporal NaN extent".into()))?,
+        );
+        BufferRange::new(&view)?;
+        unsafe {
+            batch_tokens::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&view, 1),
+                0x7fc00001u32,
+                0,
+                0,
+                0,
+                0,
+                0,
+                0,
+                0,
+                1,
+            );
+        }
+        Ok(1)
+    }
+    pub(crate) fn preflight(original: &ResidentProgram) -> Result<ResidentPreflight> {
+        Dependencies::new(&original.plan)?;
+        if original.plan.prefix_capacity() != 512 {
+            return Err(Error::Unsupported(
+                "temporal original capacity512 only".into(),
+            ));
+        }
+        BatchProgram::preflight(original, 4)
+    }
+    pub(crate) fn prepare(
+        client: &Client,
+        original: &ResidentProgram,
+        causal: &BatchProgram,
+        allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
+    ) -> Result<Self> {
+        if causal.rows != 2
+            || causal.graph_pinned()
+            || !Arc::ptr_eq(&causal.inner.plan_lineage, &original.plan_lineage)
+        {
+            return Err(Error::State(
+                "temporal same original plan/B2 lineage".into(),
+            ));
+        }
+        Self::preflight(original)?;
+        let flat = BatchProgram::prepare(client, original, 4, allocate)?;
+        Ok(Self {
+            flat,
+            dependencies: Dependencies::new(&original.plan)?,
+        })
+    }
+    pub(crate) fn budget(&self) -> &ResidentPreflight {
+        self.flat.budget()
+    }
+    #[cfg(test)]
+    pub(crate) fn expected_native_projections(&self) -> usize {
+        self.dependencies
+            .flat
+            .iter()
+            .filter(|&&i| matches!(self.flat.nodes[i].operation, PlanOperation::Project { .. }))
+            .count()
+            + 2 * self
+                .dependencies
+                .causal
+                .iter()
+                .filter(|&&i| matches!(self.flat.nodes[i].operation, PlanOperation::Project { .. }))
+                .count()
+    }
+    /// Diagnostic metadata only; caller must retain actual leases and use the
+    /// original queue/read-only completion before inspecting T0's packed bank.
+    #[cfg(test)]
+    pub(crate) fn first_state_buffers(causal: &BatchProgram) -> Vec<(StateSlot, StateBuffers)> {
+        causal
+            .packed
+            .iter()
+            .map(|binding| (binding.slot, binding.candidate.clone()))
+            .collect()
+    }
+    pub(crate) fn owned_handles(&self) -> Vec<Handle> {
+        self.flat.owned_handles()
+    }
+    pub(crate) fn execution_handles(&self) -> Vec<Handle> {
+        self.flat.execution_handles(ResidentOutput::StateOnly)
+    }
+    pub(crate) fn validate(
+        &self,
+        client: &Client,
+        causal: &BatchProgram,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<()> {
+        if causal.rows != 2
+            || self.flat.rows != 4
+            || steps.len() != 2
+            || causal.graph_pinned()
+            || self.flat.graph_pinned()
+            || !Arc::ptr_eq(&self.flat.inner.plan_lineage, &causal.inner.plan_lineage)
+            || !Arc::ptr_eq(&self.flat.inner._source_owner, &causal.inner._source_owner)
+            || self.dependencies.count != causal.inner.plan.state_nodes().len()
+            || self.dependencies.flat.is_empty()
+            || self.dependencies.causal.is_empty()
+        {
+            return Err(Error::State("temporal closed B2/T2 layout/lineage".into()));
+        }
+        causal.validate_state_chunk(client, steps, bindings)?;
+        let causal_handles = causal
+            .execution_handles(ResidentOutput::StateOnly)
+            .into_iter()
+            .map(|h| BufferRange::new(&h))
+            .collect::<Result<Vec<_>>>()?;
+        for handle in self.flat.owned_handles() {
+            let range = BufferRange::new(&handle)?;
+            if range.service != client.service_id()
+                || causal_handles.iter().any(|other| range.overlaps(other))
+            {
+                return Err(Error::State(
+                    "temporal private layouts alias/foreign service".into(),
+                ));
+            }
+        }
+        DeviceBuffer::new(self.flat.inner.status.clone(), 1).check(client, 1)?;
+        DeviceBuffer::new(self.flat.inner.token.clone(), 4).check(client, 4)?;
+        DeviceBuffer::new(self.flat.positions.clone(), 4).check(client, 4)?;
+        for (index, shape) in causal.inner.plan.values().iter().enumerate() {
+            let words = shape.capacity_elements()?;
+            let small = &causal.inner.values[index];
+            let large = &self.flat.inner.values[index];
+            small.check(
+                client,
+                words
+                    .checked_mul(2)
+                    .ok_or_else(|| Error::Limit("temporal B2 value bound".into()))?,
+            )?;
+            large.check(
+                client,
+                words
+                    .checked_mul(4)
+                    .ok_or_else(|| Error::Limit("temporal B4 value bound".into()))?,
+            )?;
+            for teacher in 0..2 {
+                Self::bridge_span(small, large, teacher)?;
+            }
+        }
+        // Every actual stateless node is checked against the sealed partition.
+        // No fake B4 RootBindings, StateBindings or future Ready are constructed.
+        for &index in &self.dependencies.flat {
+            let node = &self.flat.nodes[index];
+            if node.state.is_some() || !Dependencies::reads(node).is_empty() {
+                return Err(Error::State(
+                    "temporal flat state dependency changed".into(),
+                ));
+            }
+        }
+        Ok(())
+    }
+    fn bridge_span(small: &DeviceBuffer, large: &DeviceBuffer, teacher: usize) -> Result<usize> {
+        if teacher >= 2 || small.elements.checked_mul(2) != Some(large.elements) {
+            return Err(Error::State("temporal bridge true row geometry".into()));
+        }
+        let offset = teacher
+            .checked_mul(small.elements)
+            .ok_or_else(|| Error::Limit("temporal bridge offset".into()))?;
+        if offset
+            .checked_add(small.elements)
+            .is_none_or(|end| end > large.elements)
+        {
+            return Err(Error::Limit("temporal bridge range".into()));
+        }
+        let a = BufferRange::new(&small.handle)?;
+        let b = BufferRange::new(&large.handle)?;
+        if a.service != b.service
+            || a.overlaps(&b)
+            || a.bytes.end - a.bytes.start < bytes(small.elements, 4)? as u64
+            || b.bytes.end - b.bytes.start < bytes(large.elements, 4)? as u64
+        {
+            return Err(Error::State("temporal bridge binding/alias".into()));
+        }
+        Ok(offset)
+    }
+    fn bridge(
+        &self,
+        client: &Client,
+        causal: &BatchProgram,
+        id: ValueId,
+        teacher: usize,
+        to_flat: bool,
+    ) -> Result<usize> {
+        let small = &causal.inner.values[id.0 as usize];
+        let large = &self.flat.inner.values[id.0 as usize];
+        let offset = Self::bridge_span(small, large, teacher)?;
+        Ok(if to_flat {
+            causal.copy_dense(
+                client,
+                &small.handle,
+                small.elements,
+                0,
+                &large.handle,
+                large.elements,
+                offset,
+                small.elements,
+            )
+        } else {
+            causal.copy_dense(
+                client,
+                &large.handle,
+                large.elements,
+                offset,
+                &small.handle,
+                small.elements,
+                0,
+                small.elements,
+            )
+        })
+    }
+    fn certify_candidate(
+        client: &Client,
+        program: &BatchProgram,
+        slot: StateSlot,
+    ) -> Result<usize> {
+        let binding = program
+            .packed
+            .iter()
+            .find(|binding| binding.slot == slot)
+            .ok_or_else(|| Error::State("temporal effect packed domain".into()))?;
+        Ok(binding
+            .candidate
+            .buffers()
+            .iter()
+            .map(|value| program.certify(client, &value.handle, value.elements))
+            .sum())
+    }
+    fn selected_status(&self, causal: &BatchProgram, submission: &mut ResidentSubmission) {
+        submission.quant_status.clear();
+        submission.embedding_status.clear();
+        for (program, selected) in [
+            (&self.flat, &self.dependencies.flat),
+            (causal, &self.dependencies.causal),
+        ] {
+            for &index in selected {
+                match &program.nodes[index].operation {
+                    PlanOperation::Project { tensor } => {
+                        if let Some(layout) = program.inner.tensors[&tensor.name].layout {
+                            let flags = &program.inner.node_status[index];
+                            submission.quant_status.push((
+                                flags.block.as_ref().unwrap().clone(),
+                                program.rows * layout.blocks,
+                                flags.row.as_ref().unwrap().clone(),
+                                program.rows * layout.rows,
+                            ));
+                        }
+                    }
+                    PlanOperation::Embed { tensor, .. } => submission.embedding_status.push((
+                        program.inner.node_status[index]
+                            .block
+                            .as_ref()
+                            .unwrap()
+                            .clone(),
+                        program.rows * tensor.columns,
+                    )),
+                    _ => {}
+                }
+            }
+        }
+    }
+    pub(crate) fn launch(
+        &self,
+        client: &Client,
+        causal: &mut BatchProgram,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<ResidentSubmission> {
+        self.validate(client, causal, steps, bindings)?;
+        let prefix = causal.launch_inputs(client, &steps[0], bindings, None)?;
+        let flat_parameters = steps.iter().flatten().copied().collect::<Vec<_>>();
+        unsafe {
+            crate::program_kernels::clear_step_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&self.flat.inner.status, 1),
+            );
+        }
+        let mut body = BatchBodyReceipt {
+            dispatches: 1 + self.flat.stage_teacher_parameters(client, &flat_parameters),
+            fused_dispatches: 0,
+        };
+        for stage in &self.dependencies.stages {
+            match stage {
+                Stage::Flat(index) => {
+                    let receipt =
+                        self.flat
+                            .launch_checked_node(client, *index, &flat_parameters)?;
+                    body.dispatches += receipt.dispatches;
+                    body.fused_dispatches += receipt.fused_dispatches;
+                }
+                Stage::Causal { nodes, slots } => {
+                    let mut banks = Banks {
+                        program: causal,
+                        slots,
+                        flipped: false,
+                    };
+                    for (teacher, parameters) in steps.iter().enumerate() {
+                        if teacher == 1 {
+                            banks.flip();
+                        }
+                        body.dispatches +=
+                            banks.program.stage_teacher_parameters(client, parameters);
+                        for index in nodes.clone() {
+                            let node = &self.flat.inner.plan.nodes()[index];
+                            for id in &node.inputs {
+                                body.dispatches +=
+                                    self.bridge(client, banks.program, *id, teacher, false)?;
+                            }
+                            let receipt = banks
+                                .program
+                                .launch_checked_node(client, index, parameters)?;
+                            body.dispatches += receipt.dispatches;
+                            body.fused_dispatches += receipt.fused_dispatches;
+                            // Retain every transient finite/status decision before
+                            // its actual bank/value can be reused or overwritten.
+                            if let Some(effect) = node.state {
+                                #[cfg(test)]
+                                if teacher == 0
+                                    && matches!(effect.write.slot, StateSlot::Kv(_))
+                                    && INACTIVE_NAN.with(|flag| flag.replace(false))
+                                {
+                                    if parameters.iter().any(|p| p.valid_tokens >= 512) {
+                                        return Err(Error::State(
+                                            "temporal NaN probe requires inactive KV".into(),
+                                        ));
+                                    }
+                                    body.dispatches += Self::poison_intermediate(
+                                        client,
+                                        banks.program,
+                                        effect.write.slot,
+                                    )?;
+                                }
+                                body.dispatches += Self::certify_candidate(
+                                    client,
+                                    banks.program,
+                                    effect.write.slot,
+                                )?;
+                            }
+                            for id in &node.outputs {
+                                body.dispatches +=
+                                    self.bridge(client, banks.program, *id, teacher, true)?;
+                            }
+                        }
+                    }
+                }
+            }
+        }
+        // TWO persistent OR statuses; one real terminal status/readback. No
+        // per-node/per-teacher host synchronization or status reset occurred.
+        unsafe {
+            crate::resident_bindings::merge_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&self.flat.inner.status, 1),
+                arg(&causal.inner.status, 1),
+                1,
+            );
+        }
+        body.dispatches += 1;
+        let slots = causal
+            .packed
+            .iter()
+            .map(|state| state.slot)
+            .collect::<Vec<_>>();
+        let mut final_banks = Banks {
+            program: causal,
+            slots: &slots,
+            flipped: false,
+        };
+        final_banks.flip(); // T1 lives in original prior A; candidate now A.
+        let mut submission = final_banks.program.launch_outputs(
+            client,
+            bindings,
+            ResidentOutput::StateOnly,
+            prefix,
+            body,
+        )?;
+        self.selected_status(final_banks.program, &mut submission);
+        Ok(submission)
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use zerogravity_core::accelerator::program::{
+        AttentionGeometry, ConvDomain, KvDomain, ParameterRef, ParameterSlot, StateEffect,
+        StaticVectorId,
+    };
+    fn node(
+        operation: PlanOperation,
+        inputs: &[u32],
+        outputs: &[u32],
+        state: Option<StateEffect>,
+    ) -> PlanNode {
+        PlanNode {
+            name: "synthetic descriptor".into(),
+            operation,
+            inputs: inputs.iter().copied().map(ValueId).collect(),
+            outputs: outputs.iter().copied().map(ValueId).collect(),
+            state,
+        }
+    }
+    #[test]
+    fn hidden_reads_are_not_flat_and_unproduced_inputs_refuse() {
+        let state = StateVersion {
+            slot: StateSlot::Kv(KvDomain(0)),
+            ordinal: 1,
+        };
+        let qk = node(
+            PlanOperation::AttentionQk {
+                state,
+                geometry: AttentionGeometry {
+                    width: 128,
+                    query_heads: 2,
+                    kv_heads: 1,
+                },
+                valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                scale: 1.0,
+            },
+            &[],
+            &[],
+            None,
+        );
+        assert_eq!(Dependencies::reads(&qk), vec![state]);
+        assert!(Dependencies::nodes(&[qk], 1, 0).is_err());
+        let add = node(PlanOperation::Add, &[0, 1], &[2], None);
+        assert!(Dependencies::nodes(&[add], 1, 3).is_err());
+    }
+    #[test]
+    fn duplicate_effect_and_reading_overwritten_initial_version_refuse() {
+        let slot = StateSlot::Conv(ConvDomain(0));
+        let effect = StateEffect {
+            read: StateVersion { slot, ordinal: 0 },
+            write: StateVersion { slot, ordinal: 1 },
+        };
+        let conv = node(
+            PlanOperation::Conv {
+                weights: StaticVectorId(0),
+            },
+            &[],
+            &[],
+            Some(effect),
+        );
+        let observe = node(
+            PlanOperation::ObserveState {
+                layer: 0,
+                stage: "synthetic",
+                state: effect.read,
+            },
+            &[],
+            &[],
+            None,
+        );
+        assert!(Dependencies::nodes(&[conv.clone(), conv.clone()], 2, 0).is_err());
+        assert!(Dependencies::nodes(&[conv, observe], 2, 0).is_err());
+    }
+    #[test]
+    fn genuine_teacher_axis_is_not_a_state_or_strided_kv_axis() {
+        assert_eq!(
+            (0..2)
+                .flat_map(|teacher| (0..2).map(move |row| teacher * 2 + row))
+                .collect::<Vec<_>>(),
+            vec![0, 1, 2, 3]
+        );
+        let slot = StateSlot::Conv(ConvDomain(0));
+        let effect = StateEffect {
+            read: StateVersion { slot, ordinal: 0 },
+            write: StateVersion { slot, ordinal: 1 },
+        };
+        let nodes = vec![
+            node(PlanOperation::Contiguous, &[], &[0], None),
+            node(
+                PlanOperation::Conv {
+                    weights: StaticVectorId(0),
+                },
+                &[0],
+                &[1],
+                Some(effect),
+            ),
+            node(PlanOperation::Silu { multiplied: false }, &[1], &[2], None),
+            node(
+                PlanOperation::ObserveState {
+                    layer: 0,
+                    stage: "synthetic",
+                    state: effect.write,
+                },
+                &[],
+                &[],
+                None,
+            ),
+        ];
+        let descriptor = Dependencies::nodes(&nodes, 4, 3).unwrap();
+        assert_eq!(descriptor.flat, BTreeSet::from([0, 2]));
+        assert_eq!(descriptor.causal, BTreeSet::from([1, 3]));
+        assert_eq!(descriptor.stages.len(), 4);
+    }
+    #[test]
+    fn attention_keeps_hidden_reads_and_softmax_in_one_causal_region_and_clips_original_cut() {
+        let slot = StateSlot::Kv(KvDomain(0));
+        let prior = StateVersion { slot, ordinal: 0 };
+        let proposed = StateVersion { slot, ordinal: 1 };
+        let geometry = AttentionGeometry {
+            width: 128,
+            query_heads: 2,
+            kv_heads: 1,
+        };
+        let nodes = vec![
+            node(PlanOperation::Contiguous, &[], &[0], None),
+            node(
+                PlanOperation::KvPrefixGuard {
+                    state: prior,
+                    position: ParameterRef(ParameterSlot::Position),
+                },
+                &[],
+                &[],
+                None,
+            ),
+            node(
+                PlanOperation::AppendKv,
+                &[0, 0],
+                &[],
+                Some(StateEffect {
+                    read: prior,
+                    write: proposed,
+                }),
+            ),
+            node(
+                PlanOperation::AttentionQk {
+                    state: proposed,
+                    geometry,
+                    valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                    scale: 1.0,
+                },
+                &[0],
+                &[1, 2],
+                None,
+            ),
+            node(
+                PlanOperation::AttentionSoftmax {
+                    heads: 2,
+                    valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                },
+                &[2],
+                &[3],
+                None,
+            ),
+            node(
+                PlanOperation::AttentionValue {
+                    state: proposed,
+                    geometry,
+                    valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
+                    padding: 256,
+                },
+                &[3],
+                &[4],
+                None,
+            ),
+        ];
+        let full = Dependencies::nodes(&nodes, 6, 5).unwrap();
+        assert_eq!(
+            full.stages[1],
+            Stage::Causal {
+                nodes: 1..6,
+                slots: vec![slot]
+            }
+        );
+        let cut = Dependencies::nodes(&nodes, 3, 5).unwrap();
+        assert_eq!(
+            cut.stages[1],
+            Stage::Causal {
+                nodes: 1..3,
+                slots: vec![slot]
+            }
+        );
+        assert_eq!(cut.causal, BTreeSet::from([1, 2]));
+        let mut wrong = nodes.clone();
+        wrong[3].inputs = vec![ValueId(4)];
+        assert!(Dependencies::nodes(&wrong, 6, 5).is_err());
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -64,6 +64,10 @@
 
 #[cfg(all(feature = "qualification", feature = "cuda"))]
 mod chunk;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod temporal;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+pub(crate) use temporal::TemporalProgram;
 
 impl BatchProgram {
     /// New allocations only: the original source handles remain shared.
@@ -261,6 +265,8 @@
         let positions = allocator.empty(ResidentBufferRole::Parameter, bytes(rows, 4)?)?;
         let mut inner = ResidentProgram {
             _source_owner: Arc::clone(&original._source_owner),
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            plan_lineage: Arc::clone(&original.plan_lineage),
             plan: original.plan.clone(),
             preflight,
             tensors,
@@ -599,48 +605,61 @@
         parameters: &[RuntimeParameters],
         output: ResidentOutput,
     ) -> Result<BatchBodyReceipt> {
-        let mut dispatches = 0;
-        // One actual native projection-dot launch per Project. Activation
-        // preparation/status stages remain in the total dispatch count only.
-        let mut fused_dispatches = 0;
-        let count = output.nodes(&self.inner.plan).len();
-        for (index, node) in self.nodes[..count].iter().enumerate() {
-            dispatches += self.launch_node(client, index, node, parameters)?;
-            if matches!(node.operation, PlanOperation::Project { .. }) {
-                fused_dispatches += 1;
-            }
-            if matches!(node.operation, PlanOperation::View(_)) {
-                continue;
-            }
-            for (out_index, id) in node.outputs.iter().enumerate() {
-                let value = &self.inner.values[id.0 as usize];
-                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
-                    && out_index == 1
-                {
-                    unsafe {
-                        crate::resident_bindings::certify_causal_rows::launch_unchecked(
-                            client,
-                            grid(value.elements),
-                            threads(),
-                            arg(&value.handle, value.elements),
-                            arg(&self.inner.status, 1),
-                            arg(&self.positions, self.rows),
-                            geometry.query_heads / self.rows,
-                            self.rows,
-                            self.inner.plan.prefix_capacity(),
-                        );
-                    }
-                } else {
-                    dispatches += self.certify(client, &value.handle, value.elements);
-                    continue;
-                }
-                dispatches += 1;
-            }
-        }
-        Ok(BatchBodyReceipt {
-            dispatches,
-            fused_dispatches,
-        })
+        let mut receipt = BatchBodyReceipt {
+            dispatches: 0,
+            fused_dispatches: 0,
+        };
+        for index in 0..output.nodes(&self.inner.plan).len() {
+            let node = self.launch_checked_node(client, index, parameters)?;
+            receipt.dispatches += node.dispatches;
+            receipt.fused_dispatches += node.fused_dispatches;
+        }
+        Ok(receipt)
+    }
+
+    /// The original single node and its original output certifications. Shared
+    /// by the unchanged ordered body and the closed dependency lowering.
+    fn launch_checked_node(
+        &self,
+        client: &Client,
+        index: usize,
+        parameters: &[RuntimeParameters],
+    ) -> Result<BatchBodyReceipt> {
+        let node = self
+            .nodes
+            .get(index)
+            .ok_or_else(|| Error::State("native node index".into()))?;
+        let mut receipt = BatchBodyReceipt {
+            dispatches: self.launch_node(client, index, node, parameters)?,
+            fused_dispatches: usize::from(matches!(node.operation, PlanOperation::Project { .. })),
+        };
+        if matches!(node.operation, PlanOperation::View(_)) {
+            return Ok(receipt);
+        }
+        for (out_index, id) in node.outputs.iter().enumerate() {
+            let value = &self.inner.values[id.0 as usize];
+            if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
+                && out_index == 1
+            {
+                unsafe {
+                    crate::resident_bindings::certify_causal_rows::launch_unchecked(
+                        client,
+                        grid(value.elements),
+                        threads(),
+                        arg(&value.handle, value.elements),
+                        arg(&self.inner.status, 1),
+                        arg(&self.positions, self.rows),
+                        geometry.query_heads / self.rows,
+                        self.rows,
+                        self.inner.plan.prefix_capacity(),
+                    );
+                }
+                receipt.dispatches += 1;
+            } else {
+                receipt.dispatches += self.certify(client, &value.handle, value.elements);
+            }
+        }
+        Ok(receipt)
     }
 
     fn launch_outputs(
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -40,7 +40,7 @@
 mod batch;
 pub(crate) use batch::BatchProgram;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
-pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};
+pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay, TemporalProgram};
 
 #[cfg(feature = "qualification")]
 #[path = "resident_projection_qualification.rs"]
@@ -302,6 +302,8 @@
 pub(crate) struct ResidentProgram {
     // The immutable original allocation remains alive for every resident handle.
     _source_owner: Arc<dyn TensorSource>,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    plan_lineage: Arc<()>,
     plan: ForwardPlan,
     preflight: ResidentPreflight,
     tensors: BTreeMap<String, ResidentTensor>,
@@ -977,6 +979,8 @@
             .collect::<Result<Vec<_>>>()?;
         let mut program = Self {
             _source_owner: model,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            plan_lineage: Arc::new(()),
             plan,
             preflight,
             tensors,

````

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/ROOT_WIRING.patch

Original UTF-8 bytes: 332; SHA256 `a73246fae76e296b84d5b1673a49c1ee926c6bb47e4ec47e579bc5931c2c6f1d`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
@@ -673,3 +673,5 @@
 fn qwen2_chunk_certificate_refusal() {
     futures_lite::future::block_on(qualify_certificate_refusal("qwen35-2b")).unwrap();
 }
+
+mod temporal;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/NEW_DELTA.patch

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

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/ROOT_WIRING.patch

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

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/BYTE_BUDGET_ONLY.patch

Original UTF-8 bytes: 467; SHA256 `e127b2795c7dcbbd6eabb2379d7da4a365b625d364cbcc1a6edcf78f3943bb8b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-gpu/src/ordered_silu/qualification.rs
+++ crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -567,8 +567,8 @@
                 "ordered-silu-synthetic-operator",
                 "original-physical-oracle",
             )?,
-            256 * 1024 * *2,
-            640 * 1024 * *2,
+            256 * 1024 * 1024,
+            640 * 1024 * 1024,
         )?;
         let key = service.physical_key();
         let mut compared = 0;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/INITIALIZATION_MATH_FLIGHTS.patch

Original UTF-8 bytes: 1777; SHA256 `7cfbe9bb8038056096ea94b88ebc74462001d7d0afc2a8d06e6259af90d3787b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-gpu/src/ordered_silu/qualification.rs
+++ crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -271,8 +271,9 @@
         }],
         None,
     )?;
-    let mut slot = ticket.wait().await?;
-    slot.begin_submission()?;
+    // Initialization owns its original begun slot until physical Ready.
+    let mut init_slot = ticket.wait().await?;
+    init_slot.begin_submission()?;
     let mut init = service.begin_initialization(0)?;
     service
         .client()
@@ -294,6 +295,7 @@
         return Err(Error::State("ordered SiLU initialization coverage".into()));
     }
     drop(initialized);
+    init_slot.finish()?;
     let handles = leases
         .iter()
         .zip(sizes)
@@ -442,6 +444,20 @@
         .collect::<Result<Vec<_>>>()?;
     let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
     let uploads = service.uploaded_value_bytes();
+    let mxcsr = cpu_math_environment()?;
+    // All bindings/readbacks/refusals are checked before this separate math
+    // flight. Same prepaid envelope, original fair queue and retained leases.
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
     let flight = service.submit(0, &accesses)?;
     let old = original(
         client,
@@ -472,7 +488,6 @@
         }
         return Ok(0);
     }
-    let mxcsr = cpu_math_environment()?;
     let mut checked = 0;
     let mut rejected_status = false;
     let result=flight.complete_with_readback(&leases[2..],&reads,|read| {

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/QUALIFIER_V2.review.patch

Original UTF-8 bytes: 2122; SHA256 `6a67cb618e3140c6914a7ac682c5df5bf49a9b26809950d3f6ea2e396266da1c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-gpu/src/ordered_silu/qualification.rs
+++ crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -271,8 +271,9 @@
         }],
         None,
     )?;
-    let mut slot = ticket.wait().await?;
-    slot.begin_submission()?;
+    // Initialization owns its original begun slot until physical Ready.
+    let mut init_slot = ticket.wait().await?;
+    init_slot.begin_submission()?;
     let mut init = service.begin_initialization(0)?;
     service
         .client()
@@ -294,6 +295,7 @@
         return Err(Error::State("ordered SiLU initialization coverage".into()));
     }
     drop(initialized);
+    init_slot.finish()?;
     let handles = leases
         .iter()
         .zip(sizes)
@@ -442,6 +444,20 @@
         .collect::<Result<Vec<_>>>()?;
     let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
     let uploads = service.uploaded_value_bytes();
+    let mxcsr = cpu_math_environment()?;
+    // All bindings/readbacks/refusals are checked before this separate math
+    // flight. Same prepaid envelope, original fair queue and retained leases.
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
     let flight = service.submit(0, &accesses)?;
     let old = original(
         client,
@@ -472,7 +488,6 @@
         }
         return Ok(0);
     }
-    let mxcsr = cpu_math_environment()?;
     let mut checked = 0;
     let mut rejected_status = false;
     let result=flight.complete_with_readback(&leases[2..],&reads,|read| {
@@ -567,8 +582,8 @@
                 "ordered-silu-synthetic-operator",
                 "original-physical-oracle",
             )?,
-            256 * 1024 * *2,
-            640 * 1024 * *2,
+            256 * 1024 * 1024,
+            640 * 1024 * 1024,
         )?;
         let key = service.physical_key();
         let mut compared = 0;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/frozen/BYTE_BUDGET_ONLY.patch

Original UTF-8 bytes: 467; SHA256 `e127b2795c7dcbbd6eabb2379d7da4a365b625d364cbcc1a6edcf78f3943bb8b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-gpu/src/ordered_silu/qualification.rs
+++ crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -567,8 +567,8 @@
                 "ordered-silu-synthetic-operator",
                 "original-physical-oracle",
             )?,
-            256 * 1024 * *2,
-            640 * 1024 * *2,
+            256 * 1024 * 1024,
+            640 * 1024 * 1024,
         )?;
         let key = service.physical_key();
         let mut compared = 0;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/frozen/INITIALIZATION_MATH_FLIGHTS.patch

Original UTF-8 bytes: 1777; SHA256 `7cfbe9bb8038056096ea94b88ebc74462001d7d0afc2a8d06e6259af90d3787b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-gpu/src/ordered_silu/qualification.rs
+++ crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -271,8 +271,9 @@
         }],
         None,
     )?;
-    let mut slot = ticket.wait().await?;
-    slot.begin_submission()?;
+    // Initialization owns its original begun slot until physical Ready.
+    let mut init_slot = ticket.wait().await?;
+    init_slot.begin_submission()?;
     let mut init = service.begin_initialization(0)?;
     service
         .client()
@@ -294,6 +295,7 @@
         return Err(Error::State("ordered SiLU initialization coverage".into()));
     }
     drop(initialized);
+    init_slot.finish()?;
     let handles = leases
         .iter()
         .zip(sizes)
@@ -442,6 +444,20 @@
         .collect::<Result<Vec<_>>>()?;
     let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
     let uploads = service.uploaded_value_bytes();
+    let mxcsr = cpu_math_environment()?;
+    // All bindings/readbacks/refusals are checked before this separate math
+    // flight. Same prepaid envelope, original fair queue and retained leases.
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
     let flight = service.submit(0, &accesses)?;
     let old = original(
         client,
@@ -472,7 +488,6 @@
         }
         return Ok(0);
     }
-    let mxcsr = cpu_math_environment()?;
     let mut checked = 0;
     let mut rejected_status = false;
     let result=flight.complete_with_readback(&leases[2..],&reads,|read| {

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/frozen/QUALIFIER_V2.review.patch

Original UTF-8 bytes: 2122; SHA256 `6a67cb618e3140c6914a7ac682c5df5bf49a9b26809950d3f6ea2e396266da1c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-gpu/src/ordered_silu/qualification.rs
+++ crates/zerogravity-gpu/src/ordered_silu/qualification.rs
@@ -271,8 +271,9 @@
         }],
         None,
     )?;
-    let mut slot = ticket.wait().await?;
-    slot.begin_submission()?;
+    // Initialization owns its original begun slot until physical Ready.
+    let mut init_slot = ticket.wait().await?;
+    init_slot.begin_submission()?;
     let mut init = service.begin_initialization(0)?;
     service
         .client()
@@ -294,6 +295,7 @@
         return Err(Error::State("ordered SiLU initialization coverage".into()));
     }
     drop(initialized);
+    init_slot.finish()?;
     let handles = leases
         .iter()
         .zip(sizes)
@@ -442,6 +444,20 @@
         .collect::<Result<Vec<_>>>()?;
     let quarantine_before = crate::physical_quarantine::quarantined_handles_for(client)?;
     let uploads = service.uploaded_value_bytes();
+    let mxcsr = cpu_math_environment()?;
+    // All bindings/readbacks/refusals are checked before this separate math
+    // flight. Same prepaid envelope, original fair queue and retained leases.
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
     let flight = service.submit(0, &accesses)?;
     let old = original(
         client,
@@ -472,7 +488,6 @@
         }
         return Ok(0);
     }
-    let mxcsr = cpu_math_environment()?;
     let mut checked = 0;
     let mut rejected_status = false;
     let result=flight.complete_with_readback(&leases[2..],&reads,|read| {
@@ -567,8 +582,8 @@
                 "ordered-silu-synthetic-operator",
                 "original-physical-oracle",
             )?,
-            256 * 1024 * *2,
-            640 * 1024 * *2,
+            256 * 1024 * 1024,
+            640 * 1024 * 1024,
         )?;
         let key = service.physical_key();
         let mut compared = 0;

````

