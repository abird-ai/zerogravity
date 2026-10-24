# Draft Patches — volume 2

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v3-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 37254; SHA256 `dc5328d086111b4e84307f78ab558eed7bd61908e12169752dc6e038d0e14113`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
@@ -31,6 +31,7 @@
 use zerogravity_core::quant::QuantKind;
 use zerogravity_core::{Error, Result};
 
+pub(crate) mod adversary;
 pub(crate) mod qualification;
 mod resident;
 pub(crate) use resident::{PreparedProjection, ProjectionReadback};
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -988,3 +988,21 @@
     )
     .await
 }
+
+/// Explicitly SYNTHETIC generic operator fixtures; no trained source/model claim.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub use crate::cpu_bf16_projection::adversary::{
+    ADVERSARY_WIDTHS as CPU_BF16_ADVERSARY_WIDTHS, AdversaryKind as CpuBf16AdversaryKind,
+    SYNTHETIC_ROWS as CPU_BF16_SYNTHETIC_ROWS, bad_geometry_name as cpu_bf16_bad_geometry_name,
+};
+
+/// Root-only same-kernel guarded proof or ONE deliberate nonfinite status case.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_adversaries(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    device_cap: u64,
+    fault: Option<(CpuBf16AdversaryKind, usize)>,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::adversary::qualify(device, source, device_cap, fault).await
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
@@ -0,0 +1,876 @@
+//! Source-authenticated SYNTHETIC generic operator adversaries. Never a model.
+//! Launches the exact candidate kernel on the original resident queue/leases.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentService};
+use cubecl::{Device, bytes::Bytes, server::Handle};
+use std::sync::atomic::{AtomicBool, Ordering};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+
+pub const ADVERSARY_WIDTHS: [usize; 6] = [64, 128, 1024, 2048, 5120, 8192];
+pub const SYNTHETIC_ROWS: usize = 5;
+const FIRST: usize = 1;
+const LAST: usize = 4;
+const OUTPUT_ROWS: usize = LAST - FIRST;
+const GUARD_WORDS: usize = 4;
+const GUARD_BYTES: usize = GUARD_WORDS * 4;
+const PROVIDER: usize = 64 * 1024 * 1024;
+const RAM: usize = 4 * 1024 * 1024;
+const EXPECTED_REFUSAL: &str = "synthetic BF16 arithmetic intentionally refuses Ready";
+
+pub fn bad_geometry_name(columns: usize) -> Result<String> {
+    if ![65, 8193].contains(&columns) {
+        return Err(malformed("explicit synthetic refused geometry"));
+    }
+    Ok(format!("synthetic.cpu64.cold-refuse-width{columns}"))
+}
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum AdversaryKind {
+    SignedZero,
+    EmptyTailZero,
+    GradualPositive,
+    GradualNegative,
+    SubnormalProduct,
+    SubnormalTieEven,
+    SubnormalTieOdd,
+    ReductionTree,
+    RoundTieEven,
+    RoundTieOdd,
+    RoundNegativeEven,
+    RoundNegativeOdd,
+    FmaUnderflow,
+    FmaOverflowCancellation,
+    Fp32Overflow,
+    Bf16Overflow,
+    SourceNan,
+    SourceInfinity,
+}
+impl AdversaryKind {
+    pub fn all(columns: usize) -> Result<Vec<Self>> {
+        if !ADVERSARY_WIDTHS.contains(&columns) {
+            return Err(malformed("synthetic fixture closed width"));
+        }
+        let mut kinds = vec![
+            Self::SignedZero,
+            Self::EmptyTailZero,
+            Self::GradualPositive,
+            Self::GradualNegative,
+            Self::SubnormalProduct,
+            Self::SubnormalTieEven,
+            Self::SubnormalTieOdd,
+            Self::ReductionTree,
+            Self::RoundTieEven,
+            Self::RoundTieOdd,
+            Self::RoundNegativeEven,
+            Self::RoundNegativeOdd,
+            Self::Fp32Overflow,
+            Self::Bf16Overflow,
+            Self::SourceNan,
+            Self::SourceInfinity,
+        ];
+        if columns >= 128 {
+            kinds.extend([Self::FmaUnderflow, Self::FmaOverflowCancellation]);
+        }
+        Ok(kinds)
+    }
+    pub fn name(self) -> &'static str {
+        match self {
+            Self::SignedZero => "signed-zero-input",
+            Self::EmptyTailZero => "mandatory-empty-tail-positive-zero",
+            Self::GradualPositive => "positive-bf16-subnormal",
+            Self::GradualNegative => "negative-bf16-subnormal",
+            Self::SubnormalProduct => "f32-subnormal-product",
+            Self::SubnormalTieEven => "bf16-subnormal-tie-even",
+            Self::SubnormalTieOdd => "bf16-subnormal-tie-odd",
+            Self::ReductionTree => "cpu64-horizontal-cancellation",
+            Self::RoundTieEven => "bf16-normal-tie-even",
+            Self::RoundTieOdd => "bf16-normal-tie-odd",
+            Self::RoundNegativeEven => "bf16-negative-tie-even",
+            Self::RoundNegativeOdd => "bf16-negative-tie-odd",
+            Self::FmaUnderflow => "fused-subnormal-cancellation",
+            Self::FmaOverflowCancellation => "fused-overflow-cancellation",
+            Self::Fp32Overflow => "finite-input-f32-overflow",
+            Self::Bf16Overflow => "finite-accumulation-bf16-overflow",
+            Self::SourceNan => "nonfinite-source-nan-cold-refusal",
+            Self::SourceInfinity => "nonfinite-source-inf-cold-refusal",
+        }
+    }
+    pub fn source_name(self, columns: usize) -> String {
+        format!("synthetic.cpu64.width{columns}.{}", self.name())
+    }
+    fn fault(self) -> bool {
+        matches!(self, Self::Fp32Overflow | Self::Bf16Overflow)
+    }
+    fn nonfinite_source(self) -> bool {
+        matches!(self, Self::SourceNan | Self::SourceInfinity)
+    }
+
+    /// Integer BF16 recipes only. ROOT seals these generic synthetic matrices;
+    /// expected arithmetic is later obtained from the ORIGINAL CPU operator.
+    pub fn fixture(self, columns: usize) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
+        if !Self::all(columns)?.contains(&self) {
+            return Err(malformed("synthetic case requires at least128 columns"));
+        }
+        let mut weights = vec![0u16; columns];
+        let mut input = vec![0u16; columns];
+        match self {
+            Self::SignedZero => {
+                weights.fill(0x3f80);
+                input.fill(0x8000);
+            }
+            Self::EmptyTailZero => {
+                weights.fill(0x0080);
+                input.fill(0x8080);
+            }
+            Self::GradualPositive => {
+                weights[0] = 0x0001;
+                input[0] = 0x3f80;
+            }
+            Self::GradualNegative => {
+                weights[0] = 0x8001;
+                input[0] = 0x3f80;
+            }
+            Self::SubnormalProduct => {
+                weights[0] = 0x0001;
+                input[0] = 0x3b00;
+            }
+            Self::SubnormalTieEven => {
+                weights[0] = 0x0001;
+                input[0] = 0x3f00;
+            }
+            Self::SubnormalTieOdd => {
+                weights[0] = 0x0003;
+                input[0] = 0x3f00;
+            }
+            Self::ReductionTree => {
+                weights[0] = 0x3f80;
+                weights[2] = 0x3f80;
+                weights[4] = 0x3f80;
+                input[0] = 0x4b80;
+                input[2] = 0x3f80;
+                input[4] = 0xcb80;
+            }
+            Self::RoundTieEven
+            | Self::RoundTieOdd
+            | Self::RoundNegativeEven
+            | Self::RoundNegativeOdd => {
+                weights[0] = 0x3f80;
+                weights[1] = 0x3f80;
+                input[0] = match self {
+                    Self::RoundTieEven => 0x3f80,
+                    Self::RoundTieOdd => 0x3f81,
+                    Self::RoundNegativeEven => 0xbf80,
+                    _ => 0xbf81,
+                };
+                input[1] = if matches!(self, Self::RoundNegativeEven | Self::RoundNegativeOdd) {
+                    0xbb80
+                } else {
+                    0x3b80
+                };
+            }
+            Self::FmaUnderflow => {
+                weights[0] = 0x0001;
+                input[0] = 0x3780;
+                weights[64] = 0x8001;
+                input[64] = 0x3700;
+            }
+            Self::FmaOverflowCancellation => {
+                weights[0] = 0x7f7f;
+                input[0] = 0xbf80;
+                weights[64] = 0x7f7f;
+                input[64] = 0x4000;
+            }
+            Self::Fp32Overflow => {
+                weights[0] = 0x7f7f;
+                input[0] = 0x4000;
+            }
+            Self::Bf16Overflow => {
+                weights[0] = 0x7f7f;
+                input[0] = 0x3f80;
+                weights[1] = 0x7b00;
+                input[1] = 0x3f80;
+            }
+            Self::SourceNan => {
+                weights[0] = 0x7fc1;
+                input[0] = 0x3f80;
+            }
+            Self::SourceInfinity => {
+                weights[0] = 0x7f80;
+                input[0] = 0x3f80;
+            }
+        }
+        let marker = columns - 1;
+        let mut full = Vec::with_capacity(SYNTHETIC_ROWS * columns * 2);
+        for row in 0..SYNTHETIC_ROWS {
+            for (column, &weight) in weights.iter().enumerate() {
+                let word = match row {
+                    0 => 0x3e23u16 + (column % 16) as u16,
+                    1 => weight,
+                    2 => {
+                        if column == marker {
+                            0x3f80
+                        } else {
+                            0
+                        }
+                    }
+                    3 => {
+                        if column == marker {
+                            0xbf80
+                        } else {
+                            0
+                        }
+                    }
+                    _ => 0xbe4du16 + (column % 16) as u16,
+                };
+                full.extend_from_slice(&word.to_le_bytes());
+            }
+        }
+        let inputs = (0..8)
+            .map(|request| {
+                let mut own = input.clone();
+                // Member0 keeps the exact arithmetic witness. Other members have
+                // distinct marker outputs, catching request/row broadcast aliases.
+                if request > 0 {
+                    own[marker] = 0x3f80 + request as u16;
+                }
+                own.into_iter().flat_map(u16::to_le_bytes).collect()
+            })
+            .collect();
+        Ok((full, inputs))
+    }
+    fn witness(self) -> Option<(u32, u32)> {
+        // Analytic source-free witnesses, not trained or reconstructed targets.
+        Some(match self {
+            Self::SignedZero | Self::EmptyTailZero | Self::FmaUnderflow => (0, 0),
+            Self::GradualPositive => (0x0001_0000, 0x0001_0000),
+            Self::GradualNegative => (0x8001_0000, 0x8001_0000),
+            Self::SubnormalProduct => (0x0000_0080, 0),
+            Self::SubnormalTieEven => (0x0000_8000, 0),
+            Self::SubnormalTieOdd => (0x0001_8000, 0x0002_0000),
+            Self::ReductionTree => (0x3f80_0000, 0x3f80_0000),
+            Self::RoundTieEven => (0x3f80_8000, 0x3f80_0000),
+            Self::RoundTieOdd => (0x3f81_8000, 0x3f82_0000),
+            Self::RoundNegativeEven => (0xbf80_8000, 0xbf80_0000),
+            Self::RoundNegativeOdd => (0xbf81_8000, 0xbf82_0000),
+            Self::FmaOverflowCancellation => (0x7f7f_0000, 0x7f7f_0000),
+            Self::Bf16Overflow => (0x7f7f_8000, 0x7f80_0000),
+            _ => return None,
+        })
+    }
+}
+
+fn rne(bits: u32) -> u32 {
+    bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000
+}
+fn word(bytes: &[u8]) -> u32 {
+    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
+}
+fn guard(binding: usize, side: usize, at: usize) -> u32 {
+    0x4a10_0041 + (binding as u32) * 0x10000 + (side as u32) * 0x100 + at as u32
+}
+fn guarded(binding: usize, payload: &[u8]) -> Vec<u8> {
+    let mut bytes = Vec::with_capacity(payload.len() + 2 * GUARD_BYTES);
+    for at in 0..GUARD_WORDS {
+        bytes.extend_from_slice(&guard(binding, 0, at).to_le_bytes());
+    }
+    bytes.extend_from_slice(payload);
+    for at in 0..GUARD_WORDS {
+        bytes.extend_from_slice(&guard(binding, 1, at).to_le_bytes());
+    }
+    bytes
+}
+fn check_guards(binding: usize, actual: &[u8], payload_bytes: usize) -> Result<()> {
+    if actual.len() != payload_bytes + 2 * GUARD_BYTES {
+        return Err(malformed("guarded physical readback length"));
+    }
+    for at in 0..GUARD_WORDS {
+        if word(&actual[4 * at..4 * at + 4]) != guard(binding, 0, at)
+            || word(
+                &actual[GUARD_BYTES + payload_bytes + 4 * at
+                    ..GUARD_BYTES + payload_bytes + 4 * at + 4],
+            ) != guard(binding, 1, at)
+        {
+            return Err(malformed("actual physical prefix/suffix canary changed"));
+        }
+    }
+    Ok(())
+}
+fn descriptor(source: &VerifiedTensorSource, name: &str) -> Result<SourceTensor> {
+    let matrix = source.matrix(name, MatrixAxes::OutputInput)?;
+    Ok(SourceTensor {
+        source: source.source_identity(),
+        name: name.into(),
+        kind: matrix.tensor().kind,
+        rows: matrix.rows(),
+        columns: matrix.columns(),
+        axes: matrix.axes(),
+    })
+}
+
+struct ProbeOwner {
+    _window: Arc<TensorWindow>,
+    initial: [Vec<u8>; 6],
+    _ram: WindowWorkspace,
+}
+fn lease_view(
+    service: &Arc<ResidentService>,
+    lease: &ResidentLease,
+    payload: usize,
+) -> Result<Handle> {
+    let full = payload
+        .checked_add(2 * GUARD_BYTES)
+        .ok_or_else(|| malformed("guarded lease byte overflow"))?;
+    let handle = lease.handle();
+    if !Arc::ptr_eq(lease.service(), service)
+        || lease.source_identity() != service.source_identity()
+        || lease.service_generation() != service.generation()
+        || lease.bytes() != full
+        || !lease.is_initialized()
+        || handle.service != service.client().service_id()
+        || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
+        || handle.size_in_used() < full as u64
+    {
+        return Err(malformed("guarded original lease/service/whole range"));
+    }
+    let trim = handle.size_in_used() - GUARD_BYTES as u64 - payload as u64;
+    let view = handle.offset_start(GUARD_BYTES as u64).offset_end(trim);
+    if view.size_in_used() != payload as u64 {
+        return Err(malformed("actual guarded subview byte range"));
+    }
+    Ok(view)
+}
+
+/// Direct qualifier of the SAME compiled kernel with guarded original leases.
+/// This is not a second operator implementation or a production binding.
+async fn probe(
+    service: &Arc<ResidentService>,
+    source: &VerifiedTensorSource,
+    kind: AdversaryKind,
+    columns: usize,
+    batch: usize,
+) -> Result<usize> {
+    let ram = source.reserve_workspace(RAM)?; // BEFORE recipes, CPU arrays or copies.
+    let (full, inputs) = kind.fixture(columns)?;
+    let desc = descriptor(source, &kind.source_name(columns))?;
+    if desc.rows != SYNTHETIC_ROWS || desc.columns != columns || desc.kind != QuantKind::Bf16 {
+        return Err(malformed("synthetic full original descriptor differs"));
+    }
+    let matrix = source.matrix(&desc.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, FIRST..LAST)?);
+    if window.as_slice() != &full[FIRST * columns * 2..LAST * columns * 2] {
+        return Err(malformed(
+            "sealed synthetic words differ from declared integer recipe",
+        ));
+    }
+    let bound = Bf16Window::bind(
+        source.clone(),
+        window.clone(),
+        desc.clone(),
+        FIRST..LAST,
+        PROFILE,
+    )?;
+    bound.identity.require_window(&window)?;
+    let raw = inputs[..batch].concat();
+    let bound_input = Bf16Inputs::bind(source.clone(), PROFILE, OUTPUT_ROWS, columns, batch, &raw)?;
+    if !source.same_issuer(&bound.source) || !source.same_issuer(&bound_input.source) {
+        return Err(malformed("synthetic source issuer mismatch"));
+    }
+    let mut expected = Vec::new();
+    for input in inputs.iter().take(batch) {
+        let x = input
+            .chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>();
+        let cpu = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
+            &desc,
+            FIRST..LAST,
+            &window,
+            &x,
+            None,
+            None,
+        );
+        if kind == AdversaryKind::Fp32Overflow {
+            if !matches!(cpu,Err(Error::Malformed(ref message)) if message=="nonfinite original window projection output")
+            {
+                return Err(malformed(
+                    "CPU oracle did not refuse genuine finite-input FP32 overflow",
+                ));
+            }
+            // Read the two independent finite marker rows through ORIGINAL CPU math.
+            let marker = source.read_rows(&matrix, 2..LAST)?;
+            let finite = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
+                &desc,
+                2..LAST,
+                &marker,
+                &x,
+                None,
+                None,
+            )?;
+            let mut row = vec![f32::INFINITY];
+            row.extend_from_slice(finite.accumulations());
+            expected.push(row);
+        } else {
+            expected.push(cpu?.accumulations().to_vec());
+        }
+    }
+    if expected.len() != batch || expected.iter().any(|row| row.len() != OUTPUT_ROWS) {
+        return Err(malformed("complete original CPU oracle output coverage"));
+    }
+    if let Some((raw, rounded)) = kind.witness() {
+        if expected[0][0].to_bits() != raw || rne(raw) != rounded {
+            return Err(malformed(
+                "original CPU64 analytic synthetic witness failed before CUDA",
+            ));
+        }
+    }
+    let shape = bound_input.shape;
+    let hardware = &service.client().properties().hardware;
+    service.ensure_usable()?;
+    service.pinned_cuda_queue()?;
+    if service.source_identity() != source.source_identity()
+        || !crate::activation_quant_fused::is_cuda_service(service.client())
+        || hardware.max_units_per_cube < 64
+        || hardware.max_cube_dim.0 < 64
+        || hardware.max_shared_memory_size < 512
+        || hardware.max_bindings < 6
+        || hardware.max_cube_count.0 < shape.outputs as u32
+    {
+        return Err(malformed(
+            "guarded native original source/workgroup capability",
+        ));
+    }
+    let payloads = [
+        shape.weight_bytes,
+        shape.input_bytes,
+        4,
+        shape.outputs * 4,
+        shape.outputs * 4,
+        shape.outputs * 4,
+    ];
+    let poison = vec![0x7fc0_1234u32.to_le_bytes(); shape.outputs].concat();
+    let initial = [
+        guarded(0, window.as_slice()),
+        guarded(1, &raw),
+        guarded(2, &0u32.to_le_bytes()),
+        guarded(3, &poison),
+        guarded(4, &poison),
+        guarded(5, &poison),
+    ];
+    let owner = Arc::new(ProbeOwner {
+        _window: window,
+        initial,
+        _ram: ram,
+    });
+    let retained = Arc::downgrade(&owner);
+    let total = owner.initial.iter().map(Vec::len).sum::<usize>();
+    if service.remaining_capacity()? < total {
+        return Err(Error::Limit("guarded probe capacity".into()));
+    }
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: total as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    if service.remaining_capacity()? < total {
+        return Err(Error::Limit("guarded probe queued capacity".into()));
+    }
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(0)?;
+    init.retain_completion_owner(owner.clone())?;
+    let roles = [
+        BufferRole::ImmutableWeights,
+        BufferRole::Activation,
+        BufferRole::ImmutableWeights,
+        BufferRole::Output,
+        BufferRole::Output,
+        BufferRole::Scratch,
+    ];
+    let mut leases = Vec::new();
+    for (raw, role) in owner.initial.iter().zip(roles) {
+        leases.push(init.upload_bytes(raw, role)?);
+    }
+    let initialized = init.complete_sync().await?;
+    slot.finish()?;
+    drop(initialized);
+    let mut views = Vec::new();
+    let mut ids = std::collections::BTreeSet::new();
+    for (lease, payload) in leases.iter().zip(payloads) {
+        if !ids.insert(lease.id()) {
+            return Err(malformed("guarded lease alias"));
+        }
+        views.push(lease_view(service, lease, payload)?);
+    }
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: total as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    service.ensure_usable()?;
+    let owners_before =
+        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
+    slot.begin_submission()?;
+    let accesses = leases
+        .iter()
+        .enumerate()
+        .map(|(i, l)| {
+            (
+                l.clone(),
+                if i < 3 {
+                    DeviceAccess::Read
+                } else {
+                    DeviceAccess::Write
+                },
+            )
+        })
+        .collect::<Vec<_>>();
+    let mut flight = service.submit(0, &accesses)?;
+    flight.retain_completion_owner(owner.clone())?;
+    let arg = |index: usize, n: usize| -> BufferArg {
+        // SAFETY: six original whole leases are retained by the armed flight;
+        // exact aligned interior views exclude both checked physical canaries.
+        unsafe { BufferArg::from_raw_parts(views[index].clone(), n) }
+    };
+    // SAFETY: exact native shape, one64-lane cube/output, original kernel/barriers.
+    unsafe {
+        project_cpu_bf16::launch_unchecked(
+            service.client(),
+            CubeCount::Static(shape.outputs as u32, 1, 1),
+            CubeDim::new_1d(64),
+            arg(0, shape.weight_bytes / 4),
+            arg(1, shape.input_bytes / 4),
+            arg(2, 1),
+            arg(3, shape.outputs),
+            arg(4, shape.outputs),
+            arg(5, shape.outputs),
+            shape.rows,
+            shape.columns,
+        );
+    }
+    let outputs = leases[3..].to_vec();
+    let reads = leases
+        .iter()
+        .map(|lease| lease.readback(0, lease.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let validated = AtomicBool::new(false);
+    let validate = |actual: &[Bytes]| -> Result<()> {
+        if actual.len() != 6 {
+            return Err(malformed("guarded full physical readback coverage"));
+        }
+        for (index, (actual, payload)) in actual.iter().zip(payloads).enumerate() {
+            check_guards(index, actual, payload)?;
+            if index < 3 && actual.as_ref() != owner.initial[index].as_slice() {
+                return Err(malformed(
+                    "actual readonly BF16 source/input/zero bits changed",
+                ));
+            }
+        }
+        for (request, expected_row) in expected.iter().enumerate() {
+            for (row, expected_value) in expected_row.iter().enumerate() {
+                let at = GUARD_BYTES + (request * OUTPUT_ROWS + row) * 4;
+                let accumulator = word(&actual[3][at..at + 4]);
+                let rounded = word(&actual[4][at..at + 4]);
+                let status = word(&actual[5][at..at + 4]);
+                let cpu = expected_value.to_bits();
+                let rounded_cpu = rne(cpu);
+                let expected_status = if row == 0 && kind == AdversaryKind::Fp32Overflow {
+                    24
+                } else if row == 0 && kind == AdversaryKind::Bf16Overflow {
+                    16
+                } else {
+                    0
+                };
+                if accumulator != cpu || rounded != rounded_cpu || status != expected_status {
+                    return Err(Error::State(format!(
+                        "synthetic BF16 {} C{columns} B{batch} request{request} row{row}: actual={accumulator:08x}/{rounded:08x}/{status:08x} expected={cpu:08x}/{rounded_cpu:08x}/{expected_status:08x}",
+                        kind.name()
+                    )));
+                }
+            }
+        }
+        validated.store(true, Ordering::Release);
+        if kind.fault() {
+            Err(Error::State(EXPECTED_REFUSAL.into()))
+        } else {
+            Ok(())
+        }
+    };
+    let attempt = flight
+        .complete_with_readback(&outputs, &reads, validate)
+        .await;
+    if kind.fault() {
+        if !matches!(attempt,Err(Error::State(ref message)) if message==EXPECTED_REFUSAL)
+            || !validated.load(Ordering::Acquire)
+            || service.ensure_usable().is_ok()
+            || !ResourceArbiter::process_shared()
+                .snapshot(&service.physical_key())?
+                .fenced
+            || service.pending_events()? == 0
+        {
+            return Err(malformed(
+                "actual nonfinite status did not refuse Ready into original quarantine",
+            ));
+        }
+        drop(slot);
+        drop(reads);
+        drop(outputs);
+        drop(views);
+        drop(accesses);
+        drop(leases);
+        drop(owner);
+        drop(bound);
+        drop(bound_input);
+        if retained.upgrade().is_none()
+            || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
+                <= owners_before
+            || ResourceArbiter::process_shared()
+                .snapshot(&service.physical_key())?
+                .reserved
+                < PROVIDER as u64
+        {
+            return Err(malformed(
+                "nonfinite status lost original source/completion owner",
+            ));
+        }
+        println!(
+            "CPU_BF16_SYNTHETIC_STATUS_PASS case={} columns={columns} B={batch} actual_kernel_dispatches=1 canaries=exact readonly_bits=exact CPU_refusal_or_round_overflow=exact Ready=false original_source_and_quarantine=true final_zero=false",
+            kind.name()
+        );
+        return Ok(shape.outputs);
+    }
+    let (ready, readback) = attempt?;
+    slot.finish()?;
+    if !validated.load(Ordering::Acquire)
+        || ready.outputs().len() != 3
+        || ready.event_ids().len() != 1
+    {
+        return Err(malformed("guarded native physical Ready receipt"));
+    }
+    drop(readback);
+    drop(ready);
+    drop(reads);
+    drop(outputs);
+    drop(views);
+    drop(accesses);
+    drop(leases);
+    drop(owner);
+    drop(bound);
+    drop(bound_input);
+    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+        return Err(malformed(
+            "guarded native output/event leases did not retire",
+        ));
+    }
+    println!(
+        "CPU_BF16_SYNTHETIC_OPERATOR_PASS case={} columns={columns} B={batch} source_rows=5 range=1..4 actual_kernel_dispatches=1 accumulation_and_RNE=CPU64_exact status=0 canaries=exact readonly_bits=exact Ready=true trained_model=false",
+        kind.name()
+    );
+    Ok(shape.outputs)
+}
+
+/// Full normal run or ONE failure probe in a fresh process. Never promote HF.
+pub async fn qualify(
+    device: Device,
+    source: VerifiedTensorSource,
+    device_cap: u64,
+    fault: Option<(AdversaryKind, usize)>,
+) -> Result<()> {
+    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "synthetic physical BF16 proof requires release CUDA".into(),
+        ));
+    }
+    if fault.is_some_and(|(kind, batch)| !kind.fault() || ![1, 2, 4, 8].contains(&batch)) {
+        return Err(malformed(
+            "one explicit fresh-process nonfinite failure mode",
+        ));
+    }
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("synthetic-cpu64-bf16", "generic-operator-only")?,
+        PROVIDER,
+        device_cap,
+    )?;
+    if ResourceArbiter::process_shared()
+        .snapshot(&service.physical_key())?
+        .reserved
+        != PROVIDER as u64
+    {
+        return Err(malformed(
+            "synthetic proof requires otherwise idle original physical budget",
+        ));
+    }
+    let key = service.physical_key();
+    if let Some((kind, batch)) = fault {
+        probe(&service, &source, kind, 128, batch).await?;
+        return Ok(()); // Deliberately fenced; no further device work/final-zero.
+    }
+    let mut launches = 0usize;
+    let mut words = 0usize;
+    let mut nonfinite_sources = 0usize;
+    let mut nonfinite_inputs = 0usize;
+    for columns in [65, 8193] {
+        let desc = descriptor(&source, &bad_geometry_name(columns)?)?;
+        if desc.rows != SYNTHETIC_ROWS || desc.columns != columns {
+            return Err(malformed("actual cold geometry source shape"));
+        }
+        let matrix = source.matrix(&desc.name, MatrixAxes::OutputInput)?;
+        let window = Arc::new(source.read_rows(&matrix, FIRST..LAST)?);
+        if Bf16Window::bind(source.clone(), window, desc, FIRST..LAST, PROFILE).is_ok()
+            || service.resident_bytes()? != 0
+            || service.uploaded_value_bytes() != 0
+        {
+            return Err(malformed(
+                "actual authenticated unsupported geometry crossed cold boundary",
+            ));
+        }
+    }
+    for columns in ADVERSARY_WIDTHS {
+        let _ram = source.reserve_workspace(RAM)?;
+        let uploads = service.uploaded_value_bytes();
+        for bad in [0x7fc1u16, 0x7f80] {
+            let mut raw = vec![0u8; columns * 2];
+            raw[0..2].copy_from_slice(&bad.to_le_bytes());
+            if Bf16Inputs::bind(source.clone(), PROFILE, OUTPUT_ROWS, columns, 1, &raw).is_ok()
+                || service.resident_bytes()? != 0
+                || service.uploaded_value_bytes() != uploads
+            {
+                return Err(malformed("nonfinite explicit input crossed cold boundary"));
+            }
+            nonfinite_inputs += 1;
+        }
+        if Bf16Inputs::bind(
+            source.clone(),
+            PROFILE,
+            OUTPUT_ROWS,
+            columns,
+            3,
+            &vec![0; columns * 6],
+        )
+        .is_ok()
+        {
+            return Err(malformed("unpadded unsupported B3 crossed cold boundary"));
+        }
+        for kind in AdversaryKind::all(columns)? {
+            if kind.fault() {
+                continue;
+            }
+            if kind.nonfinite_source() {
+                let desc = descriptor(&source, &kind.source_name(columns))?;
+                let matrix = source.matrix(&desc.name, MatrixAxes::OutputInput)?;
+                let window = Arc::new(source.read_rows(&matrix, FIRST..LAST)?);
+                let uploads = service.uploaded_value_bytes();
+                if Bf16Window::bind(source.clone(), window, desc, FIRST..LAST, PROFILE).is_ok()
+                    || service.resident_bytes()? != 0
+                    || service.uploaded_value_bytes() != uploads
+                {
+                    return Err(malformed(
+                        "authenticated nonfinite synthetic source crossed cold boundary",
+                    ));
+                }
+                nonfinite_sources += 1;
+                continue;
+            }
+            for batch in [1, 2, 4, 8] {
+                words += probe(&service, &source, kind, columns, batch).await?;
+                launches += 1;
+            }
+        }
+    }
+    if launches != 328 || words != 3690 || nonfinite_sources != 12 || nonfinite_inputs != 12 {
+        return Err(malformed(
+            "synthetic complete case/native-band/cold-refusal coverage",
+        ));
+    }
+    drop(service);
+    let final_row = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_row.reserved != 0 || final_row.reservations != 0 || final_row.fenced {
+        return Err(malformed(
+            "synthetic normal original physical budget final zero",
+        ));
+    }
+    println!(
+        "CPU_BF16_SYNTHETIC_NORMAL_PASS actual_kernel_dispatches={launches} accumulation_words={words} rounded_words={words} source_nonfinite_cold_refusals=12 input_nonfinite_cold_refusals=12 actual_source_geometry_refusals=2 B3_refusals=6 original_GPU_reservations=0 trained_model=false HF_or_vLLM_profile_promotion=false"
+    );
+    Ok(())
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn synthetic_source_range_and_member_markers_remain_distinct() {
+        for columns in ADVERSARY_WIDTHS {
+            for kind in AdversaryKind::all(columns).unwrap() {
+                let (full, inputs) = kind.fixture(columns).unwrap();
+                assert_eq!(full.len(), SYNTHETIC_ROWS * columns * 2);
+                assert_eq!(inputs.len(), 8);
+                assert!(inputs.iter().all(|raw| raw.len() == columns * 2));
+                assert_ne!(
+                    &full[..columns * 2],
+                    &full[(SYNTHETIC_ROWS - 1) * columns * 2..]
+                );
+                for pair in inputs.windows(2) {
+                    assert_ne!(pair[0], pair[1]);
+                }
+                let source_word = u16::from_le_bytes([full[columns * 2], full[columns * 2 + 1]]);
+                if kind.nonfinite_source() {
+                    assert_eq!(source_word & 0x7f80, 0x7f80);
+                } else {
+                    assert!(finite_bf16(&full));
+                }
+                assert!(inputs.iter().all(|raw| finite_bf16(raw)));
+            }
+        }
+    }
+    #[test]
+    fn every_physical_guard_word_mutation_is_detected() {
+        for binding in 0..6 {
+            let bytes = guarded(binding, &[0x52u8; 12]);
+            check_guards(binding, &bytes, 12).unwrap();
+            for at in (0..GUARD_BYTES).chain(GUARD_BYTES + 12..bytes.len()) {
+                let mut changed = bytes.clone();
+                changed[at] ^= 1;
+                assert!(check_guards(binding, &changed, 12).is_err());
+            }
+            assert!(check_guards(binding, &bytes, 16).is_err());
+            // Payload is checked against CPU/status separately, not by canaries.
+            let mut changed = bytes.clone();
+            changed[GUARD_BYTES] ^= 1;
+            check_guards(binding, &changed, 12).unwrap();
+        }
+    }
+    #[test]
+    fn recipes_do_not_add_scalar_tails_or_fake_source_widths() {
+        for columns in [0, 63, 65, 8193] {
+            assert!(AdversaryKind::all(columns).is_err());
+        }
+        assert!(AdversaryKind::FmaUnderflow.fixture(64).is_err());
+        assert!(AdversaryKind::FmaOverflowCancellation.fixture(64).is_err());
+        assert_eq!(
+            ADVERSARY_WIDTHS
+                .iter()
+                .map(|&c| AdversaryKind::all(c)
+                    .unwrap()
+                    .into_iter()
+                    .filter(|kind| !kind.fault() && !kind.nonfinite_source())
+                    .count()
+                    * 4)
+                .sum::<usize>(),
+            328
+        );
+    }
+}

--- V1/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
+++ ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
@@ -571,7 +571,7 @@
         }
         for (index, (actual, payload)) in actual.iter().zip(payloads).enumerate() {
             check_guards(index, actual, payload)?;
-            if index < 3 && actual.as_ref() != owner.initial[index].as_slice() {
+            if index < 3 && &actual[..] != owner.initial[index].as_slice() {
                 return Err(malformed(
                     "actual readonly BF16 source/input/zero bits changed",
                 ));

--- ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
+++ ROOT_V3/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
@@ -419,12 +419,12 @@
     if expected.len() != batch || expected.iter().any(|row| row.len() != OUTPUT_ROWS) {
         return Err(malformed("complete original CPU oracle output coverage"));
     }
-    if let Some((raw, rounded)) = kind.witness() {
-        if expected[0][0].to_bits() != raw || rne(raw) != rounded {
-            return Err(malformed(
-                "original CPU64 analytic synthetic witness failed before CUDA",
-            ));
-        }
+    if let Some((raw, rounded)) = kind.witness()
+        && (expected[0][0].to_bits() != raw || rne(raw) != rounded)
+    {
+        return Err(malformed(
+            "original CPU64 analytic synthetic witness failed before CUDA",
+        ));
     }
     let shape = bound_input.shape;
     let hardware = &service.client().properties().hardware;

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v3-20261020/ROOT_V3.review.patch

Original UTF-8 bytes: 958; SHA256 `fedfa921eedb923e6bdd8eea7b937cc12d20837c4e4618c26640019dd6bb114c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
+++ ROOT_V3/crates/zerogravity-gpu/src/cpu_bf16_projection/adversary.rs
@@ -419,12 +419,12 @@
     if expected.len() != batch || expected.iter().any(|row| row.len() != OUTPUT_ROWS) {
         return Err(malformed("complete original CPU oracle output coverage"));
     }
-    if let Some((raw, rounded)) = kind.witness() {
-        if expected[0][0].to_bits() != raw || rne(raw) != rounded {
-            return Err(malformed(
-                "original CPU64 analytic synthetic witness failed before CUDA",
-            ));
-        }
+    if let Some((raw, rounded)) = kind.witness()
+        && (expected[0][0].to_bits() != raw || rne(raw) != rounded)
+    {
+        return Err(malformed(
+            "original CPU64 analytic synthetic witness failed before CUDA",
+        ));
     }
     let shape = bound_input.shape;
     let hardware = &service.client().properties().hardware;

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v3-20261020/WIRING.patch

Original UTF-8 bytes: 1374; SHA256 `d59ebed5c94361657cb1665d9b1790e0d55b721c80a72eb37d4ff298afa7f42b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
@@ -31,6 +31,7 @@
 use zerogravity_core::quant::QuantKind;
 use zerogravity_core::{Error, Result};
 
+pub(crate) mod adversary;
 pub(crate) mod qualification;
 mod resident;
 pub(crate) use resident::{PreparedProjection, ProjectionReadback};
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -988,3 +988,21 @@
     )
     .await
 }
+
+/// Explicitly SYNTHETIC generic operator fixtures; no trained source/model claim.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub use crate::cpu_bf16_projection::adversary::{
+    ADVERSARY_WIDTHS as CPU_BF16_ADVERSARY_WIDTHS, AdversaryKind as CpuBf16AdversaryKind,
+    SYNTHETIC_ROWS as CPU_BF16_SYNTHETIC_ROWS, bad_geometry_name as cpu_bf16_bad_geometry_name,
+};
+
+/// Root-only same-kernel guarded proof or ONE deliberate nonfinite status case.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_adversaries(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    device_cap: u64,
+    fault: Option<(CpuBf16AdversaryKind, usize)>,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::adversary::qualify(device, source, device_cap, fault).await
+}

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v2-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 49204; SHA256 `14d602a979be557c96be9b055f63cad0c9266dd6cd2530d29a71c6ea3e7fc325`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/Cargo.toml
+++ b/crates/zerogravity-gpu/Cargo.toml
@@ -13,7 +13,7 @@
 cuda = ["cubecl/cuda", "cubecl/cuda-cpp"]
 hip = ["cubecl/hip", "cubecl/hip-cpp"]
 all = ["vulkan", "cuda", "hip"]
-qualification = ["dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
+qualification = ["zerogravity-artifacts/cache", "zerogravity-cpu/cache", "dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
 hardware-abort-probe = ["qualification"]
 portable-checkpoint = ["dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/diagnostic-cpu-checkpoint"]
 diagnostic-gdn-trace = ["qualification", "zerogravity-model-qwen35/diagnostic-gdn-trace"]
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -24,6 +24,8 @@
 mod batch_quant_qualification;
 #[cfg(any(feature = "qualification", test))]
 mod batch_recurrent;
+#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
+mod cpu_bf16_projection;
 mod device_activation;
 mod elementwise;
 mod embedding;
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -949,3 +949,42 @@
         "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nUse rope-angles-cuda MODEL for physical angle differential qualification.\nUse rope-angles-model-cuda MODEL [PROVIDER_MiB DEVICE_MiB ROWS] for existing CPU/B1/native model gates.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
     );
 }
+
+/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_window(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_window(
+        device,
+        source,
+        descriptor,
+        rows,
+        inputs,
+        public_outputs,
+        device_cap,
+    )
+    .await
+}
+
+/// Separate fresh process only. Actual unknown completion must stay fenced.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_abandon(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_abandon(
+        device, source, descriptor, rows, input, device_cap,
+    )
+    .await
+}
--- a/crates/zerogravity-gpu/src/resident_leases.rs
+++ b/crates/zerogravity-gpu/src/resident_leases.rs
@@ -814,6 +814,15 @@
     armed: bool,
 }
 impl ResidentInitialization<'_> {
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub(crate) fn retain_completion_owner(
+        &mut self,
+        owner: crate::physical_quarantine::CompletionOwner,
+    ) -> Result<()> {
+        // Attach first: refusal must retain owners of previously queued work.
+        self.completion_owners.push(owner);
+        self.service.ensure_usable()
+    }
     pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
         self.service.ensure_usable()?;
         let allocation_generation = issue(&self.service.next_allocation)?;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -0,0 +1,391 @@
+//! ROOT-only dual CPU-reference / authentic public-output physical proof.
+//! External caller must authenticate the original source and actual archived
+//! teacher inputs/outputs. Synthetic words alone cannot qualify trained math.
+use super::*;
+use crate::resident_leases::ResidentService;
+use cubecl::Device;
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::source_weights::VerifiedTensorSource;
+use zerogravity_core::device_events::BufferRole;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+
+const PROVIDER_CAP: usize = 64 * 1024 * 1024;
+
+fn bytes_word(bytes: &[u8]) -> u32 {
+    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
+}
+fn check_exact(
+    result: &ProjectionReadback,
+    expected: &[Vec<f32>],
+    torch: &[Vec<u8>],
+    shape: Shape,
+) -> Result<usize> {
+    if result.batch != shape.batch
+        || result.kernel_dispatches != 1
+        || result.ready_outputs() != 3
+        || result.completion_events() != 1
+        || result.accumulation().len() != shape.outputs * 4
+        || result.rounded().len() != shape.outputs * 4
+    {
+        return Err(malformed("physical single-dispatch/Ready/output coverage"));
+    }
+    for request in 0..shape.batch {
+        let teacher = request % 2;
+        for row in 0..shape.rows {
+            let at = request * shape.rows + row;
+            let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
+            let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
+            let expected_bits = expected[teacher][row].to_bits();
+            let full_row = result.identity.rows.start + row;
+            let reference = u32::from(u16::from_le_bytes([
+                torch[teacher][2 * full_row],
+                torch[teacher][2 * full_row + 1],
+            ])) << 16;
+            if actual != expected_bits || rounded != reference {
+                return Err(Error::State(format!(
+                    "CPU-profile BF16 CUDA B{} request{request} row{full_row}: accumulation {actual:08x}!={expected_bits:08x}, final {rounded:08x}!={reference:08x}",
+                    shape.batch
+                )));
+            }
+        }
+    }
+    Ok(shape.outputs)
+}
+
+/// Test-only public entry receives actual source and ROOT-authenticated raw
+/// records; no model factory/provider fallback or per-token foreign runtime.
+pub async fn qualify_window(
+    device: Device,
+    source: VerifiedTensorSource,
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> Result<()> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "physical BF16 projection requires release".into(),
+        ));
+    }
+    if !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "physical BF16 projection requires CUDA".into(),
+        ));
+    }
+    Shape::new(rows.len(), descriptor.columns, 1)?;
+    if descriptor.rows == 0 || descriptor.rows > 1_048_576 {
+        return Err(malformed("bounded original full row count"));
+    }
+    let _host = source.reserve_workspace(2 * 1024 * 1024)?;
+    if descriptor.source != source.source_identity()
+        || inputs
+            .iter()
+            .any(|raw| raw.len() != descriptor.columns * 2 || !finite_bf16(raw))
+        || public_outputs
+            .iter()
+            .any(|raw| raw.len() != descriptor.rows * 2 || !finite_bf16(raw))
+    {
+        return Err(malformed(
+            "complete actual original two-teacher input/public output geometry",
+        ));
+    }
+    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
+    Bf16Window::bind(
+        source.clone(),
+        window.clone(),
+        descriptor.clone(),
+        rows.clone(),
+        PROFILE,
+    )?;
+    // Real sealed original window, deliberately wrong full-descriptor tags.
+    // None may become a partial/fake source or reach CUDA initialization.
+    for changed in 0..4 {
+        let mut wrong = descriptor.clone();
+        match changed {
+            0 => {
+                wrong.source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation()?
+            }
+            1 => wrong.axes = MatrixAxes::InputOutput,
+            2 => wrong.rows = wrong.rows.saturating_add(1),
+            _ => wrong.name.push_str(".foreign"),
+        }
+        if Bf16Window::bind(source.clone(), window.clone(), wrong, rows.clone(), PROFILE).is_ok() {
+            return Err(malformed("wrong original full descriptor was accepted"));
+        }
+    }
+    let mut expected = Vec::new();
+    for raw in &inputs {
+        let values = raw
+            .chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>();
+        let cpu = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
+            &descriptor,
+            rows.clone(),
+            &window,
+            &values,
+            None,
+            None,
+        )?;
+        if cpu.identity().descriptor() != &descriptor || cpu.identity().row_range() != rows {
+            return Err(malformed("original CPU oracle full identity/range"));
+        }
+        expected.push(cpu.accumulations().to_vec());
+    }
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("original-cpu-bf16-reference", &descriptor.name)?,
+        PROVIDER_CAP,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != PROVIDER_CAP as u64 {
+        return Err(Error::State(
+            "BF16 diagnostic requires otherwise idle physical budget".into(),
+        ));
+    }
+    // Real original physical allocation, leaving four bytes less than required.
+    // The new binding must refuse BEFORE H2D, allocation or model dispatch.
+    let shape = Shape::new(rows.len(), descriptor.columns, 8)?;
+    let need = shape.owned_bytes()?;
+    let filler_bytes = PROVIDER_CAP
+        .checked_sub(need)
+        .and_then(|n| n.checked_add(4))
+        .ok_or_else(|| malformed("pressure fixture cap"))?;
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: filler_bytes as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(99)?;
+    let filler = init.allocate_empty(filler_bytes, BufferRole::Scratch)?;
+    let initialized = init.complete_sync().await?;
+    slot.finish()?;
+    drop(initialized);
+    let raw = inputs[0].repeat(8);
+    let uploads = service.uploaded_value_bytes();
+    let pressure = PreparedProjection::prepare(
+        service.clone(),
+        Bf16Window::bind(
+            source.clone(),
+            window.clone(),
+            descriptor.clone(),
+            rows.clone(),
+            PROFILE,
+        )?,
+        Bf16Inputs::bind(
+            source.clone(),
+            PROFILE,
+            rows.len(),
+            descriptor.columns,
+            8,
+            &raw,
+        )?,
+        99,
+        None,
+    )
+    .await;
+    if !matches!(pressure, Err(Error::Limit(ref message)) if message == "private BF16 operator capacity before launch")
+        || service.resident_bytes()? != filler_bytes
+        || service.uploaded_value_bytes() != uploads
+        || service.pending_events()? != 0
+    {
+        return Err(Error::State(
+            "BF16 real capacity refusal crossed prelaunch boundary".into(),
+        ));
+    }
+    drop(filler);
+    if service.resident_bytes()? != 0 {
+        return Err(Error::State(
+            "BF16 pressure allocation did not retire".into(),
+        ));
+    }
+    let mut compared = 0usize;
+    let mut actual_dispatches = 0usize;
+    for (case, batch) in [1usize, 1usize, 2usize, 4usize, 8usize]
+        .into_iter()
+        .enumerate()
+    {
+        let shape = Shape::new(rows.len(), descriptor.columns, batch)?;
+        let selected = if case == 1 { 1usize } else { 0usize };
+        let raw = (0..batch)
+            .flat_map(|request| inputs[(request + selected) % 2].iter().copied())
+            .collect::<Vec<_>>();
+        let bind_input = || {
+            Bf16Inputs::bind(
+                source.clone(),
+                PROFILE,
+                rows.len(),
+                descriptor.columns,
+                batch,
+                &raw,
+            )
+        };
+        let source_window = || {
+            Bf16Window::bind(
+                source.clone(),
+                window.clone(),
+                descriptor.clone(),
+                rows.clone(),
+                PROFILE,
+            )
+        };
+        // Queued/prelaunch cancellation cannot create a lease, H2D or kernel.
+        let uploads = service.uploaded_value_bytes();
+        if PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            Some(&AtomicBool::new(true)),
+        )
+        .await
+        .is_ok()
+            || service.resident_bytes()? != 0
+            || service.uploaded_value_bytes() != uploads
+        {
+            return Err(Error::State(
+                "BF16 pre-cancel performed physical work".into(),
+            ));
+        }
+        // Cold initialized leases may also cancel before their first kernel.
+        let prepared = PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            None,
+        )
+        .await?;
+        let uploads = service.uploaded_value_bytes();
+        if prepared
+            .execute(case as u64, Some(&AtomicBool::new(true)))
+            .await
+            .is_ok()
+            || service.resident_bytes()? != 0
+            || service.pending_events()? != 0
+            || service.uploaded_value_bytes() != uploads
+        {
+            return Err(Error::State(
+                "BF16 initialized prelaunch cancellation leaked or submitted work".into(),
+            ));
+        }
+        let prepared = PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            None,
+        )
+        .await?;
+        let result = prepared.execute(case as u64, None).await?;
+        result.identity.require_window(&window)?;
+        if result.identity.descriptor() != &descriptor || result.identity.rows() != rows {
+            return Err(malformed("actual GPU full descriptor/range"));
+        }
+        // Second independent B1 uses actual teacher353, never padded/relabelled.
+        if selected == 0 {
+            compared += check_exact(&result, &expected, &public_outputs, shape)?;
+        } else {
+            let expected_swapped = [expected[1].clone(), expected[0].clone()];
+            let public_swapped = [public_outputs[1].clone(), public_outputs[0].clone()];
+            compared += check_exact(&result, &expected_swapped, &public_swapped, shape)?;
+        }
+        actual_dispatches += result.kernel_dispatches;
+        drop(result);
+        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+            return Err(Error::State(
+                "BF16 physical outputs/events failed retirement".into(),
+            ));
+        }
+        println!(
+            "CPU_PROFILE_BF16_CUDA_OPERATOR_PASS name={} full_rows={} range={}..{} columns={} B={batch} accumulation_bits=exact archived_public_BF16_bits=exact actual_native_dispatches=1 physical_pressure_and_prelaunch_cancel=true state_or_HF_profile_promotion=false",
+            descriptor.name, descriptor.rows, rows.start, rows.end, descriptor.columns
+        );
+    }
+    if compared != 16 * rows.len() || actual_dispatches != 5 {
+        return Err(Error::State(
+            "BF16 physical all-row dual reference coverage incomplete".into(),
+        ));
+    }
+    drop(service);
+    let final_row = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_row.reserved != 0 || final_row.reservations != 0 || final_row.fenced {
+        return Err(Error::State(
+            "BF16 original GPU budget did not finish at zero".into(),
+        ));
+    }
+    Ok(())
+}
+
+/// Separate fresh process only: ACTUAL queued kernel, no trusted completion.
+/// Success here means the original uncertainty quarantine held, never Ready.
+pub async fn qualify_abandon(
+    device: Device,
+    source: VerifiedTensorSource,
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> Result<()> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "physical BF16 abort requires release".into(),
+        ));
+    }
+    if !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "physical BF16 abort requires CUDA".into(),
+        ));
+    }
+    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
+    let witness = Arc::downgrade(&window);
+    let inputs = Bf16Inputs::bind(
+        source.clone(),
+        PROFILE,
+        rows.len(),
+        descriptor.columns,
+        1,
+        &input,
+    )?;
+    let original = Bf16Window::bind(source.clone(), window.clone(), descriptor, rows, PROFILE)?;
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("original-cpu-bf16-reference", "physical-abandon")?,
+        PROVIDER_CAP,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
+    prepared.abandon_after_launch(0).await?;
+    drop(window);
+    drop(source);
+    let actual = ResourceArbiter::process_shared().snapshot(&key)?;
+    if !actual.fenced
+        || actual.reserved < PROVIDER_CAP as u64
+        || service.ensure_usable().is_ok()
+        || witness.upgrade().is_none()
+        || service.pending_events()? == 0
+    {
+        return Err(Error::State(
+            "BF16 actual unknown work lost original leases/source/fence".into(),
+        ));
+    }
+    println!(
+        "CPU_PROFILE_BF16_CUDA_ABANDON_PASS actual_kernel_submitted=true trusted_completion=false Ready=false original_source_and_leases_quarantined=true final_zero_not_claimed=true"
+    );
+    Ok(())
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/resident.rs
@@ -0,0 +1,337 @@
+//! Private operator ownership through the one original resident GPU service.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentReady, ResidentService, ResidentSubmission};
+use cubecl::bytes::Bytes;
+use std::sync::atomic::{AtomicBool, Ordering};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+struct SourceOwner {
+    _window: Arc<TensorWindow>,
+    _input: Arc<[u8]>,
+    _input_admission: Arc<WindowWorkspace>,
+    _weight_transport: WindowWorkspace,
+    _input_transport: WindowWorkspace,
+    _readback_admission: WindowWorkspace,
+}
+fn cancel(flag: Option<&AtomicBool>) -> Result<()> {
+    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        Err(Error::State("private BF16 projection cancelled".into()))
+    } else {
+        Ok(())
+    }
+}
+
+pub(crate) struct PreparedProjection {
+    service: Arc<ResidentService>,
+    shape: Shape,
+    identity: Identity,
+    owner: Arc<SourceOwner>,
+    weights: ResidentLease,
+    input: ResidentLease,
+    zero: ResidentLease,
+    accumulation: ResidentLease,
+    rounded: ResidentLease,
+    status: ResidentLease,
+}
+impl PreparedProjection {
+    /// A PURE operator correlation, never an invented SessionAuthority/root.
+    /// The caller supplies an existing pinned original service, not a device
+    /// from which a second context/queue might be created.
+    pub(crate) async fn prepare(
+        service: Arc<ResidentService>,
+        window: Bf16Window,
+        input: Bf16Inputs,
+        correlation: u64,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        cancel(cancelled)?;
+        let shape = input.shape;
+        let bytes = shape.owned_bytes()?;
+        let client = service.client();
+        let hardware = &client.properties().hardware;
+        if !crate::activation_quant_fused::is_cuda_service(client)
+            || hardware.max_units_per_cube < 64
+            || hardware.max_cube_dim.0 < 64
+            || hardware.max_shared_memory_size < 512
+            || hardware.max_bindings < 6
+            || hardware.max_cube_count.0 < shape.outputs as u32
+        {
+            return Err(Error::Unsupported(
+                "CPU-profile BF16 CUDA workgroup/shape capability".into(),
+            ));
+        }
+        service.pinned_cuda_queue()?;
+        if window.identity.descriptor.source != input.source.source_identity()
+            || service.source_identity() != input.source.source_identity()
+            || !window.source.same_issuer(&input.source)
+            || window.identity.rows.len() != shape.rows
+            || window.identity.descriptor.columns != shape.columns
+            || window.identity.window_bytes != shape.weight_bytes
+        {
+            return Err(malformed(
+                "cold original service/source/full geometry binding",
+            ));
+        }
+        if service.remaining_capacity()? < bytes {
+            return Err(Error::Limit(
+                "private BF16 operator capacity before launch".into(),
+            ));
+        }
+        let ticket = service.queue().submit(
+            service.budget(),
+            service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: bytes as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        cancel(cancelled)?;
+        // Other fair work may have retained outputs while this ticket waited.
+        // Refuse before the first initialization command or armed work guard.
+        service.ensure_usable()?;
+        if service.remaining_capacity()? < bytes {
+            return Err(Error::Limit(
+                "private BF16 operator capacity before launch".into(),
+            ));
+        }
+        // Retain authentic staged bytes before ANY allocation/H2D. The armed
+        // original initialization puts this owner in its original quarantine.
+        let owner = Arc::new(SourceOwner {
+            _window: window.owner.clone(),
+            _input: input.bytes.clone(),
+            _input_admission: input._admission.clone(),
+            // Original upload_bytes creates one Vec for each queued H2D.
+            // These actual same-source RAM charges cover those extra copies.
+            _weight_transport: window.source.reserve_workspace(shape.weight_bytes)?,
+            _input_transport: window.source.reserve_workspace(shape.input_bytes + 4)?,
+            _readback_admission: window.source.reserve_workspace(shape.outputs * 12)?,
+        });
+        slot.begin_submission()?;
+        let mut init = service.begin_initialization(correlation)?;
+        init.retain_completion_owner(owner.clone())?;
+        let weights = init.upload_bytes(window.owner.as_slice(), BufferRole::ImmutableWeights)?;
+        let input = init.upload_bytes(&input.bytes, BufferRole::Activation)?;
+        let zero = init.upload_bytes(&0u32.to_le_bytes(), BufferRole::ImmutableWeights)?;
+        let accumulation = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
+        let rounded = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
+        let status = init.allocate_empty(shape.outputs * 4, BufferRole::Scratch)?;
+        let initialized = init.complete_sync().await?;
+        slot.finish()?;
+        drop(initialized);
+        cancel(cancelled)?;
+        let prepared = Self {
+            service,
+            shape,
+            identity: window.identity,
+            owner,
+            weights,
+            input,
+            zero,
+            accumulation,
+            rounded,
+            status,
+        };
+        prepared.validate_bindings()?;
+        Ok(prepared)
+    }
+    fn validate_bindings(&self) -> Result<()> {
+        self.service.ensure_usable()?;
+        self.service.pinned_cuda_queue()?;
+        let expected = [
+            self.shape.weight_bytes,
+            self.shape.input_bytes,
+            4,
+            self.shape.outputs * 4,
+            self.shape.outputs * 4,
+            self.shape.outputs * 4,
+        ];
+        let leases = [
+            &self.weights,
+            &self.input,
+            &self.zero,
+            &self.accumulation,
+            &self.rounded,
+            &self.status,
+        ];
+        let mut ids = std::collections::BTreeSet::new();
+        for (lease, bytes) in leases.into_iter().zip(expected) {
+            let handle = lease.handle();
+            if !Arc::ptr_eq(lease.service(), &self.service)
+                || lease.source_identity() != self.identity.descriptor.source
+                || lease.service_generation() != self.service.generation()
+                || lease.bytes() != bytes
+                || handle.service != self.service.client().service_id()
+                || !ids.insert(lease.id())
+                || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
+                || handle.size_in_used() < bytes as u64
+            {
+                return Err(Error::State(
+                    "BF16 original lease/service/generation/size identity".into(),
+                ));
+            }
+        }
+        if [&self.weights, &self.input, &self.zero]
+            .iter()
+            .any(|lease| !lease.is_initialized())
+        {
+            return Err(Error::State(
+                "BF16 input/source not physically initialized".into(),
+            ));
+        }
+        Ok(())
+    }
+    fn launch(&self, correlation: u64) -> Result<ResidentSubmission<'_>> {
+        self.validate_bindings()?;
+        let accesses = [
+            (self.weights.clone(), DeviceAccess::Read),
+            (self.input.clone(), DeviceAccess::Read),
+            (self.zero.clone(), DeviceAccess::Read),
+            (self.accumulation.clone(), DeviceAccess::Write),
+            (self.rounded.clone(), DeviceAccess::Write),
+            (self.status.clone(), DeviceAccess::Write),
+        ];
+        let mut flight = self.service.submit(correlation, &accesses)?;
+        flight.retain_completion_owner(self.owner.clone())?;
+        let arg = |lease: &ResidentLease, n: usize| -> BufferArg {
+            // SAFETY: exact owning leases are already pinned by the armed
+            // original submission; cached whole allocations passed preflight.
+            unsafe { BufferArg::from_raw_parts(lease.handle(), n) }
+        };
+        // SAFETY: one64-lane cube per full [request,output row], all shared
+        // memory barriers reached by every lane; no padded request or writer.
+        unsafe {
+            project_cpu_bf16::launch_unchecked(
+                self.service.client(),
+                CubeCount::Static(self.shape.outputs as u32, 1, 1),
+                CubeDim::new_1d(64),
+                arg(&self.weights, self.shape.weight_bytes / 4),
+                arg(&self.input, self.shape.input_bytes / 4),
+                arg(&self.zero, 1),
+                arg(&self.accumulation, self.shape.outputs),
+                arg(&self.rounded, self.shape.outputs),
+                arg(&self.status, self.shape.outputs),
+                self.shape.rows,
+                self.shape.columns,
+            );
+        }
+        Ok(flight)
+    }
+    /// Consume the workspace: prior Ready outputs cannot be overwritten by a
+    /// later request. No model/session/state publication is performed here.
+    pub(crate) async fn execute(
+        self,
+        correlation: u64,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<ProjectionReadback> {
+        cancel(cancelled)?;
+        self.validate_bindings()?;
+        let ticket = self.service.queue().submit(
+            self.service.budget(),
+            self.service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: self.shape.owned_bytes()? as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        cancel(cancelled)?;
+        slot.begin_submission()?;
+        let flight = self.launch(correlation)?; // ONE actual native B1/2/4/8 launch.
+        let outputs = [
+            self.accumulation.clone(),
+            self.rounded.clone(),
+            self.status.clone(),
+        ];
+        let reads = [
+            self.accumulation.readback(0, self.shape.outputs * 4)?,
+            self.rounded.readback(0, self.shape.outputs * 4)?,
+            self.status.readback(0, self.shape.outputs * 4)?,
+        ];
+        let (ready, raw) = flight
+            .complete_with_readback(&outputs, &reads, |raw| {
+                if raw.len() != 3
+                    || raw
+                        .iter()
+                        .any(|value| value.len() != self.shape.outputs * 4)
+                    || raw[0]
+                        .chunks_exact(4)
+                        .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
+                    || raw[1].chunks_exact(4).any(|b| {
+                        !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()
+                            || u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0xffff != 0
+                    })
+                    || raw[2]
+                        .chunks_exact(4)
+                        .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != 0)
+                {
+                    return Err(Error::State(
+                        "BF16 physical output/status certification failed".into(),
+                    ));
+                }
+                Ok(())
+            })
+            .await?;
+        slot.finish()?;
+        cancel(cancelled)?;
+        Ok(ProjectionReadback {
+            identity: self.identity.clone(),
+            batch: self.shape.batch,
+            raw,
+            ready,
+            _owner: self.owner.clone(),
+            kernel_dispatches: 1,
+        })
+    }
+    /// Separate process-only physical abort probe. No Ready is fabricated;
+    /// real submitted original work and every source/lease are quarantined.
+    pub(crate) async fn abandon_after_launch(self, correlation: u64) -> Result<()> {
+        self.validate_bindings()?;
+        let ticket = self.service.queue().submit(
+            self.service.budget(),
+            self.service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: self.shape.owned_bytes()? as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        slot.begin_submission()?;
+        let flight = self.launch(correlation)?;
+        drop(flight);
+        drop(slot);
+        if self.service.ensure_usable().is_ok() {
+            return Err(Error::State(
+                "abandoned BF16 submission did not fence original card".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+
+pub(crate) struct ProjectionReadback {
+    pub(crate) identity: Identity,
+    pub(crate) batch: usize,
+    pub(crate) kernel_dispatches: usize,
+    raw: Vec<Bytes>,
+    ready: ResidentReady,
+    _owner: Arc<SourceOwner>,
+}
+impl ProjectionReadback {
+    pub(crate) fn accumulation(&self) -> &[u8] {
+        &self.raw[0]
+    }
+    pub(crate) fn rounded(&self) -> &[u8] {
+        &self.raw[1]
+    }
+    pub(crate) fn ready_outputs(&self) -> usize {
+        self.ready.outputs().len()
+    }
+    pub(crate) fn completion_events(&self) -> usize {
+        self.ready.event_ids().len()
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
@@ -0,0 +1,401 @@
+//! Closed CUDA implementation of the existing CPU BF16 projection contract.
+//! This is not cuBLAS/vLLM arithmetic or a complete HF execution profile.
+// Numerical topology: existing weights::dot_bf16_cpu_avx2, pinned PyTorch
+// cf30153c4c131c8164ee7798e5022d810682e2cb AVX2. Attribution follows that reducer.
+//
+// MIT License
+// Copyright (c) 2023-2024 The ggml authors
+//
+// Permission is hereby granted, free of charge, to any person obtaining a copy
+// of this software and associated documentation files (the "Software"), to deal
+// in the Software without restriction, including without limitation the rights
+// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
+// copies of the Software, and to permit persons to whom the Software is
+// furnished to do so, subject to the following conditions:
+//
+// The above copyright notice and this permission notice shall be included in
+// all copies or substantial portions of the Software.
+//
+// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
+// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
+// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
+// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
+// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
+// THE SOFTWARE.
+use cubecl::prelude::*;
+use std::ops::Range;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedTensorSource, WindowWorkspace};
+use zerogravity_core::accelerator::program::{MatrixAxes, ProgramNumericalProfile, SourceTensor};
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::{Error, Result};
+
+pub(crate) mod qualification;
+mod resident;
+pub(crate) use resident::{PreparedProjection, ProjectionReadback};
+
+const MAX_WEIGHT_BYTES: usize = 32 * 1024 * 1024;
+const MAX_ROWS: usize = 16384;
+const MAX_COLUMNS: usize = 8192;
+const LANES: usize = 64;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+
+fn malformed(message: &str) -> Error {
+    Error::Malformed(format!("CPU-profile BF16 CUDA: {message}"))
+}
+fn finite_bf16(bytes: &[u8]) -> bool {
+    bytes.len().is_multiple_of(2)
+        && bytes
+            .chunks_exact(2)
+            .all(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 != 0x7f80)
+}
+
+#[derive(Clone, Debug, Eq, PartialEq)]
+pub(crate) struct Identity {
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    source_byte_offset: u64,
+    original_byte_offset: u64,
+    original_byte_len: u64,
+    window_bytes: usize,
+    bundle_sha256: String,
+}
+impl Identity {
+    pub(crate) fn descriptor(&self) -> &SourceTensor {
+        &self.descriptor
+    }
+    pub(crate) fn rows(&self) -> Range<usize> {
+        self.rows.clone()
+    }
+    fn require_window(&self, window: &TensorWindow) -> Result<()> {
+        let raw = window.matrix().tensor();
+        if self.descriptor.source != window.matrix().source_identity()
+            || self.descriptor.name != raw.name
+            || self.rows != window.row_range()
+            || self.source_byte_offset != window.source_offset()
+            || self.original_byte_offset != raw.byte_offset
+            || self.original_byte_len != raw.byte_len
+            || self.window_bytes != window.as_slice().len()
+            || self.bundle_sha256 != window.verified_bundle_sha256()
+        {
+            return Err(malformed(
+                "private result lost full original window identity",
+            ));
+        }
+        Ok(())
+    }
+}
+
+/// A real sealed original matrix range, never a reduced SourceTensorRef.
+pub(crate) struct Bf16Window {
+    source: VerifiedTensorSource,
+    owner: Arc<TensorWindow>,
+    identity: Identity,
+}
+impl Bf16Window {
+    pub(crate) fn bind(
+        source: VerifiedTensorSource,
+        owner: Arc<TensorWindow>,
+        descriptor: SourceTensor,
+        rows: Range<usize>,
+        profile: ProgramNumericalProfile,
+    ) -> Result<Self> {
+        let matrix = owner.matrix();
+        let raw = matrix.tensor();
+        let full_bytes = descriptor
+            .columns
+            .checked_mul(descriptor.rows)
+            .and_then(|n| n.checked_mul(2))
+            .ok_or_else(|| malformed("original full size overflow"))?;
+        let offset = u64::try_from(rows.start)
+            .ok()
+            .and_then(|n| n.checked_mul(descriptor.columns as u64))
+            .and_then(|n| n.checked_mul(2))
+            .and_then(|n| raw.byte_offset.checked_add(n))
+            .ok_or_else(|| malformed("original range offset overflow"))?;
+        raw.byte_offset
+            .checked_add(raw.byte_len)
+            .ok_or_else(|| malformed("original full byte end overflow"))?;
+        if profile != PROFILE
+            || !descriptor.source.is_bound()
+            || source.source_identity() != descriptor.source
+            || source.verified_bundle_sha256() != owner.verified_bundle_sha256()
+            || descriptor.source != matrix.source_identity()
+            || descriptor.name != raw.name
+            || descriptor.name.is_empty()
+            || descriptor.kind != QuantKind::Bf16
+            || raw.kind != QuantKind::Bf16
+            || descriptor.axes != MatrixAxes::OutputInput
+            || matrix.axes() != MatrixAxes::OutputInput
+            || raw.shape.as_slice() != [descriptor.rows as u64, descriptor.columns as u64]
+            || matrix.rows() != descriptor.rows
+            || matrix.columns() != descriptor.columns
+            || raw.byte_len != full_bytes as u64
+            || descriptor.rows == 0
+            || descriptor.rows > 1_048_576
+            || rows.start >= rows.end
+            || rows.end > descriptor.rows
+            || owner.row_range() != rows
+            || owner.source_offset() != offset
+        {
+            return Err(malformed(
+                "full original descriptor/profile/issuer/range differs",
+            ));
+        }
+        let shape = Shape::new(rows.len(), descriptor.columns, 1)?;
+        if owner.as_slice().len() != shape.weight_bytes || !finite_bf16(owner.as_slice()) {
+            return Err(malformed("authentic range bytes or finite BF16 weights"));
+        }
+        let identity = Identity {
+            descriptor,
+            rows,
+            source_byte_offset: offset,
+            original_byte_offset: raw.byte_offset,
+            original_byte_len: raw.byte_len,
+            window_bytes: owner.as_slice().len(),
+            bundle_sha256: owner.verified_bundle_sha256().into(),
+        };
+        Ok(Self {
+            source,
+            owner,
+            identity,
+        })
+    }
+}
+
+/// Explicit source-tagged operator input transport; no trained producer claim.
+pub(crate) struct Bf16Inputs {
+    source: VerifiedTensorSource,
+    shape: Shape,
+    bytes: Arc<[u8]>,
+    _admission: Arc<WindowWorkspace>,
+}
+impl Bf16Inputs {
+    pub(crate) fn bind(
+        source: VerifiedTensorSource,
+        profile: ProgramNumericalProfile,
+        rows: usize,
+        columns: usize,
+        batch: usize,
+        raw: &[u8],
+    ) -> Result<Self> {
+        let shape = validate_input(source.source_identity(), profile, rows, columns, batch, raw)?;
+        let admission = Arc::new(source.reserve_workspace(raw.len())?);
+        let bytes = Arc::from(raw); // Same-source admission precedes private copy.
+        Ok(Self {
+            source,
+            shape,
+            bytes,
+            _admission: admission,
+        })
+    }
+}
+fn validate_input(
+    source: zerogravity_core::accelerator::SourceIdentity,
+    profile: ProgramNumericalProfile,
+    rows: usize,
+    columns: usize,
+    batch: usize,
+    bytes: &[u8],
+) -> Result<Shape> {
+    let shape = Shape::new(rows, columns, batch)?;
+    if !source.is_bound()
+        || profile != PROFILE
+        || bytes.len() != shape.input_bytes
+        || !finite_bf16(bytes)
+    {
+        return Err(malformed(
+            "explicit finite BF16 input source/profile/full [B,C]",
+        ));
+    }
+    Ok(shape)
+}
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+struct Shape {
+    rows: usize,
+    columns: usize,
+    batch: usize,
+    weight_bytes: usize,
+    input_bytes: usize,
+    outputs: usize,
+}
+impl Shape {
+    fn new(rows: usize, columns: usize, batch: usize) -> Result<Self> {
+        if rows == 0
+            || rows > MAX_ROWS
+            || columns == 0
+            || columns > MAX_COLUMNS
+            || !columns.is_multiple_of(LANES)
+            || ![1, 2, 4, 8].contains(&batch)
+        {
+            return Err(malformed("closed rows/width-div64/native batch geometry"));
+        }
+        let weight_bytes = rows
+            .checked_mul(columns)
+            .and_then(|n| n.checked_mul(2))
+            .filter(|n| *n <= MAX_WEIGHT_BYTES)
+            .ok_or_else(|| malformed("bounded original weight range"))?;
+        let input_bytes = batch
+            .checked_mul(columns)
+            .and_then(|n| n.checked_mul(2))
+            .ok_or_else(|| malformed("input bytes overflow"))?;
+        let outputs = batch
+            .checked_mul(rows)
+            .filter(|n| u32::try_from(*n).is_ok())
+            .ok_or_else(|| malformed("native output domain"))?;
+        Ok(Self {
+            rows,
+            columns,
+            batch,
+            weight_bytes,
+            input_bytes,
+            outputs,
+        })
+    }
+    fn owned_bytes(self) -> Result<usize> {
+        self.outputs
+            .checked_mul(12)
+            .and_then(|n| n.checked_add(self.weight_bytes))
+            .and_then(|n| n.checked_add(self.input_bytes))
+            .and_then(|n| n.checked_add(4))
+            .ok_or_else(|| malformed("all leased payloads overflow"))
+    }
+}
+
+/// One [request,original output row] per64-thread workgroup. The lane index
+/// is exactly CPU register*8+lane; no subgroup reduction or TensorCore math.
+#[cube(launch_unchecked)]
+fn project_cpu_bf16(
+    weights: &[u32],
+    input: &[u32],
+    zero: &[f32],
+    accumulation: &mut [f32],
+    rounded: &mut [f32],
+    status: &mut [u32],
+    #[comptime] rows: usize,
+    #[comptime] columns: usize,
+) {
+    let output = usize::cast_from(CUBE_POS_X);
+    let request = output / rows;
+    let row = output % rows;
+    let lane = usize::cast_from(UNIT_POS);
+    let mut sums = Shared::<[f32]>::new_slice(64usize);
+    let mut flags = Shared::<[u32]>::new_slice(64usize);
+    // Dynamic immutable +0 defeats CubeCL's literal FAdd(x,0) simplifier.
+    // CPU initializes all64 partials with +0 and adds its empty-tail +0 last.
+    let z = zero[0];
+    let mut accumulator = z;
+    let mut invalid = 0u32;
+    if u32::reinterpret(z) != 0u32 {
+        invalid |= 4u32;
+    }
+    for chunk in 0..columns / 64usize {
+        let column = chunk * 64usize + lane;
+        let wi = row * columns + column;
+        let xi = request * columns + column;
+        let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
+        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
+        if (wb & 0x7f80u32) == 0x7f80u32 {
+            invalid |= 1u32;
+        }
+        if (xb & 0x7f80u32) == 0x7f80u32 {
+            invalid |= 2u32;
+        }
+        accumulator = fma(
+            f32::reinterpret(wb << 16u32),
+            f32::reinterpret(xb << 16u32),
+            accumulator,
+        );
+    }
+    sums[lane] = accumulator;
+    flags[lane] = invalid;
+    sync_cube();
+    if lane < 32usize {
+        let collapsed = sums[lane] + sums[lane + 32usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane < 16usize {
+        let collapsed = sums[lane] + sums[lane + 16usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane < 8usize {
+        let collapsed = sums[lane] + sums[lane + 8usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane == 0usize {
+        let r0 = sums[0] + sums[4];
+        let r1 = sums[1] + sums[5];
+        let r2 = sums[2] + sums[6];
+        let r3 = sums[3] + sums[7];
+        let result = ((r0 + r2) + (r1 + r3)) + z;
+        let bits = u32::reinterpret(result);
+        let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
+        let mut combined = 0u32;
+        for at in 0..64usize {
+            combined |= flags[at];
+        }
+        if (bits & 0x7f800000u32) == 0x7f800000u32 {
+            combined |= 8u32;
+        }
+        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
+            combined |= 16u32;
+        }
+        accumulation[output] = result;
+        rounded[output] = f32::reinterpret(rounded_bits);
+        status[output] = combined;
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn native_shape_and_exact_allocation_accounting() {
+        for batch in [1, 2, 4, 8] {
+            let shape = Shape::new(512, 2048, batch).unwrap();
+            assert_eq!(
+                shape.owned_bytes().unwrap(),
+                2 * 512 * 2048 + 2 * batch * 2048 + 12 * batch * 512 + 4
+            );
+        }
+        for (r, c, b) in [
+            (0, 1024, 1),
+            (1, 0, 1),
+            (1, 63, 1),
+            (1, 65, 1),
+            (1, 8193, 1),
+            (1, 64, 3),
+            (16385, 64, 1),
+            (16384, 8192, 8),
+        ] {
+            assert!(Shape::new(r, c, b).is_err());
+        }
+    }
+    #[test]
+    fn explicit_input_bits_never_cast_or_admit_wrong_source_profile() {
+        let source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation().unwrap();
+        let mut raw = vec![0; 128];
+        for (at, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f].iter().enumerate() {
+            raw[2 * at..2 * at + 2].copy_from_slice(&bits.to_le_bytes());
+        }
+        validate_input(source, PROFILE, 1, 64, 1, &raw).unwrap();
+        raw[0..2].copy_from_slice(&0x7f80u16.to_le_bytes());
+        assert!(validate_input(source, PROFILE, 1, 64, 1, &raw).is_err());
+        assert!(
+            validate_input(
+                source,
+                ProgramNumericalProfile::GgufSourceV1,
+                1,
+                64,
+                1,
+                &[0; 128]
+            )
+            .is_err()
+        );
+    }
+}

--- V1/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -33,11 +33,11 @@
     }
     for request in 0..shape.batch {
         let teacher = request % 2;
-        for row in 0..shape.rows {
+        for (row, expected_value) in expected[teacher].iter().enumerate().take(shape.rows) {
             let at = request * shape.rows + row;
             let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
             let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
-            let expected_bits = expected[teacher][row].to_bits();
+            let expected_bits = expected_value.to_bits();
             let full_row = result.identity.rows.start + row;
             let reference = u32::from(u16::from_le_bytes([
                 torch[teacher][2 * full_row],
@@ -369,6 +369,8 @@
         device_cap,
     )?;
     let key = service.physical_key();
+    let owners_before =
+        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
     let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
     prepared.abandon_after_launch(0).await?;
     drop(window);
@@ -379,6 +381,8 @@
         || service.ensure_usable().is_ok()
         || witness.upgrade().is_none()
         || service.pending_events()? == 0
+        || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
+            <= owners_before
     {
         return Err(Error::State(
             "BF16 actual unknown work lost original leases/source/fence".into(),

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v2-20261020/ROOT_V2.review.patch

Original UTF-8 bytes: 1645; SHA256 `a7b2f632521c494f2831175fa4d45900ac118002e7973f54e1ce21c8b5c09df9`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- V1/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -33,11 +33,11 @@
     }
     for request in 0..shape.batch {
         let teacher = request % 2;
-        for row in 0..shape.rows {
+        for (row, expected_value) in expected[teacher].iter().enumerate().take(shape.rows) {
             let at = request * shape.rows + row;
             let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
             let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
-            let expected_bits = expected[teacher][row].to_bits();
+            let expected_bits = expected_value.to_bits();
             let full_row = result.identity.rows.start + row;
             let reference = u32::from(u16::from_le_bytes([
                 torch[teacher][2 * full_row],
@@ -369,6 +369,8 @@
         device_cap,
     )?;
     let key = service.physical_key();
+    let owners_before =
+        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
     let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
     prepared.abandon_after_launch(0).await?;
     drop(window);
@@ -379,6 +381,8 @@
         || service.ensure_usable().is_ok()
         || witness.upgrade().is_none()
         || service.pending_events()? == 0
+        || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
+            <= owners_before
     {
         return Err(Error::State(
             "BF16 actual unknown work lost original leases/source/fence".into(),

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v2-20261020/WIRING.patch

Original UTF-8 bytes: 4015; SHA256 `561c8382154d40ecbc024e6b566ebb95f12b24f28256a1a7ef6eedfdb9a5a35c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/Cargo.toml
+++ b/crates/zerogravity-gpu/Cargo.toml
@@ -13,7 +13,7 @@
 cuda = ["cubecl/cuda", "cubecl/cuda-cpp"]
 hip = ["cubecl/hip", "cubecl/hip-cpp"]
 all = ["vulkan", "cuda", "hip"]
-qualification = ["dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
+qualification = ["zerogravity-artifacts/cache", "zerogravity-cpu/cache", "dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
 hardware-abort-probe = ["qualification"]
 portable-checkpoint = ["dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/diagnostic-cpu-checkpoint"]
 diagnostic-gdn-trace = ["qualification", "zerogravity-model-qwen35/diagnostic-gdn-trace"]
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -24,6 +24,8 @@
 mod batch_quant_qualification;
 #[cfg(any(feature = "qualification", test))]
 mod batch_recurrent;
+#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
+mod cpu_bf16_projection;
 mod device_activation;
 mod elementwise;
 mod embedding;
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -949,3 +949,42 @@
         "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nUse rope-angles-cuda MODEL for physical angle differential qualification.\nUse rope-angles-model-cuda MODEL [PROVIDER_MiB DEVICE_MiB ROWS] for existing CPU/B1/native model gates.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
     );
 }
+
+/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_window(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_window(
+        device,
+        source,
+        descriptor,
+        rows,
+        inputs,
+        public_outputs,
+        device_cap,
+    )
+    .await
+}
+
+/// Separate fresh process only. Actual unknown completion must stay fenced.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_abandon(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_abandon(
+        device, source, descriptor, rows, input, device_cap,
+    )
+    .await
+}
--- a/crates/zerogravity-gpu/src/resident_leases.rs
+++ b/crates/zerogravity-gpu/src/resident_leases.rs
@@ -814,6 +814,15 @@
     armed: bool,
 }
 impl ResidentInitialization<'_> {
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub(crate) fn retain_completion_owner(
+        &mut self,
+        owner: crate::physical_quarantine::CompletionOwner,
+    ) -> Result<()> {
+        // Attach first: refusal must retain owners of previously queued work.
+        self.completion_owners.push(owner);
+        self.service.ensure_usable()
+    }
     pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
         self.service.ensure_usable()?;
         let allocation_generation = issue(&self.service.next_allocation)?;

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v3-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 49739; SHA256 `5a2179431c41353159cbe51b74555c5f593d88bb2097c27e3dc30af29169c67f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/Cargo.toml
+++ b/crates/zerogravity-gpu/Cargo.toml
@@ -13,7 +13,7 @@
 cuda = ["cubecl/cuda", "cubecl/cuda-cpp"]
 hip = ["cubecl/hip", "cubecl/hip-cpp"]
 all = ["vulkan", "cuda", "hip"]
-qualification = ["dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
+qualification = ["zerogravity-artifacts/cache", "zerogravity-cpu/cache", "dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
 hardware-abort-probe = ["qualification"]
 portable-checkpoint = ["dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/diagnostic-cpu-checkpoint"]
 diagnostic-gdn-trace = ["qualification", "zerogravity-model-qwen35/diagnostic-gdn-trace"]
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -24,6 +24,8 @@
 mod batch_quant_qualification;
 #[cfg(any(feature = "qualification", test))]
 mod batch_recurrent;
+#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
+mod cpu_bf16_projection;
 mod device_activation;
 mod elementwise;
 mod embedding;
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -949,3 +949,42 @@
         "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nUse rope-angles-cuda MODEL for physical angle differential qualification.\nUse rope-angles-model-cuda MODEL [PROVIDER_MiB DEVICE_MiB ROWS] for existing CPU/B1/native model gates.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
     );
 }
+
+/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_window(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_window(
+        device,
+        source,
+        descriptor,
+        rows,
+        inputs,
+        public_outputs,
+        device_cap,
+    )
+    .await
+}
+
+/// Separate fresh process only. Actual unknown completion must stay fenced.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_abandon(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_abandon(
+        device, source, descriptor, rows, input, device_cap,
+    )
+    .await
+}
--- a/crates/zerogravity-gpu/src/resident_leases.rs
+++ b/crates/zerogravity-gpu/src/resident_leases.rs
@@ -814,6 +814,15 @@
     armed: bool,
 }
 impl ResidentInitialization<'_> {
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub(crate) fn retain_completion_owner(
+        &mut self,
+        owner: crate::physical_quarantine::CompletionOwner,
+    ) -> Result<()> {
+        // Attach first: refusal must retain owners of previously queued work.
+        self.completion_owners.push(owner);
+        self.service.ensure_usable()
+    }
     pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
         self.service.ensure_usable()?;
         let allocation_generation = issue(&self.service.next_allocation)?;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -0,0 +1,391 @@
+//! ROOT-only dual CPU-reference / authentic public-output physical proof.
+//! External caller must authenticate the original source and actual archived
+//! teacher inputs/outputs. Synthetic words alone cannot qualify trained math.
+use super::*;
+use crate::resident_leases::ResidentService;
+use cubecl::Device;
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::source_weights::VerifiedTensorSource;
+use zerogravity_core::device_events::BufferRole;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+
+const PROVIDER_CAP: usize = 64 * 1024 * 1024;
+
+fn bytes_word(bytes: &[u8]) -> u32 {
+    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
+}
+fn check_exact(
+    result: &ProjectionReadback,
+    expected: &[Vec<f32>],
+    torch: &[Vec<u8>],
+    shape: Shape,
+) -> Result<usize> {
+    if result.batch != shape.batch
+        || result.kernel_dispatches != 1
+        || result.ready_outputs() != 3
+        || result.completion_events() != 1
+        || result.accumulation().len() != shape.outputs * 4
+        || result.rounded().len() != shape.outputs * 4
+    {
+        return Err(malformed("physical single-dispatch/Ready/output coverage"));
+    }
+    for request in 0..shape.batch {
+        let teacher = request % 2;
+        for row in 0..shape.rows {
+            let at = request * shape.rows + row;
+            let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
+            let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
+            let expected_bits = expected[teacher][row].to_bits();
+            let full_row = result.identity.rows.start + row;
+            let reference = u32::from(u16::from_le_bytes([
+                torch[teacher][2 * full_row],
+                torch[teacher][2 * full_row + 1],
+            ])) << 16;
+            if actual != expected_bits || rounded != reference {
+                return Err(Error::State(format!(
+                    "CPU-profile BF16 CUDA B{} request{request} row{full_row}: accumulation {actual:08x}!={expected_bits:08x}, final {rounded:08x}!={reference:08x}",
+                    shape.batch
+                )));
+            }
+        }
+    }
+    Ok(shape.outputs)
+}
+
+/// Test-only public entry receives actual source and ROOT-authenticated raw
+/// records; no model factory/provider fallback or per-token foreign runtime.
+pub async fn qualify_window(
+    device: Device,
+    source: VerifiedTensorSource,
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> Result<()> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "physical BF16 projection requires release".into(),
+        ));
+    }
+    if !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "physical BF16 projection requires CUDA".into(),
+        ));
+    }
+    Shape::new(rows.len(), descriptor.columns, 1)?;
+    if descriptor.rows == 0 || descriptor.rows > 1_048_576 {
+        return Err(malformed("bounded original full row count"));
+    }
+    let _host = source.reserve_workspace(2 * 1024 * 1024)?;
+    if descriptor.source != source.source_identity()
+        || inputs
+            .iter()
+            .any(|raw| raw.len() != descriptor.columns * 2 || !finite_bf16(raw))
+        || public_outputs
+            .iter()
+            .any(|raw| raw.len() != descriptor.rows * 2 || !finite_bf16(raw))
+    {
+        return Err(malformed(
+            "complete actual original two-teacher input/public output geometry",
+        ));
+    }
+    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
+    Bf16Window::bind(
+        source.clone(),
+        window.clone(),
+        descriptor.clone(),
+        rows.clone(),
+        PROFILE,
+    )?;
+    // Real sealed original window, deliberately wrong full-descriptor tags.
+    // None may become a partial/fake source or reach CUDA initialization.
+    for changed in 0..4 {
+        let mut wrong = descriptor.clone();
+        match changed {
+            0 => {
+                wrong.source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation()?
+            }
+            1 => wrong.axes = MatrixAxes::InputOutput,
+            2 => wrong.rows = wrong.rows.saturating_add(1),
+            _ => wrong.name.push_str(".foreign"),
+        }
+        if Bf16Window::bind(source.clone(), window.clone(), wrong, rows.clone(), PROFILE).is_ok() {
+            return Err(malformed("wrong original full descriptor was accepted"));
+        }
+    }
+    let mut expected = Vec::new();
+    for raw in &inputs {
+        let values = raw
+            .chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>();
+        let cpu = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
+            &descriptor,
+            rows.clone(),
+            &window,
+            &values,
+            None,
+            None,
+        )?;
+        if cpu.identity().descriptor() != &descriptor || cpu.identity().row_range() != rows {
+            return Err(malformed("original CPU oracle full identity/range"));
+        }
+        expected.push(cpu.accumulations().to_vec());
+    }
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("original-cpu-bf16-reference", &descriptor.name)?,
+        PROVIDER_CAP,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != PROVIDER_CAP as u64 {
+        return Err(Error::State(
+            "BF16 diagnostic requires otherwise idle physical budget".into(),
+        ));
+    }
+    // Real original physical allocation, leaving four bytes less than required.
+    // The new binding must refuse BEFORE H2D, allocation or model dispatch.
+    let shape = Shape::new(rows.len(), descriptor.columns, 8)?;
+    let need = shape.owned_bytes()?;
+    let filler_bytes = PROVIDER_CAP
+        .checked_sub(need)
+        .and_then(|n| n.checked_add(4))
+        .ok_or_else(|| malformed("pressure fixture cap"))?;
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: filler_bytes as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(99)?;
+    let filler = init.allocate_empty(filler_bytes, BufferRole::Scratch)?;
+    let initialized = init.complete_sync().await?;
+    slot.finish()?;
+    drop(initialized);
+    let raw = inputs[0].repeat(8);
+    let uploads = service.uploaded_value_bytes();
+    let pressure = PreparedProjection::prepare(
+        service.clone(),
+        Bf16Window::bind(
+            source.clone(),
+            window.clone(),
+            descriptor.clone(),
+            rows.clone(),
+            PROFILE,
+        )?,
+        Bf16Inputs::bind(
+            source.clone(),
+            PROFILE,
+            rows.len(),
+            descriptor.columns,
+            8,
+            &raw,
+        )?,
+        99,
+        None,
+    )
+    .await;
+    if !matches!(pressure, Err(Error::Limit(ref message)) if message == "private BF16 operator capacity before launch")
+        || service.resident_bytes()? != filler_bytes
+        || service.uploaded_value_bytes() != uploads
+        || service.pending_events()? != 0
+    {
+        return Err(Error::State(
+            "BF16 real capacity refusal crossed prelaunch boundary".into(),
+        ));
+    }
+    drop(filler);
+    if service.resident_bytes()? != 0 {
+        return Err(Error::State(
+            "BF16 pressure allocation did not retire".into(),
+        ));
+    }
+    let mut compared = 0usize;
+    let mut actual_dispatches = 0usize;
+    for (case, batch) in [1usize, 1usize, 2usize, 4usize, 8usize]
+        .into_iter()
+        .enumerate()
+    {
+        let shape = Shape::new(rows.len(), descriptor.columns, batch)?;
+        let selected = if case == 1 { 1usize } else { 0usize };
+        let raw = (0..batch)
+            .flat_map(|request| inputs[(request + selected) % 2].iter().copied())
+            .collect::<Vec<_>>();
+        let bind_input = || {
+            Bf16Inputs::bind(
+                source.clone(),
+                PROFILE,
+                rows.len(),
+                descriptor.columns,
+                batch,
+                &raw,
+            )
+        };
+        let source_window = || {
+            Bf16Window::bind(
+                source.clone(),
+                window.clone(),
+                descriptor.clone(),
+                rows.clone(),
+                PROFILE,
+            )
+        };
+        // Queued/prelaunch cancellation cannot create a lease, H2D or kernel.
+        let uploads = service.uploaded_value_bytes();
+        if PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            Some(&AtomicBool::new(true)),
+        )
+        .await
+        .is_ok()
+            || service.resident_bytes()? != 0
+            || service.uploaded_value_bytes() != uploads
+        {
+            return Err(Error::State(
+                "BF16 pre-cancel performed physical work".into(),
+            ));
+        }
+        // Cold initialized leases may also cancel before their first kernel.
+        let prepared = PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            None,
+        )
+        .await?;
+        let uploads = service.uploaded_value_bytes();
+        if prepared
+            .execute(case as u64, Some(&AtomicBool::new(true)))
+            .await
+            .is_ok()
+            || service.resident_bytes()? != 0
+            || service.pending_events()? != 0
+            || service.uploaded_value_bytes() != uploads
+        {
+            return Err(Error::State(
+                "BF16 initialized prelaunch cancellation leaked or submitted work".into(),
+            ));
+        }
+        let prepared = PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            None,
+        )
+        .await?;
+        let result = prepared.execute(case as u64, None).await?;
+        result.identity.require_window(&window)?;
+        if result.identity.descriptor() != &descriptor || result.identity.rows() != rows {
+            return Err(malformed("actual GPU full descriptor/range"));
+        }
+        // Second independent B1 uses actual teacher353, never padded/relabelled.
+        if selected == 0 {
+            compared += check_exact(&result, &expected, &public_outputs, shape)?;
+        } else {
+            let expected_swapped = [expected[1].clone(), expected[0].clone()];
+            let public_swapped = [public_outputs[1].clone(), public_outputs[0].clone()];
+            compared += check_exact(&result, &expected_swapped, &public_swapped, shape)?;
+        }
+        actual_dispatches += result.kernel_dispatches;
+        drop(result);
+        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+            return Err(Error::State(
+                "BF16 physical outputs/events failed retirement".into(),
+            ));
+        }
+        println!(
+            "CPU_PROFILE_BF16_CUDA_OPERATOR_PASS name={} full_rows={} range={}..{} columns={} B={batch} accumulation_bits=exact archived_public_BF16_bits=exact actual_native_dispatches=1 physical_pressure_and_prelaunch_cancel=true state_or_HF_profile_promotion=false",
+            descriptor.name, descriptor.rows, rows.start, rows.end, descriptor.columns
+        );
+    }
+    if compared != 16 * rows.len() || actual_dispatches != 5 {
+        return Err(Error::State(
+            "BF16 physical all-row dual reference coverage incomplete".into(),
+        ));
+    }
+    drop(service);
+    let final_row = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_row.reserved != 0 || final_row.reservations != 0 || final_row.fenced {
+        return Err(Error::State(
+            "BF16 original GPU budget did not finish at zero".into(),
+        ));
+    }
+    Ok(())
+}
+
+/// Separate fresh process only: ACTUAL queued kernel, no trusted completion.
+/// Success here means the original uncertainty quarantine held, never Ready.
+pub async fn qualify_abandon(
+    device: Device,
+    source: VerifiedTensorSource,
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> Result<()> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "physical BF16 abort requires release".into(),
+        ));
+    }
+    if !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "physical BF16 abort requires CUDA".into(),
+        ));
+    }
+    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
+    let witness = Arc::downgrade(&window);
+    let inputs = Bf16Inputs::bind(
+        source.clone(),
+        PROFILE,
+        rows.len(),
+        descriptor.columns,
+        1,
+        &input,
+    )?;
+    let original = Bf16Window::bind(source.clone(), window.clone(), descriptor, rows, PROFILE)?;
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("original-cpu-bf16-reference", "physical-abandon")?,
+        PROVIDER_CAP,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
+    prepared.abandon_after_launch(0).await?;
+    drop(window);
+    drop(source);
+    let actual = ResourceArbiter::process_shared().snapshot(&key)?;
+    if !actual.fenced
+        || actual.reserved < PROVIDER_CAP as u64
+        || service.ensure_usable().is_ok()
+        || witness.upgrade().is_none()
+        || service.pending_events()? == 0
+    {
+        return Err(Error::State(
+            "BF16 actual unknown work lost original leases/source/fence".into(),
+        ));
+    }
+    println!(
+        "CPU_PROFILE_BF16_CUDA_ABANDON_PASS actual_kernel_submitted=true trusted_completion=false Ready=false original_source_and_leases_quarantined=true final_zero_not_claimed=true"
+    );
+    Ok(())
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/resident.rs
@@ -0,0 +1,337 @@
+//! Private operator ownership through the one original resident GPU service.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentReady, ResidentService, ResidentSubmission};
+use cubecl::bytes::Bytes;
+use std::sync::atomic::{AtomicBool, Ordering};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+struct SourceOwner {
+    _window: Arc<TensorWindow>,
+    _input: Arc<[u8]>,
+    _input_admission: Arc<WindowWorkspace>,
+    _weight_transport: WindowWorkspace,
+    _input_transport: WindowWorkspace,
+    _readback_admission: WindowWorkspace,
+}
+fn cancel(flag: Option<&AtomicBool>) -> Result<()> {
+    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        Err(Error::State("private BF16 projection cancelled".into()))
+    } else {
+        Ok(())
+    }
+}
+
+pub(crate) struct PreparedProjection {
+    service: Arc<ResidentService>,
+    shape: Shape,
+    identity: Identity,
+    owner: Arc<SourceOwner>,
+    weights: ResidentLease,
+    input: ResidentLease,
+    zero: ResidentLease,
+    accumulation: ResidentLease,
+    rounded: ResidentLease,
+    status: ResidentLease,
+}
+impl PreparedProjection {
+    /// A PURE operator correlation, never an invented SessionAuthority/root.
+    /// The caller supplies an existing pinned original service, not a device
+    /// from which a second context/queue might be created.
+    pub(crate) async fn prepare(
+        service: Arc<ResidentService>,
+        window: Bf16Window,
+        input: Bf16Inputs,
+        correlation: u64,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        cancel(cancelled)?;
+        let shape = input.shape;
+        let bytes = shape.owned_bytes()?;
+        let client = service.client();
+        let hardware = &client.properties().hardware;
+        if !crate::activation_quant_fused::is_cuda_service(client)
+            || hardware.max_units_per_cube < 64
+            || hardware.max_cube_dim.0 < 64
+            || hardware.max_shared_memory_size < 512
+            || hardware.max_bindings < 6
+            || hardware.max_cube_count.0 < shape.outputs as u32
+        {
+            return Err(Error::Unsupported(
+                "CPU-profile BF16 CUDA workgroup/shape capability".into(),
+            ));
+        }
+        service.pinned_cuda_queue()?;
+        if window.identity.descriptor.source != input.source.source_identity()
+            || service.source_identity() != input.source.source_identity()
+            || !window.source.same_issuer(&input.source)
+            || window.identity.rows.len() != shape.rows
+            || window.identity.descriptor.columns != shape.columns
+            || window.identity.window_bytes != shape.weight_bytes
+        {
+            return Err(malformed(
+                "cold original service/source/full geometry binding",
+            ));
+        }
+        if service.remaining_capacity()? < bytes {
+            return Err(Error::Limit(
+                "private BF16 operator capacity before launch".into(),
+            ));
+        }
+        let ticket = service.queue().submit(
+            service.budget(),
+            service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: bytes as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        cancel(cancelled)?;
+        // Other fair work may have retained outputs while this ticket waited.
+        // Refuse before the first initialization command or armed work guard.
+        service.ensure_usable()?;
+        if service.remaining_capacity()? < bytes {
+            return Err(Error::Limit(
+                "private BF16 operator capacity before launch".into(),
+            ));
+        }
+        // Retain authentic staged bytes before ANY allocation/H2D. The armed
+        // original initialization puts this owner in its original quarantine.
+        let owner = Arc::new(SourceOwner {
+            _window: window.owner.clone(),
+            _input: input.bytes.clone(),
+            _input_admission: input._admission.clone(),
+            // Original upload_bytes creates one Vec for each queued H2D.
+            // These actual same-source RAM charges cover those extra copies.
+            _weight_transport: window.source.reserve_workspace(shape.weight_bytes)?,
+            _input_transport: window.source.reserve_workspace(shape.input_bytes + 4)?,
+            _readback_admission: window.source.reserve_workspace(shape.outputs * 12)?,
+        });
+        slot.begin_submission()?;
+        let mut init = service.begin_initialization(correlation)?;
+        init.retain_completion_owner(owner.clone())?;
+        let weights = init.upload_bytes(window.owner.as_slice(), BufferRole::ImmutableWeights)?;
+        let input = init.upload_bytes(&input.bytes, BufferRole::Activation)?;
+        let zero = init.upload_bytes(&0u32.to_le_bytes(), BufferRole::ImmutableWeights)?;
+        let accumulation = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
+        let rounded = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
+        let status = init.allocate_empty(shape.outputs * 4, BufferRole::Scratch)?;
+        let initialized = init.complete_sync().await?;
+        slot.finish()?;
+        drop(initialized);
+        cancel(cancelled)?;
+        let prepared = Self {
+            service,
+            shape,
+            identity: window.identity,
+            owner,
+            weights,
+            input,
+            zero,
+            accumulation,
+            rounded,
+            status,
+        };
+        prepared.validate_bindings()?;
+        Ok(prepared)
+    }
+    fn validate_bindings(&self) -> Result<()> {
+        self.service.ensure_usable()?;
+        self.service.pinned_cuda_queue()?;
+        let expected = [
+            self.shape.weight_bytes,
+            self.shape.input_bytes,
+            4,
+            self.shape.outputs * 4,
+            self.shape.outputs * 4,
+            self.shape.outputs * 4,
+        ];
+        let leases = [
+            &self.weights,
+            &self.input,
+            &self.zero,
+            &self.accumulation,
+            &self.rounded,
+            &self.status,
+        ];
+        let mut ids = std::collections::BTreeSet::new();
+        for (lease, bytes) in leases.into_iter().zip(expected) {
+            let handle = lease.handle();
+            if !Arc::ptr_eq(lease.service(), &self.service)
+                || lease.source_identity() != self.identity.descriptor.source
+                || lease.service_generation() != self.service.generation()
+                || lease.bytes() != bytes
+                || handle.service != self.service.client().service_id()
+                || !ids.insert(lease.id())
+                || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
+                || handle.size_in_used() < bytes as u64
+            {
+                return Err(Error::State(
+                    "BF16 original lease/service/generation/size identity".into(),
+                ));
+            }
+        }
+        if [&self.weights, &self.input, &self.zero]
+            .iter()
+            .any(|lease| !lease.is_initialized())
+        {
+            return Err(Error::State(
+                "BF16 input/source not physically initialized".into(),
+            ));
+        }
+        Ok(())
+    }
+    fn launch(&self, correlation: u64) -> Result<ResidentSubmission<'_>> {
+        self.validate_bindings()?;
+        let accesses = [
+            (self.weights.clone(), DeviceAccess::Read),
+            (self.input.clone(), DeviceAccess::Read),
+            (self.zero.clone(), DeviceAccess::Read),
+            (self.accumulation.clone(), DeviceAccess::Write),
+            (self.rounded.clone(), DeviceAccess::Write),
+            (self.status.clone(), DeviceAccess::Write),
+        ];
+        let mut flight = self.service.submit(correlation, &accesses)?;
+        flight.retain_completion_owner(self.owner.clone())?;
+        let arg = |lease: &ResidentLease, n: usize| -> BufferArg {
+            // SAFETY: exact owning leases are already pinned by the armed
+            // original submission; cached whole allocations passed preflight.
+            unsafe { BufferArg::from_raw_parts(lease.handle(), n) }
+        };
+        // SAFETY: one64-lane cube per full [request,output row], all shared
+        // memory barriers reached by every lane; no padded request or writer.
+        unsafe {
+            project_cpu_bf16::launch_unchecked(
+                self.service.client(),
+                CubeCount::Static(self.shape.outputs as u32, 1, 1),
+                CubeDim::new_1d(64),
+                arg(&self.weights, self.shape.weight_bytes / 4),
+                arg(&self.input, self.shape.input_bytes / 4),
+                arg(&self.zero, 1),
+                arg(&self.accumulation, self.shape.outputs),
+                arg(&self.rounded, self.shape.outputs),
+                arg(&self.status, self.shape.outputs),
+                self.shape.rows,
+                self.shape.columns,
+            );
+        }
+        Ok(flight)
+    }
+    /// Consume the workspace: prior Ready outputs cannot be overwritten by a
+    /// later request. No model/session/state publication is performed here.
+    pub(crate) async fn execute(
+        self,
+        correlation: u64,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<ProjectionReadback> {
+        cancel(cancelled)?;
+        self.validate_bindings()?;
+        let ticket = self.service.queue().submit(
+            self.service.budget(),
+            self.service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: self.shape.owned_bytes()? as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        cancel(cancelled)?;
+        slot.begin_submission()?;
+        let flight = self.launch(correlation)?; // ONE actual native B1/2/4/8 launch.
+        let outputs = [
+            self.accumulation.clone(),
+            self.rounded.clone(),
+            self.status.clone(),
+        ];
+        let reads = [
+            self.accumulation.readback(0, self.shape.outputs * 4)?,
+            self.rounded.readback(0, self.shape.outputs * 4)?,
+            self.status.readback(0, self.shape.outputs * 4)?,
+        ];
+        let (ready, raw) = flight
+            .complete_with_readback(&outputs, &reads, |raw| {
+                if raw.len() != 3
+                    || raw
+                        .iter()
+                        .any(|value| value.len() != self.shape.outputs * 4)
+                    || raw[0]
+                        .chunks_exact(4)
+                        .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
+                    || raw[1].chunks_exact(4).any(|b| {
+                        !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()
+                            || u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0xffff != 0
+                    })
+                    || raw[2]
+                        .chunks_exact(4)
+                        .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != 0)
+                {
+                    return Err(Error::State(
+                        "BF16 physical output/status certification failed".into(),
+                    ));
+                }
+                Ok(())
+            })
+            .await?;
+        slot.finish()?;
+        cancel(cancelled)?;
+        Ok(ProjectionReadback {
+            identity: self.identity.clone(),
+            batch: self.shape.batch,
+            raw,
+            ready,
+            _owner: self.owner.clone(),
+            kernel_dispatches: 1,
+        })
+    }
+    /// Separate process-only physical abort probe. No Ready is fabricated;
+    /// real submitted original work and every source/lease are quarantined.
+    pub(crate) async fn abandon_after_launch(self, correlation: u64) -> Result<()> {
+        self.validate_bindings()?;
+        let ticket = self.service.queue().submit(
+            self.service.budget(),
+            self.service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: self.shape.owned_bytes()? as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        slot.begin_submission()?;
+        let flight = self.launch(correlation)?;
+        drop(flight);
+        drop(slot);
+        if self.service.ensure_usable().is_ok() {
+            return Err(Error::State(
+                "abandoned BF16 submission did not fence original card".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+
+pub(crate) struct ProjectionReadback {
+    pub(crate) identity: Identity,
+    pub(crate) batch: usize,
+    pub(crate) kernel_dispatches: usize,
+    raw: Vec<Bytes>,
+    ready: ResidentReady,
+    _owner: Arc<SourceOwner>,
+}
+impl ProjectionReadback {
+    pub(crate) fn accumulation(&self) -> &[u8] {
+        &self.raw[0]
+    }
+    pub(crate) fn rounded(&self) -> &[u8] {
+        &self.raw[1]
+    }
+    pub(crate) fn ready_outputs(&self) -> usize {
+        self.ready.outputs().len()
+    }
+    pub(crate) fn completion_events(&self) -> usize {
+        self.ready.event_ids().len()
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
@@ -0,0 +1,401 @@
+//! Closed CUDA implementation of the existing CPU BF16 projection contract.
+//! This is not cuBLAS/vLLM arithmetic or a complete HF execution profile.
+// Numerical topology: existing weights::dot_bf16_cpu_avx2, pinned PyTorch
+// cf30153c4c131c8164ee7798e5022d810682e2cb AVX2. Attribution follows that reducer.
+//
+// MIT License
+// Copyright (c) 2023-2024 The ggml authors
+//
+// Permission is hereby granted, free of charge, to any person obtaining a copy
+// of this software and associated documentation files (the "Software"), to deal
+// in the Software without restriction, including without limitation the rights
+// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
+// copies of the Software, and to permit persons to whom the Software is
+// furnished to do so, subject to the following conditions:
+//
+// The above copyright notice and this permission notice shall be included in
+// all copies or substantial portions of the Software.
+//
+// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
+// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
+// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
+// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
+// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
+// THE SOFTWARE.
+use cubecl::prelude::*;
+use std::ops::Range;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedTensorSource, WindowWorkspace};
+use zerogravity_core::accelerator::program::{MatrixAxes, ProgramNumericalProfile, SourceTensor};
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::{Error, Result};
+
+pub(crate) mod qualification;
+mod resident;
+pub(crate) use resident::{PreparedProjection, ProjectionReadback};
+
+const MAX_WEIGHT_BYTES: usize = 32 * 1024 * 1024;
+const MAX_ROWS: usize = 16384;
+const MAX_COLUMNS: usize = 8192;
+const LANES: usize = 64;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+
+fn malformed(message: &str) -> Error {
+    Error::Malformed(format!("CPU-profile BF16 CUDA: {message}"))
+}
+fn finite_bf16(bytes: &[u8]) -> bool {
+    bytes.len().is_multiple_of(2)
+        && bytes
+            .chunks_exact(2)
+            .all(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 != 0x7f80)
+}
+
+#[derive(Clone, Debug, Eq, PartialEq)]
+pub(crate) struct Identity {
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    source_byte_offset: u64,
+    original_byte_offset: u64,
+    original_byte_len: u64,
+    window_bytes: usize,
+    bundle_sha256: String,
+}
+impl Identity {
+    pub(crate) fn descriptor(&self) -> &SourceTensor {
+        &self.descriptor
+    }
+    pub(crate) fn rows(&self) -> Range<usize> {
+        self.rows.clone()
+    }
+    fn require_window(&self, window: &TensorWindow) -> Result<()> {
+        let raw = window.matrix().tensor();
+        if self.descriptor.source != window.matrix().source_identity()
+            || self.descriptor.name != raw.name
+            || self.rows != window.row_range()
+            || self.source_byte_offset != window.source_offset()
+            || self.original_byte_offset != raw.byte_offset
+            || self.original_byte_len != raw.byte_len
+            || self.window_bytes != window.as_slice().len()
+            || self.bundle_sha256 != window.verified_bundle_sha256()
+        {
+            return Err(malformed(
+                "private result lost full original window identity",
+            ));
+        }
+        Ok(())
+    }
+}
+
+/// A real sealed original matrix range, never a reduced SourceTensorRef.
+pub(crate) struct Bf16Window {
+    source: VerifiedTensorSource,
+    owner: Arc<TensorWindow>,
+    identity: Identity,
+}
+impl Bf16Window {
+    pub(crate) fn bind(
+        source: VerifiedTensorSource,
+        owner: Arc<TensorWindow>,
+        descriptor: SourceTensor,
+        rows: Range<usize>,
+        profile: ProgramNumericalProfile,
+    ) -> Result<Self> {
+        let matrix = owner.matrix();
+        let raw = matrix.tensor();
+        let full_bytes = descriptor
+            .columns
+            .checked_mul(descriptor.rows)
+            .and_then(|n| n.checked_mul(2))
+            .ok_or_else(|| malformed("original full size overflow"))?;
+        let offset = u64::try_from(rows.start)
+            .ok()
+            .and_then(|n| n.checked_mul(descriptor.columns as u64))
+            .and_then(|n| n.checked_mul(2))
+            .and_then(|n| raw.byte_offset.checked_add(n))
+            .ok_or_else(|| malformed("original range offset overflow"))?;
+        raw.byte_offset
+            .checked_add(raw.byte_len)
+            .ok_or_else(|| malformed("original full byte end overflow"))?;
+        if profile != PROFILE
+            || !descriptor.source.is_bound()
+            || source.source_identity() != descriptor.source
+            || source.verified_bundle_sha256() != owner.verified_bundle_sha256()
+            || descriptor.source != matrix.source_identity()
+            || descriptor.name != raw.name
+            || descriptor.name.is_empty()
+            || descriptor.kind != QuantKind::Bf16
+            || raw.kind != QuantKind::Bf16
+            || descriptor.axes != MatrixAxes::OutputInput
+            || matrix.axes() != MatrixAxes::OutputInput
+            || raw.shape.as_slice() != [descriptor.rows as u64, descriptor.columns as u64]
+            || matrix.rows() != descriptor.rows
+            || matrix.columns() != descriptor.columns
+            || raw.byte_len != full_bytes as u64
+            || descriptor.rows == 0
+            || descriptor.rows > 1_048_576
+            || rows.start >= rows.end
+            || rows.end > descriptor.rows
+            || owner.row_range() != rows
+            || owner.source_offset() != offset
+        {
+            return Err(malformed(
+                "full original descriptor/profile/issuer/range differs",
+            ));
+        }
+        let shape = Shape::new(rows.len(), descriptor.columns, 1)?;
+        if owner.as_slice().len() != shape.weight_bytes || !finite_bf16(owner.as_slice()) {
+            return Err(malformed("authentic range bytes or finite BF16 weights"));
+        }
+        let identity = Identity {
+            descriptor,
+            rows,
+            source_byte_offset: offset,
+            original_byte_offset: raw.byte_offset,
+            original_byte_len: raw.byte_len,
+            window_bytes: owner.as_slice().len(),
+            bundle_sha256: owner.verified_bundle_sha256().into(),
+        };
+        Ok(Self {
+            source,
+            owner,
+            identity,
+        })
+    }
+}
+
+/// Explicit source-tagged operator input transport; no trained producer claim.
+pub(crate) struct Bf16Inputs {
+    source: VerifiedTensorSource,
+    shape: Shape,
+    bytes: Arc<[u8]>,
+    _admission: Arc<WindowWorkspace>,
+}
+impl Bf16Inputs {
+    pub(crate) fn bind(
+        source: VerifiedTensorSource,
+        profile: ProgramNumericalProfile,
+        rows: usize,
+        columns: usize,
+        batch: usize,
+        raw: &[u8],
+    ) -> Result<Self> {
+        let shape = validate_input(source.source_identity(), profile, rows, columns, batch, raw)?;
+        let admission = Arc::new(source.reserve_workspace(raw.len())?);
+        let bytes = Arc::from(raw); // Same-source admission precedes private copy.
+        Ok(Self {
+            source,
+            shape,
+            bytes,
+            _admission: admission,
+        })
+    }
+}
+fn validate_input(
+    source: zerogravity_core::accelerator::SourceIdentity,
+    profile: ProgramNumericalProfile,
+    rows: usize,
+    columns: usize,
+    batch: usize,
+    bytes: &[u8],
+) -> Result<Shape> {
+    let shape = Shape::new(rows, columns, batch)?;
+    if !source.is_bound()
+        || profile != PROFILE
+        || bytes.len() != shape.input_bytes
+        || !finite_bf16(bytes)
+    {
+        return Err(malformed(
+            "explicit finite BF16 input source/profile/full [B,C]",
+        ));
+    }
+    Ok(shape)
+}
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+struct Shape {
+    rows: usize,
+    columns: usize,
+    batch: usize,
+    weight_bytes: usize,
+    input_bytes: usize,
+    outputs: usize,
+}
+impl Shape {
+    fn new(rows: usize, columns: usize, batch: usize) -> Result<Self> {
+        if rows == 0
+            || rows > MAX_ROWS
+            || columns == 0
+            || columns > MAX_COLUMNS
+            || !columns.is_multiple_of(LANES)
+            || ![1, 2, 4, 8].contains(&batch)
+        {
+            return Err(malformed("closed rows/width-div64/native batch geometry"));
+        }
+        let weight_bytes = rows
+            .checked_mul(columns)
+            .and_then(|n| n.checked_mul(2))
+            .filter(|n| *n <= MAX_WEIGHT_BYTES)
+            .ok_or_else(|| malformed("bounded original weight range"))?;
+        let input_bytes = batch
+            .checked_mul(columns)
+            .and_then(|n| n.checked_mul(2))
+            .ok_or_else(|| malformed("input bytes overflow"))?;
+        let outputs = batch
+            .checked_mul(rows)
+            .filter(|n| u32::try_from(*n).is_ok())
+            .ok_or_else(|| malformed("native output domain"))?;
+        Ok(Self {
+            rows,
+            columns,
+            batch,
+            weight_bytes,
+            input_bytes,
+            outputs,
+        })
+    }
+    fn owned_bytes(self) -> Result<usize> {
+        self.outputs
+            .checked_mul(12)
+            .and_then(|n| n.checked_add(self.weight_bytes))
+            .and_then(|n| n.checked_add(self.input_bytes))
+            .and_then(|n| n.checked_add(4))
+            .ok_or_else(|| malformed("all leased payloads overflow"))
+    }
+}
+
+/// One [request,original output row] per64-thread workgroup. The lane index
+/// is exactly CPU register*8+lane; no subgroup reduction or TensorCore math.
+#[cube(launch_unchecked)]
+fn project_cpu_bf16(
+    weights: &[u32],
+    input: &[u32],
+    zero: &[f32],
+    accumulation: &mut [f32],
+    rounded: &mut [f32],
+    status: &mut [u32],
+    #[comptime] rows: usize,
+    #[comptime] columns: usize,
+) {
+    let output = usize::cast_from(CUBE_POS_X);
+    let request = output / rows;
+    let row = output % rows;
+    let lane = usize::cast_from(UNIT_POS);
+    let mut sums = Shared::<[f32]>::new_slice(64usize);
+    let mut flags = Shared::<[u32]>::new_slice(64usize);
+    // Dynamic immutable +0 defeats CubeCL's literal FAdd(x,0) simplifier.
+    // CPU initializes all64 partials with +0 and adds its empty-tail +0 last.
+    let z = zero[0];
+    let mut accumulator = z;
+    let mut invalid = 0u32;
+    if u32::reinterpret(z) != 0u32 {
+        invalid |= 4u32;
+    }
+    for chunk in 0..columns / 64usize {
+        let column = chunk * 64usize + lane;
+        let wi = row * columns + column;
+        let xi = request * columns + column;
+        let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
+        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
+        if (wb & 0x7f80u32) == 0x7f80u32 {
+            invalid |= 1u32;
+        }
+        if (xb & 0x7f80u32) == 0x7f80u32 {
+            invalid |= 2u32;
+        }
+        accumulator = fma(
+            f32::reinterpret(wb << 16u32),
+            f32::reinterpret(xb << 16u32),
+            accumulator,
+        );
+    }
+    sums[lane] = accumulator;
+    flags[lane] = invalid;
+    sync_cube();
+    if lane < 32usize {
+        let collapsed = sums[lane] + sums[lane + 32usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane < 16usize {
+        let collapsed = sums[lane] + sums[lane + 16usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane < 8usize {
+        let collapsed = sums[lane] + sums[lane + 8usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane == 0usize {
+        let r0 = sums[0] + sums[4];
+        let r1 = sums[1] + sums[5];
+        let r2 = sums[2] + sums[6];
+        let r3 = sums[3] + sums[7];
+        let result = ((r0 + r2) + (r1 + r3)) + z;
+        let bits = u32::reinterpret(result);
+        let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
+        let mut combined = 0u32;
+        for at in 0..64usize {
+            combined |= flags[at];
+        }
+        if (bits & 0x7f800000u32) == 0x7f800000u32 {
+            combined |= 8u32;
+        }
+        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
+            combined |= 16u32;
+        }
+        accumulation[output] = result;
+        rounded[output] = f32::reinterpret(rounded_bits);
+        status[output] = combined;
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn native_shape_and_exact_allocation_accounting() {
+        for batch in [1, 2, 4, 8] {
+            let shape = Shape::new(512, 2048, batch).unwrap();
+            assert_eq!(
+                shape.owned_bytes().unwrap(),
+                2 * 512 * 2048 + 2 * batch * 2048 + 12 * batch * 512 + 4
+            );
+        }
+        for (r, c, b) in [
+            (0, 1024, 1),
+            (1, 0, 1),
+            (1, 63, 1),
+            (1, 65, 1),
+            (1, 8193, 1),
+            (1, 64, 3),
+            (16385, 64, 1),
+            (16384, 8192, 8),
+        ] {
+            assert!(Shape::new(r, c, b).is_err());
+        }
+    }
+    #[test]
+    fn explicit_input_bits_never_cast_or_admit_wrong_source_profile() {
+        let source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation().unwrap();
+        let mut raw = vec![0; 128];
+        for (at, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f].iter().enumerate() {
+            raw[2 * at..2 * at + 2].copy_from_slice(&bits.to_le_bytes());
+        }
+        validate_input(source, PROFILE, 1, 64, 1, &raw).unwrap();
+        raw[0..2].copy_from_slice(&0x7f80u16.to_le_bytes());
+        assert!(validate_input(source, PROFILE, 1, 64, 1, &raw).is_err());
+        assert!(
+            validate_input(
+                source,
+                ProgramNumericalProfile::GgufSourceV1,
+                1,
+                64,
+                1,
+                &[0; 128]
+            )
+            .is_err()
+        );
+    }
+}

--- V1/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -33,11 +33,11 @@
     }
     for request in 0..shape.batch {
         let teacher = request % 2;
-        for row in 0..shape.rows {
+        for (row, expected_value) in expected[teacher].iter().enumerate().take(shape.rows) {
             let at = request * shape.rows + row;
             let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
             let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
-            let expected_bits = expected[teacher][row].to_bits();
+            let expected_bits = expected_value.to_bits();
             let full_row = result.identity.rows.start + row;
             let reference = u32::from(u16::from_le_bytes([
                 torch[teacher][2 * full_row],
@@ -369,6 +369,8 @@
         device_cap,
     )?;
     let key = service.physical_key();
+    let owners_before =
+        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
     let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
     prepared.abandon_after_launch(0).await?;
     drop(window);
@@ -379,6 +381,8 @@
         || service.ensure_usable().is_ok()
         || witness.upgrade().is_none()
         || service.pending_events()? == 0
+        || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
+            <= owners_before
     {
         return Err(Error::State(
             "BF16 actual unknown work lost original leases/source/fence".into(),

--- ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V3/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -22,7 +22,9 @@
     torch: &[Vec<u8>],
     shape: Shape,
 ) -> Result<usize> {
-    if result.batch != shape.batch
+    if expected.len() != 2
+        || expected.iter().any(|values| values.len() != shape.rows)
+        || result.batch != shape.batch
         || result.kernel_dispatches != 1
         || result.ready_outputs() != 3
         || result.completion_events() != 1

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v3-20261020/ROOT_V3.review.patch

Original UTF-8 bytes: 534; SHA256 `86e5823b9d5b2b438ad24d587f7aab723690f62ab45ba0adc826689fd35d1d3a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V3/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -22,7 +22,9 @@
     torch: &[Vec<u8>],
     shape: Shape,
 ) -> Result<usize> {
-    if result.batch != shape.batch
+    if expected.len() != 2
+        || expected.iter().any(|values| values.len() != shape.rows)
+        || result.batch != shape.batch
         || result.kernel_dispatches != 1
         || result.ready_outputs() != 3
         || result.completion_events() != 1

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v3-20261020/WIRING.patch

Original UTF-8 bytes: 4015; SHA256 `561c8382154d40ecbc024e6b566ebb95f12b24f28256a1a7ef6eedfdb9a5a35c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/Cargo.toml
+++ b/crates/zerogravity-gpu/Cargo.toml
@@ -13,7 +13,7 @@
 cuda = ["cubecl/cuda", "cubecl/cuda-cpp"]
 hip = ["cubecl/hip", "cubecl/hip-cpp"]
 all = ["vulkan", "cuda", "hip"]
-qualification = ["dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
+qualification = ["zerogravity-artifacts/cache", "zerogravity-cpu/cache", "dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
 hardware-abort-probe = ["qualification"]
 portable-checkpoint = ["dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/diagnostic-cpu-checkpoint"]
 diagnostic-gdn-trace = ["qualification", "zerogravity-model-qwen35/diagnostic-gdn-trace"]
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -24,6 +24,8 @@
 mod batch_quant_qualification;
 #[cfg(any(feature = "qualification", test))]
 mod batch_recurrent;
+#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
+mod cpu_bf16_projection;
 mod device_activation;
 mod elementwise;
 mod embedding;
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -949,3 +949,42 @@
         "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nUse rope-angles-cuda MODEL for physical angle differential qualification.\nUse rope-angles-model-cuda MODEL [PROVIDER_MiB DEVICE_MiB ROWS] for existing CPU/B1/native model gates.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
     );
 }
+
+/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_window(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_window(
+        device,
+        source,
+        descriptor,
+        rows,
+        inputs,
+        public_outputs,
+        device_cap,
+    )
+    .await
+}
+
+/// Separate fresh process only. Actual unknown completion must stay fenced.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_abandon(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_abandon(
+        device, source, descriptor, rows, input, device_cap,
+    )
+    .await
+}
--- a/crates/zerogravity-gpu/src/resident_leases.rs
+++ b/crates/zerogravity-gpu/src/resident_leases.rs
@@ -814,6 +814,15 @@
     armed: bool,
 }
 impl ResidentInitialization<'_> {
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub(crate) fn retain_completion_owner(
+        &mut self,
+        owner: crate::physical_quarantine::CompletionOwner,
+    ) -> Result<()> {
+        // Attach first: refusal must retain owners of previously queued work.
+        self.completion_owners.push(owner);
+        self.service.ensure_usable()
+    }
     pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
         self.service.ensure_usable()?;
         let allocation_generation = issue(&self.service.next_allocation)?;

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v4-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 49739; SHA256 `5a2179431c41353159cbe51b74555c5f593d88bb2097c27e3dc30af29169c67f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/Cargo.toml
+++ b/crates/zerogravity-gpu/Cargo.toml
@@ -13,7 +13,7 @@
 cuda = ["cubecl/cuda", "cubecl/cuda-cpp"]
 hip = ["cubecl/hip", "cubecl/hip-cpp"]
 all = ["vulkan", "cuda", "hip"]
-qualification = ["dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
+qualification = ["zerogravity-artifacts/cache", "zerogravity-cpu/cache", "dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
 hardware-abort-probe = ["qualification"]
 portable-checkpoint = ["dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/diagnostic-cpu-checkpoint"]
 diagnostic-gdn-trace = ["qualification", "zerogravity-model-qwen35/diagnostic-gdn-trace"]
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -24,6 +24,8 @@
 mod batch_quant_qualification;
 #[cfg(any(feature = "qualification", test))]
 mod batch_recurrent;
+#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
+mod cpu_bf16_projection;
 mod device_activation;
 mod elementwise;
 mod embedding;
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -949,3 +949,42 @@
         "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nUse rope-angles-cuda MODEL for physical angle differential qualification.\nUse rope-angles-model-cuda MODEL [PROVIDER_MiB DEVICE_MiB ROWS] for existing CPU/B1/native model gates.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
     );
 }
+
+/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_window(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_window(
+        device,
+        source,
+        descriptor,
+        rows,
+        inputs,
+        public_outputs,
+        device_cap,
+    )
+    .await
+}
+
+/// Separate fresh process only. Actual unknown completion must stay fenced.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_abandon(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_abandon(
+        device, source, descriptor, rows, input, device_cap,
+    )
+    .await
+}
--- a/crates/zerogravity-gpu/src/resident_leases.rs
+++ b/crates/zerogravity-gpu/src/resident_leases.rs
@@ -814,6 +814,15 @@
     armed: bool,
 }
 impl ResidentInitialization<'_> {
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub(crate) fn retain_completion_owner(
+        &mut self,
+        owner: crate::physical_quarantine::CompletionOwner,
+    ) -> Result<()> {
+        // Attach first: refusal must retain owners of previously queued work.
+        self.completion_owners.push(owner);
+        self.service.ensure_usable()
+    }
     pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
         self.service.ensure_usable()?;
         let allocation_generation = issue(&self.service.next_allocation)?;
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -0,0 +1,391 @@
+//! ROOT-only dual CPU-reference / authentic public-output physical proof.
+//! External caller must authenticate the original source and actual archived
+//! teacher inputs/outputs. Synthetic words alone cannot qualify trained math.
+use super::*;
+use crate::resident_leases::ResidentService;
+use cubecl::Device;
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::source_weights::VerifiedTensorSource;
+use zerogravity_core::device_events::BufferRole;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
+};
+
+const PROVIDER_CAP: usize = 64 * 1024 * 1024;
+
+fn bytes_word(bytes: &[u8]) -> u32 {
+    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
+}
+fn check_exact(
+    result: &ProjectionReadback,
+    expected: &[Vec<f32>],
+    torch: &[Vec<u8>],
+    shape: Shape,
+) -> Result<usize> {
+    if result.batch != shape.batch
+        || result.kernel_dispatches != 1
+        || result.ready_outputs() != 3
+        || result.completion_events() != 1
+        || result.accumulation().len() != shape.outputs * 4
+        || result.rounded().len() != shape.outputs * 4
+    {
+        return Err(malformed("physical single-dispatch/Ready/output coverage"));
+    }
+    for request in 0..shape.batch {
+        let teacher = request % 2;
+        for row in 0..shape.rows {
+            let at = request * shape.rows + row;
+            let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
+            let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
+            let expected_bits = expected[teacher][row].to_bits();
+            let full_row = result.identity.rows.start + row;
+            let reference = u32::from(u16::from_le_bytes([
+                torch[teacher][2 * full_row],
+                torch[teacher][2 * full_row + 1],
+            ])) << 16;
+            if actual != expected_bits || rounded != reference {
+                return Err(Error::State(format!(
+                    "CPU-profile BF16 CUDA B{} request{request} row{full_row}: accumulation {actual:08x}!={expected_bits:08x}, final {rounded:08x}!={reference:08x}",
+                    shape.batch
+                )));
+            }
+        }
+    }
+    Ok(shape.outputs)
+}
+
+/// Test-only public entry receives actual source and ROOT-authenticated raw
+/// records; no model factory/provider fallback or per-token foreign runtime.
+pub async fn qualify_window(
+    device: Device,
+    source: VerifiedTensorSource,
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> Result<()> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "physical BF16 projection requires release".into(),
+        ));
+    }
+    if !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "physical BF16 projection requires CUDA".into(),
+        ));
+    }
+    Shape::new(rows.len(), descriptor.columns, 1)?;
+    if descriptor.rows == 0 || descriptor.rows > 1_048_576 {
+        return Err(malformed("bounded original full row count"));
+    }
+    let _host = source.reserve_workspace(2 * 1024 * 1024)?;
+    if descriptor.source != source.source_identity()
+        || inputs
+            .iter()
+            .any(|raw| raw.len() != descriptor.columns * 2 || !finite_bf16(raw))
+        || public_outputs
+            .iter()
+            .any(|raw| raw.len() != descriptor.rows * 2 || !finite_bf16(raw))
+    {
+        return Err(malformed(
+            "complete actual original two-teacher input/public output geometry",
+        ));
+    }
+    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
+    Bf16Window::bind(
+        source.clone(),
+        window.clone(),
+        descriptor.clone(),
+        rows.clone(),
+        PROFILE,
+    )?;
+    // Real sealed original window, deliberately wrong full-descriptor tags.
+    // None may become a partial/fake source or reach CUDA initialization.
+    for changed in 0..4 {
+        let mut wrong = descriptor.clone();
+        match changed {
+            0 => {
+                wrong.source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation()?
+            }
+            1 => wrong.axes = MatrixAxes::InputOutput,
+            2 => wrong.rows = wrong.rows.saturating_add(1),
+            _ => wrong.name.push_str(".foreign"),
+        }
+        if Bf16Window::bind(source.clone(), window.clone(), wrong, rows.clone(), PROFILE).is_ok() {
+            return Err(malformed("wrong original full descriptor was accepted"));
+        }
+    }
+    let mut expected = Vec::new();
+    for raw in &inputs {
+        let values = raw
+            .chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>();
+        let cpu = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
+            &descriptor,
+            rows.clone(),
+            &window,
+            &values,
+            None,
+            None,
+        )?;
+        if cpu.identity().descriptor() != &descriptor || cpu.identity().row_range() != rows {
+            return Err(malformed("original CPU oracle full identity/range"));
+        }
+        expected.push(cpu.accumulations().to_vec());
+    }
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("original-cpu-bf16-reference", &descriptor.name)?,
+        PROVIDER_CAP,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != PROVIDER_CAP as u64 {
+        return Err(Error::State(
+            "BF16 diagnostic requires otherwise idle physical budget".into(),
+        ));
+    }
+    // Real original physical allocation, leaving four bytes less than required.
+    // The new binding must refuse BEFORE H2D, allocation or model dispatch.
+    let shape = Shape::new(rows.len(), descriptor.columns, 8)?;
+    let need = shape.owned_bytes()?;
+    let filler_bytes = PROVIDER_CAP
+        .checked_sub(need)
+        .and_then(|n| n.checked_add(4))
+        .ok_or_else(|| malformed("pressure fixture cap"))?;
+    let ticket = service.queue().submit(
+        service.budget(),
+        service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Scratch,
+            bytes: filler_bytes as u64,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    slot.begin_submission()?;
+    let mut init = service.begin_initialization(99)?;
+    let filler = init.allocate_empty(filler_bytes, BufferRole::Scratch)?;
+    let initialized = init.complete_sync().await?;
+    slot.finish()?;
+    drop(initialized);
+    let raw = inputs[0].repeat(8);
+    let uploads = service.uploaded_value_bytes();
+    let pressure = PreparedProjection::prepare(
+        service.clone(),
+        Bf16Window::bind(
+            source.clone(),
+            window.clone(),
+            descriptor.clone(),
+            rows.clone(),
+            PROFILE,
+        )?,
+        Bf16Inputs::bind(
+            source.clone(),
+            PROFILE,
+            rows.len(),
+            descriptor.columns,
+            8,
+            &raw,
+        )?,
+        99,
+        None,
+    )
+    .await;
+    if !matches!(pressure, Err(Error::Limit(ref message)) if message == "private BF16 operator capacity before launch")
+        || service.resident_bytes()? != filler_bytes
+        || service.uploaded_value_bytes() != uploads
+        || service.pending_events()? != 0
+    {
+        return Err(Error::State(
+            "BF16 real capacity refusal crossed prelaunch boundary".into(),
+        ));
+    }
+    drop(filler);
+    if service.resident_bytes()? != 0 {
+        return Err(Error::State(
+            "BF16 pressure allocation did not retire".into(),
+        ));
+    }
+    let mut compared = 0usize;
+    let mut actual_dispatches = 0usize;
+    for (case, batch) in [1usize, 1usize, 2usize, 4usize, 8usize]
+        .into_iter()
+        .enumerate()
+    {
+        let shape = Shape::new(rows.len(), descriptor.columns, batch)?;
+        let selected = if case == 1 { 1usize } else { 0usize };
+        let raw = (0..batch)
+            .flat_map(|request| inputs[(request + selected) % 2].iter().copied())
+            .collect::<Vec<_>>();
+        let bind_input = || {
+            Bf16Inputs::bind(
+                source.clone(),
+                PROFILE,
+                rows.len(),
+                descriptor.columns,
+                batch,
+                &raw,
+            )
+        };
+        let source_window = || {
+            Bf16Window::bind(
+                source.clone(),
+                window.clone(),
+                descriptor.clone(),
+                rows.clone(),
+                PROFILE,
+            )
+        };
+        // Queued/prelaunch cancellation cannot create a lease, H2D or kernel.
+        let uploads = service.uploaded_value_bytes();
+        if PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            Some(&AtomicBool::new(true)),
+        )
+        .await
+        .is_ok()
+            || service.resident_bytes()? != 0
+            || service.uploaded_value_bytes() != uploads
+        {
+            return Err(Error::State(
+                "BF16 pre-cancel performed physical work".into(),
+            ));
+        }
+        // Cold initialized leases may also cancel before their first kernel.
+        let prepared = PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            None,
+        )
+        .await?;
+        let uploads = service.uploaded_value_bytes();
+        if prepared
+            .execute(case as u64, Some(&AtomicBool::new(true)))
+            .await
+            .is_ok()
+            || service.resident_bytes()? != 0
+            || service.pending_events()? != 0
+            || service.uploaded_value_bytes() != uploads
+        {
+            return Err(Error::State(
+                "BF16 initialized prelaunch cancellation leaked or submitted work".into(),
+            ));
+        }
+        let prepared = PreparedProjection::prepare(
+            service.clone(),
+            source_window()?,
+            bind_input()?,
+            case as u64,
+            None,
+        )
+        .await?;
+        let result = prepared.execute(case as u64, None).await?;
+        result.identity.require_window(&window)?;
+        if result.identity.descriptor() != &descriptor || result.identity.rows() != rows {
+            return Err(malformed("actual GPU full descriptor/range"));
+        }
+        // Second independent B1 uses actual teacher353, never padded/relabelled.
+        if selected == 0 {
+            compared += check_exact(&result, &expected, &public_outputs, shape)?;
+        } else {
+            let expected_swapped = [expected[1].clone(), expected[0].clone()];
+            let public_swapped = [public_outputs[1].clone(), public_outputs[0].clone()];
+            compared += check_exact(&result, &expected_swapped, &public_swapped, shape)?;
+        }
+        actual_dispatches += result.kernel_dispatches;
+        drop(result);
+        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
+            return Err(Error::State(
+                "BF16 physical outputs/events failed retirement".into(),
+            ));
+        }
+        println!(
+            "CPU_PROFILE_BF16_CUDA_OPERATOR_PASS name={} full_rows={} range={}..{} columns={} B={batch} accumulation_bits=exact archived_public_BF16_bits=exact actual_native_dispatches=1 physical_pressure_and_prelaunch_cancel=true state_or_HF_profile_promotion=false",
+            descriptor.name, descriptor.rows, rows.start, rows.end, descriptor.columns
+        );
+    }
+    if compared != 16 * rows.len() || actual_dispatches != 5 {
+        return Err(Error::State(
+            "BF16 physical all-row dual reference coverage incomplete".into(),
+        ));
+    }
+    drop(service);
+    let final_row = ResourceArbiter::process_shared().snapshot(&key)?;
+    if final_row.reserved != 0 || final_row.reservations != 0 || final_row.fenced {
+        return Err(Error::State(
+            "BF16 original GPU budget did not finish at zero".into(),
+        ));
+    }
+    Ok(())
+}
+
+/// Separate fresh process only: ACTUAL queued kernel, no trusted completion.
+/// Success here means the original uncertainty quarantine held, never Ready.
+pub async fn qualify_abandon(
+    device: Device,
+    source: VerifiedTensorSource,
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> Result<()> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "physical BF16 abort requires release".into(),
+        ));
+    }
+    if !matches!(device, Device::Cuda(_)) {
+        return Err(Error::Unsupported(
+            "physical BF16 abort requires CUDA".into(),
+        ));
+    }
+    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
+    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
+    let witness = Arc::downgrade(&window);
+    let inputs = Bf16Inputs::bind(
+        source.clone(),
+        PROFILE,
+        rows.len(),
+        descriptor.columns,
+        1,
+        &input,
+    )?;
+    let original = Bf16Window::bind(source.clone(), window.clone(), descriptor, rows, PROFILE)?;
+    let service = ResidentService::new_pinned_cuda(
+        device,
+        source.source_identity(),
+        ResourceOwner::new("original-cpu-bf16-reference", "physical-abandon")?,
+        PROVIDER_CAP,
+        device_cap,
+    )?;
+    let key = service.physical_key();
+    let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
+    prepared.abandon_after_launch(0).await?;
+    drop(window);
+    drop(source);
+    let actual = ResourceArbiter::process_shared().snapshot(&key)?;
+    if !actual.fenced
+        || actual.reserved < PROVIDER_CAP as u64
+        || service.ensure_usable().is_ok()
+        || witness.upgrade().is_none()
+        || service.pending_events()? == 0
+    {
+        return Err(Error::State(
+            "BF16 actual unknown work lost original leases/source/fence".into(),
+        ));
+    }
+    println!(
+        "CPU_PROFILE_BF16_CUDA_ABANDON_PASS actual_kernel_submitted=true trusted_completion=false Ready=false original_source_and_leases_quarantined=true final_zero_not_claimed=true"
+    );
+    Ok(())
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection/resident.rs
@@ -0,0 +1,337 @@
+//! Private operator ownership through the one original resident GPU service.
+use super::*;
+use crate::resident_leases::{ResidentLease, ResidentReady, ResidentService, ResidentSubmission};
+use cubecl::bytes::Bytes;
+use std::sync::atomic::{AtomicBool, Ordering};
+use zerogravity_core::device_events::{BufferRole, DeviceAccess};
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+struct SourceOwner {
+    _window: Arc<TensorWindow>,
+    _input: Arc<[u8]>,
+    _input_admission: Arc<WindowWorkspace>,
+    _weight_transport: WindowWorkspace,
+    _input_transport: WindowWorkspace,
+    _readback_admission: WindowWorkspace,
+}
+fn cancel(flag: Option<&AtomicBool>) -> Result<()> {
+    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        Err(Error::State("private BF16 projection cancelled".into()))
+    } else {
+        Ok(())
+    }
+}
+
+pub(crate) struct PreparedProjection {
+    service: Arc<ResidentService>,
+    shape: Shape,
+    identity: Identity,
+    owner: Arc<SourceOwner>,
+    weights: ResidentLease,
+    input: ResidentLease,
+    zero: ResidentLease,
+    accumulation: ResidentLease,
+    rounded: ResidentLease,
+    status: ResidentLease,
+}
+impl PreparedProjection {
+    /// A PURE operator correlation, never an invented SessionAuthority/root.
+    /// The caller supplies an existing pinned original service, not a device
+    /// from which a second context/queue might be created.
+    pub(crate) async fn prepare(
+        service: Arc<ResidentService>,
+        window: Bf16Window,
+        input: Bf16Inputs,
+        correlation: u64,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        cancel(cancelled)?;
+        let shape = input.shape;
+        let bytes = shape.owned_bytes()?;
+        let client = service.client();
+        let hardware = &client.properties().hardware;
+        if !crate::activation_quant_fused::is_cuda_service(client)
+            || hardware.max_units_per_cube < 64
+            || hardware.max_cube_dim.0 < 64
+            || hardware.max_shared_memory_size < 512
+            || hardware.max_bindings < 6
+            || hardware.max_cube_count.0 < shape.outputs as u32
+        {
+            return Err(Error::Unsupported(
+                "CPU-profile BF16 CUDA workgroup/shape capability".into(),
+            ));
+        }
+        service.pinned_cuda_queue()?;
+        if window.identity.descriptor.source != input.source.source_identity()
+            || service.source_identity() != input.source.source_identity()
+            || !window.source.same_issuer(&input.source)
+            || window.identity.rows.len() != shape.rows
+            || window.identity.descriptor.columns != shape.columns
+            || window.identity.window_bytes != shape.weight_bytes
+        {
+            return Err(malformed(
+                "cold original service/source/full geometry binding",
+            ));
+        }
+        if service.remaining_capacity()? < bytes {
+            return Err(Error::Limit(
+                "private BF16 operator capacity before launch".into(),
+            ));
+        }
+        let ticket = service.queue().submit(
+            service.budget(),
+            service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: bytes as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        cancel(cancelled)?;
+        // Other fair work may have retained outputs while this ticket waited.
+        // Refuse before the first initialization command or armed work guard.
+        service.ensure_usable()?;
+        if service.remaining_capacity()? < bytes {
+            return Err(Error::Limit(
+                "private BF16 operator capacity before launch".into(),
+            ));
+        }
+        // Retain authentic staged bytes before ANY allocation/H2D. The armed
+        // original initialization puts this owner in its original quarantine.
+        let owner = Arc::new(SourceOwner {
+            _window: window.owner.clone(),
+            _input: input.bytes.clone(),
+            _input_admission: input._admission.clone(),
+            // Original upload_bytes creates one Vec for each queued H2D.
+            // These actual same-source RAM charges cover those extra copies.
+            _weight_transport: window.source.reserve_workspace(shape.weight_bytes)?,
+            _input_transport: window.source.reserve_workspace(shape.input_bytes + 4)?,
+            _readback_admission: window.source.reserve_workspace(shape.outputs * 12)?,
+        });
+        slot.begin_submission()?;
+        let mut init = service.begin_initialization(correlation)?;
+        init.retain_completion_owner(owner.clone())?;
+        let weights = init.upload_bytes(window.owner.as_slice(), BufferRole::ImmutableWeights)?;
+        let input = init.upload_bytes(&input.bytes, BufferRole::Activation)?;
+        let zero = init.upload_bytes(&0u32.to_le_bytes(), BufferRole::ImmutableWeights)?;
+        let accumulation = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
+        let rounded = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
+        let status = init.allocate_empty(shape.outputs * 4, BufferRole::Scratch)?;
+        let initialized = init.complete_sync().await?;
+        slot.finish()?;
+        drop(initialized);
+        cancel(cancelled)?;
+        let prepared = Self {
+            service,
+            shape,
+            identity: window.identity,
+            owner,
+            weights,
+            input,
+            zero,
+            accumulation,
+            rounded,
+            status,
+        };
+        prepared.validate_bindings()?;
+        Ok(prepared)
+    }
+    fn validate_bindings(&self) -> Result<()> {
+        self.service.ensure_usable()?;
+        self.service.pinned_cuda_queue()?;
+        let expected = [
+            self.shape.weight_bytes,
+            self.shape.input_bytes,
+            4,
+            self.shape.outputs * 4,
+            self.shape.outputs * 4,
+            self.shape.outputs * 4,
+        ];
+        let leases = [
+            &self.weights,
+            &self.input,
+            &self.zero,
+            &self.accumulation,
+            &self.rounded,
+            &self.status,
+        ];
+        let mut ids = std::collections::BTreeSet::new();
+        for (lease, bytes) in leases.into_iter().zip(expected) {
+            let handle = lease.handle();
+            if !Arc::ptr_eq(lease.service(), &self.service)
+                || lease.source_identity() != self.identity.descriptor.source
+                || lease.service_generation() != self.service.generation()
+                || lease.bytes() != bytes
+                || handle.service != self.service.client().service_id()
+                || !ids.insert(lease.id())
+                || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
+                || handle.size_in_used() < bytes as u64
+            {
+                return Err(Error::State(
+                    "BF16 original lease/service/generation/size identity".into(),
+                ));
+            }
+        }
+        if [&self.weights, &self.input, &self.zero]
+            .iter()
+            .any(|lease| !lease.is_initialized())
+        {
+            return Err(Error::State(
+                "BF16 input/source not physically initialized".into(),
+            ));
+        }
+        Ok(())
+    }
+    fn launch(&self, correlation: u64) -> Result<ResidentSubmission<'_>> {
+        self.validate_bindings()?;
+        let accesses = [
+            (self.weights.clone(), DeviceAccess::Read),
+            (self.input.clone(), DeviceAccess::Read),
+            (self.zero.clone(), DeviceAccess::Read),
+            (self.accumulation.clone(), DeviceAccess::Write),
+            (self.rounded.clone(), DeviceAccess::Write),
+            (self.status.clone(), DeviceAccess::Write),
+        ];
+        let mut flight = self.service.submit(correlation, &accesses)?;
+        flight.retain_completion_owner(self.owner.clone())?;
+        let arg = |lease: &ResidentLease, n: usize| -> BufferArg {
+            // SAFETY: exact owning leases are already pinned by the armed
+            // original submission; cached whole allocations passed preflight.
+            unsafe { BufferArg::from_raw_parts(lease.handle(), n) }
+        };
+        // SAFETY: one64-lane cube per full [request,output row], all shared
+        // memory barriers reached by every lane; no padded request or writer.
+        unsafe {
+            project_cpu_bf16::launch_unchecked(
+                self.service.client(),
+                CubeCount::Static(self.shape.outputs as u32, 1, 1),
+                CubeDim::new_1d(64),
+                arg(&self.weights, self.shape.weight_bytes / 4),
+                arg(&self.input, self.shape.input_bytes / 4),
+                arg(&self.zero, 1),
+                arg(&self.accumulation, self.shape.outputs),
+                arg(&self.rounded, self.shape.outputs),
+                arg(&self.status, self.shape.outputs),
+                self.shape.rows,
+                self.shape.columns,
+            );
+        }
+        Ok(flight)
+    }
+    /// Consume the workspace: prior Ready outputs cannot be overwritten by a
+    /// later request. No model/session/state publication is performed here.
+    pub(crate) async fn execute(
+        self,
+        correlation: u64,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<ProjectionReadback> {
+        cancel(cancelled)?;
+        self.validate_bindings()?;
+        let ticket = self.service.queue().submit(
+            self.service.budget(),
+            self.service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: self.shape.owned_bytes()? as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        cancel(cancelled)?;
+        slot.begin_submission()?;
+        let flight = self.launch(correlation)?; // ONE actual native B1/2/4/8 launch.
+        let outputs = [
+            self.accumulation.clone(),
+            self.rounded.clone(),
+            self.status.clone(),
+        ];
+        let reads = [
+            self.accumulation.readback(0, self.shape.outputs * 4)?,
+            self.rounded.readback(0, self.shape.outputs * 4)?,
+            self.status.readback(0, self.shape.outputs * 4)?,
+        ];
+        let (ready, raw) = flight
+            .complete_with_readback(&outputs, &reads, |raw| {
+                if raw.len() != 3
+                    || raw
+                        .iter()
+                        .any(|value| value.len() != self.shape.outputs * 4)
+                    || raw[0]
+                        .chunks_exact(4)
+                        .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
+                    || raw[1].chunks_exact(4).any(|b| {
+                        !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()
+                            || u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0xffff != 0
+                    })
+                    || raw[2]
+                        .chunks_exact(4)
+                        .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != 0)
+                {
+                    return Err(Error::State(
+                        "BF16 physical output/status certification failed".into(),
+                    ));
+                }
+                Ok(())
+            })
+            .await?;
+        slot.finish()?;
+        cancel(cancelled)?;
+        Ok(ProjectionReadback {
+            identity: self.identity.clone(),
+            batch: self.shape.batch,
+            raw,
+            ready,
+            _owner: self.owner.clone(),
+            kernel_dispatches: 1,
+        })
+    }
+    /// Separate process-only physical abort probe. No Ready is fabricated;
+    /// real submitted original work and every source/lease are quarantined.
+    pub(crate) async fn abandon_after_launch(self, correlation: u64) -> Result<()> {
+        self.validate_bindings()?;
+        let ticket = self.service.queue().submit(
+            self.service.budget(),
+            self.service.owner().clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: self.shape.owned_bytes()? as u64,
+            }],
+            None,
+        )?;
+        let mut slot = ticket.wait().await?;
+        slot.begin_submission()?;
+        let flight = self.launch(correlation)?;
+        drop(flight);
+        drop(slot);
+        if self.service.ensure_usable().is_ok() {
+            return Err(Error::State(
+                "abandoned BF16 submission did not fence original card".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+
+pub(crate) struct ProjectionReadback {
+    pub(crate) identity: Identity,
+    pub(crate) batch: usize,
+    pub(crate) kernel_dispatches: usize,
+    raw: Vec<Bytes>,
+    ready: ResidentReady,
+    _owner: Arc<SourceOwner>,
+}
+impl ProjectionReadback {
+    pub(crate) fn accumulation(&self) -> &[u8] {
+        &self.raw[0]
+    }
+    pub(crate) fn rounded(&self) -> &[u8] {
+        &self.raw[1]
+    }
+    pub(crate) fn ready_outputs(&self) -> usize {
+        self.ready.outputs().len()
+    }
+    pub(crate) fn completion_events(&self) -> usize {
+        self.ready.event_ids().len()
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/cpu_bf16_projection.rs
@@ -0,0 +1,401 @@
+//! Closed CUDA implementation of the existing CPU BF16 projection contract.
+//! This is not cuBLAS/vLLM arithmetic or a complete HF execution profile.
+// Numerical topology: existing weights::dot_bf16_cpu_avx2, pinned PyTorch
+// cf30153c4c131c8164ee7798e5022d810682e2cb AVX2. Attribution follows that reducer.
+//
+// MIT License
+// Copyright (c) 2023-2024 The ggml authors
+//
+// Permission is hereby granted, free of charge, to any person obtaining a copy
+// of this software and associated documentation files (the "Software"), to deal
+// in the Software without restriction, including without limitation the rights
+// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
+// copies of the Software, and to permit persons to whom the Software is
+// furnished to do so, subject to the following conditions:
+//
+// The above copyright notice and this permission notice shall be included in
+// all copies or substantial portions of the Software.
+//
+// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
+// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
+// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
+// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
+// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
+// THE SOFTWARE.
+use cubecl::prelude::*;
+use std::ops::Range;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedTensorSource, WindowWorkspace};
+use zerogravity_core::accelerator::program::{MatrixAxes, ProgramNumericalProfile, SourceTensor};
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::{Error, Result};
+
+pub(crate) mod qualification;
+mod resident;
+pub(crate) use resident::{PreparedProjection, ProjectionReadback};
+
+const MAX_WEIGHT_BYTES: usize = 32 * 1024 * 1024;
+const MAX_ROWS: usize = 16384;
+const MAX_COLUMNS: usize = 8192;
+const LANES: usize = 64;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+
+fn malformed(message: &str) -> Error {
+    Error::Malformed(format!("CPU-profile BF16 CUDA: {message}"))
+}
+fn finite_bf16(bytes: &[u8]) -> bool {
+    bytes.len().is_multiple_of(2)
+        && bytes
+            .chunks_exact(2)
+            .all(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 != 0x7f80)
+}
+
+#[derive(Clone, Debug, Eq, PartialEq)]
+pub(crate) struct Identity {
+    descriptor: SourceTensor,
+    rows: Range<usize>,
+    source_byte_offset: u64,
+    original_byte_offset: u64,
+    original_byte_len: u64,
+    window_bytes: usize,
+    bundle_sha256: String,
+}
+impl Identity {
+    pub(crate) fn descriptor(&self) -> &SourceTensor {
+        &self.descriptor
+    }
+    pub(crate) fn rows(&self) -> Range<usize> {
+        self.rows.clone()
+    }
+    fn require_window(&self, window: &TensorWindow) -> Result<()> {
+        let raw = window.matrix().tensor();
+        if self.descriptor.source != window.matrix().source_identity()
+            || self.descriptor.name != raw.name
+            || self.rows != window.row_range()
+            || self.source_byte_offset != window.source_offset()
+            || self.original_byte_offset != raw.byte_offset
+            || self.original_byte_len != raw.byte_len
+            || self.window_bytes != window.as_slice().len()
+            || self.bundle_sha256 != window.verified_bundle_sha256()
+        {
+            return Err(malformed(
+                "private result lost full original window identity",
+            ));
+        }
+        Ok(())
+    }
+}
+
+/// A real sealed original matrix range, never a reduced SourceTensorRef.
+pub(crate) struct Bf16Window {
+    source: VerifiedTensorSource,
+    owner: Arc<TensorWindow>,
+    identity: Identity,
+}
+impl Bf16Window {
+    pub(crate) fn bind(
+        source: VerifiedTensorSource,
+        owner: Arc<TensorWindow>,
+        descriptor: SourceTensor,
+        rows: Range<usize>,
+        profile: ProgramNumericalProfile,
+    ) -> Result<Self> {
+        let matrix = owner.matrix();
+        let raw = matrix.tensor();
+        let full_bytes = descriptor
+            .columns
+            .checked_mul(descriptor.rows)
+            .and_then(|n| n.checked_mul(2))
+            .ok_or_else(|| malformed("original full size overflow"))?;
+        let offset = u64::try_from(rows.start)
+            .ok()
+            .and_then(|n| n.checked_mul(descriptor.columns as u64))
+            .and_then(|n| n.checked_mul(2))
+            .and_then(|n| raw.byte_offset.checked_add(n))
+            .ok_or_else(|| malformed("original range offset overflow"))?;
+        raw.byte_offset
+            .checked_add(raw.byte_len)
+            .ok_or_else(|| malformed("original full byte end overflow"))?;
+        if profile != PROFILE
+            || !descriptor.source.is_bound()
+            || source.source_identity() != descriptor.source
+            || source.verified_bundle_sha256() != owner.verified_bundle_sha256()
+            || descriptor.source != matrix.source_identity()
+            || descriptor.name != raw.name
+            || descriptor.name.is_empty()
+            || descriptor.kind != QuantKind::Bf16
+            || raw.kind != QuantKind::Bf16
+            || descriptor.axes != MatrixAxes::OutputInput
+            || matrix.axes() != MatrixAxes::OutputInput
+            || raw.shape.as_slice() != [descriptor.rows as u64, descriptor.columns as u64]
+            || matrix.rows() != descriptor.rows
+            || matrix.columns() != descriptor.columns
+            || raw.byte_len != full_bytes as u64
+            || descriptor.rows == 0
+            || descriptor.rows > 1_048_576
+            || rows.start >= rows.end
+            || rows.end > descriptor.rows
+            || owner.row_range() != rows
+            || owner.source_offset() != offset
+        {
+            return Err(malformed(
+                "full original descriptor/profile/issuer/range differs",
+            ));
+        }
+        let shape = Shape::new(rows.len(), descriptor.columns, 1)?;
+        if owner.as_slice().len() != shape.weight_bytes || !finite_bf16(owner.as_slice()) {
+            return Err(malformed("authentic range bytes or finite BF16 weights"));
+        }
+        let identity = Identity {
+            descriptor,
+            rows,
+            source_byte_offset: offset,
+            original_byte_offset: raw.byte_offset,
+            original_byte_len: raw.byte_len,
+            window_bytes: owner.as_slice().len(),
+            bundle_sha256: owner.verified_bundle_sha256().into(),
+        };
+        Ok(Self {
+            source,
+            owner,
+            identity,
+        })
+    }
+}
+
+/// Explicit source-tagged operator input transport; no trained producer claim.
+pub(crate) struct Bf16Inputs {
+    source: VerifiedTensorSource,
+    shape: Shape,
+    bytes: Arc<[u8]>,
+    _admission: Arc<WindowWorkspace>,
+}
+impl Bf16Inputs {
+    pub(crate) fn bind(
+        source: VerifiedTensorSource,
+        profile: ProgramNumericalProfile,
+        rows: usize,
+        columns: usize,
+        batch: usize,
+        raw: &[u8],
+    ) -> Result<Self> {
+        let shape = validate_input(source.source_identity(), profile, rows, columns, batch, raw)?;
+        let admission = Arc::new(source.reserve_workspace(raw.len())?);
+        let bytes = Arc::from(raw); // Same-source admission precedes private copy.
+        Ok(Self {
+            source,
+            shape,
+            bytes,
+            _admission: admission,
+        })
+    }
+}
+fn validate_input(
+    source: zerogravity_core::accelerator::SourceIdentity,
+    profile: ProgramNumericalProfile,
+    rows: usize,
+    columns: usize,
+    batch: usize,
+    bytes: &[u8],
+) -> Result<Shape> {
+    let shape = Shape::new(rows, columns, batch)?;
+    if !source.is_bound()
+        || profile != PROFILE
+        || bytes.len() != shape.input_bytes
+        || !finite_bf16(bytes)
+    {
+        return Err(malformed(
+            "explicit finite BF16 input source/profile/full [B,C]",
+        ));
+    }
+    Ok(shape)
+}
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+struct Shape {
+    rows: usize,
+    columns: usize,
+    batch: usize,
+    weight_bytes: usize,
+    input_bytes: usize,
+    outputs: usize,
+}
+impl Shape {
+    fn new(rows: usize, columns: usize, batch: usize) -> Result<Self> {
+        if rows == 0
+            || rows > MAX_ROWS
+            || columns == 0
+            || columns > MAX_COLUMNS
+            || !columns.is_multiple_of(LANES)
+            || ![1, 2, 4, 8].contains(&batch)
+        {
+            return Err(malformed("closed rows/width-div64/native batch geometry"));
+        }
+        let weight_bytes = rows
+            .checked_mul(columns)
+            .and_then(|n| n.checked_mul(2))
+            .filter(|n| *n <= MAX_WEIGHT_BYTES)
+            .ok_or_else(|| malformed("bounded original weight range"))?;
+        let input_bytes = batch
+            .checked_mul(columns)
+            .and_then(|n| n.checked_mul(2))
+            .ok_or_else(|| malformed("input bytes overflow"))?;
+        let outputs = batch
+            .checked_mul(rows)
+            .filter(|n| u32::try_from(*n).is_ok())
+            .ok_or_else(|| malformed("native output domain"))?;
+        Ok(Self {
+            rows,
+            columns,
+            batch,
+            weight_bytes,
+            input_bytes,
+            outputs,
+        })
+    }
+    fn owned_bytes(self) -> Result<usize> {
+        self.outputs
+            .checked_mul(12)
+            .and_then(|n| n.checked_add(self.weight_bytes))
+            .and_then(|n| n.checked_add(self.input_bytes))
+            .and_then(|n| n.checked_add(4))
+            .ok_or_else(|| malformed("all leased payloads overflow"))
+    }
+}
+
+/// One [request,original output row] per64-thread workgroup. The lane index
+/// is exactly CPU register*8+lane; no subgroup reduction or TensorCore math.
+#[cube(launch_unchecked)]
+fn project_cpu_bf16(
+    weights: &[u32],
+    input: &[u32],
+    zero: &[f32],
+    accumulation: &mut [f32],
+    rounded: &mut [f32],
+    status: &mut [u32],
+    #[comptime] rows: usize,
+    #[comptime] columns: usize,
+) {
+    let output = usize::cast_from(CUBE_POS_X);
+    let request = output / rows;
+    let row = output % rows;
+    let lane = usize::cast_from(UNIT_POS);
+    let mut sums = Shared::<[f32]>::new_slice(64usize);
+    let mut flags = Shared::<[u32]>::new_slice(64usize);
+    // Dynamic immutable +0 defeats CubeCL's literal FAdd(x,0) simplifier.
+    // CPU initializes all64 partials with +0 and adds its empty-tail +0 last.
+    let z = zero[0];
+    let mut accumulator = z;
+    let mut invalid = 0u32;
+    if u32::reinterpret(z) != 0u32 {
+        invalid |= 4u32;
+    }
+    for chunk in 0..columns / 64usize {
+        let column = chunk * 64usize + lane;
+        let wi = row * columns + column;
+        let xi = request * columns + column;
+        let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
+        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
+        if (wb & 0x7f80u32) == 0x7f80u32 {
+            invalid |= 1u32;
+        }
+        if (xb & 0x7f80u32) == 0x7f80u32 {
+            invalid |= 2u32;
+        }
+        accumulator = fma(
+            f32::reinterpret(wb << 16u32),
+            f32::reinterpret(xb << 16u32),
+            accumulator,
+        );
+    }
+    sums[lane] = accumulator;
+    flags[lane] = invalid;
+    sync_cube();
+    if lane < 32usize {
+        let collapsed = sums[lane] + sums[lane + 32usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane < 16usize {
+        let collapsed = sums[lane] + sums[lane + 16usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane < 8usize {
+        let collapsed = sums[lane] + sums[lane + 8usize];
+        sums[lane] = collapsed;
+    }
+    sync_cube();
+    if lane == 0usize {
+        let r0 = sums[0] + sums[4];
+        let r1 = sums[1] + sums[5];
+        let r2 = sums[2] + sums[6];
+        let r3 = sums[3] + sums[7];
+        let result = ((r0 + r2) + (r1 + r3)) + z;
+        let bits = u32::reinterpret(result);
+        let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
+        let mut combined = 0u32;
+        for at in 0..64usize {
+            combined |= flags[at];
+        }
+        if (bits & 0x7f800000u32) == 0x7f800000u32 {
+            combined |= 8u32;
+        }
+        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
+            combined |= 16u32;
+        }
+        accumulation[output] = result;
+        rounded[output] = f32::reinterpret(rounded_bits);
+        status[output] = combined;
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn native_shape_and_exact_allocation_accounting() {
+        for batch in [1, 2, 4, 8] {
+            let shape = Shape::new(512, 2048, batch).unwrap();
+            assert_eq!(
+                shape.owned_bytes().unwrap(),
+                2 * 512 * 2048 + 2 * batch * 2048 + 12 * batch * 512 + 4
+            );
+        }
+        for (r, c, b) in [
+            (0, 1024, 1),
+            (1, 0, 1),
+            (1, 63, 1),
+            (1, 65, 1),
+            (1, 8193, 1),
+            (1, 64, 3),
+            (16385, 64, 1),
+            (16384, 8192, 8),
+        ] {
+            assert!(Shape::new(r, c, b).is_err());
+        }
+    }
+    #[test]
+    fn explicit_input_bits_never_cast_or_admit_wrong_source_profile() {
+        let source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation().unwrap();
+        let mut raw = vec![0; 128];
+        for (at, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f].iter().enumerate() {
+            raw[2 * at..2 * at + 2].copy_from_slice(&bits.to_le_bytes());
+        }
+        validate_input(source, PROFILE, 1, 64, 1, &raw).unwrap();
+        raw[0..2].copy_from_slice(&0x7f80u16.to_le_bytes());
+        assert!(validate_input(source, PROFILE, 1, 64, 1, &raw).is_err());
+        assert!(
+            validate_input(
+                source,
+                ProgramNumericalProfile::GgufSourceV1,
+                1,
+                64,
+                1,
+                &[0; 128]
+            )
+            .is_err()
+        );
+    }
+}

--- V1/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -33,11 +33,11 @@
     }
     for request in 0..shape.batch {
         let teacher = request % 2;
-        for row in 0..shape.rows {
+        for (row, expected_value) in expected[teacher].iter().enumerate().take(shape.rows) {
             let at = request * shape.rows + row;
             let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
             let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
-            let expected_bits = expected[teacher][row].to_bits();
+            let expected_bits = expected_value.to_bits();
             let full_row = result.identity.rows.start + row;
             let reference = u32::from(u16::from_le_bytes([
                 torch[teacher][2 * full_row],
@@ -369,6 +369,8 @@
         device_cap,
     )?;
     let key = service.physical_key();
+    let owners_before =
+        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
     let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
     prepared.abandon_after_launch(0).await?;
     drop(window);
@@ -379,6 +381,8 @@
         || service.ensure_usable().is_ok()
         || witness.upgrade().is_none()
         || service.pending_events()? == 0
+        || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
+            <= owners_before
     {
         return Err(Error::State(
             "BF16 actual unknown work lost original leases/source/fence".into(),

--- ROOT_V2/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
+++ ROOT_V3/crates/zerogravity-gpu/src/cpu_bf16_projection/qualification.rs
@@ -22,7 +22,9 @@
     torch: &[Vec<u8>],
     shape: Shape,
 ) -> Result<usize> {
-    if result.batch != shape.batch
+    if expected.len() != 2
+        || expected.iter().any(|values| values.len() != shape.rows)
+        || result.batch != shape.batch
         || result.kernel_dispatches != 1
         || result.ready_outputs() != 3
         || result.completion_events() != 1

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v4-20261020/ROOT_V4_PYTHON_IMPORT.review.patch

Original UTF-8 bytes: 1716; SHA256 `e75e497bb3c8f5cda0a958c523d77c86abb25eb90c8a4f9d2f898c2879d76e5d`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- ROOT_V3/root-helper/proof.py
+++ ROOT_V4/root-helper/proof.py
@@ -68,9 +68,26 @@
     for row in manifest['oracle_sources']:
         pin(ORACLE/row['file'], row)
     require(str(ORACLE) not in sys.path, 'one explicit accepted observer namespace')
-    sys.path.insert(0, str(ORACLE))
-    validate = importlib.import_module('validate_replay')
-    runner = importlib.import_module('run_replay')
-    require(Path(validate.__file__).resolve().parent == ORACLE and Path(runner.__file__).resolve().parent == ORACLE,
-            'actual imported reviewed CPU observer path')
-    return validate, runner
+    names = ('owned_lifetime', 'contracts', 'prepare_inputs', 'run_replay', 'validate_replay')
+    prior_modules = {name: sys.modules.get(name) for name in names}
+    prior_path = sys.path[:]
+    try:
+        # Both independently reviewed harnesses use these basenames. Import the
+        # complete CPU observer closure in isolation, then restore this harness.
+        for name in names:
+            sys.modules.pop(name, None)
+        sys.path.insert(0, str(ORACLE))
+        validate = importlib.import_module('validate_replay')
+        runner = importlib.import_module('run_replay')
+        for name in names:
+            module = sys.modules[name]
+            require(Path(module.__file__).resolve() == ORACLE / (name + '.py'),
+                    'actual complete imported reviewed CPU observer namespace: ' + name)
+        return validate, runner
+    finally:
+        sys.path[:] = prior_path
+        for name, module in prior_modules.items():
+            if module is None:
+                sys.modules.pop(name, None)
+            else:
+                sys.modules[name] = module

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v4-20261020/WIRING.patch

Original UTF-8 bytes: 4015; SHA256 `561c8382154d40ecbc024e6b566ebb95f12b24f28256a1a7ef6eedfdb9a5a35c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/Cargo.toml
+++ b/crates/zerogravity-gpu/Cargo.toml
@@ -13,7 +13,7 @@
 cuda = ["cubecl/cuda", "cubecl/cuda-cpp"]
 hip = ["cubecl/hip", "cubecl/hip-cpp"]
 all = ["vulkan", "cuda", "hip"]
-qualification = ["dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
+qualification = ["zerogravity-artifacts/cache", "zerogravity-cpu/cache", "dep:zerogravity-cpu", "dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/cpu", "zerogravity-model-qwen35/text"]
 hardware-abort-probe = ["qualification"]
 portable-checkpoint = ["dep:zerogravity-model-qwen35", "zerogravity-model-qwen35/diagnostic-cpu-checkpoint"]
 diagnostic-gdn-trace = ["qualification", "zerogravity-model-qwen35/diagnostic-gdn-trace"]
--- a/crates/zerogravity-gpu/src/lib.rs
+++ b/crates/zerogravity-gpu/src/lib.rs
@@ -24,6 +24,8 @@
 mod batch_quant_qualification;
 #[cfg(any(feature = "qualification", test))]
 mod batch_recurrent;
+#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
+mod cpu_bf16_projection;
 mod device_activation;
 mod elementwise;
 mod embedding;
--- a/crates/zerogravity-gpu/src/qualification.rs
+++ b/crates/zerogravity-gpu/src/qualification.rs
@@ -949,3 +949,42 @@
         "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nUse rope-angles-cuda MODEL for physical angle differential qualification.\nUse rope-angles-model-cuda MODEL [PROVIDER_MiB DEVICE_MiB ROWS] for existing CPU/B1/native model gates.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
     );
 }
+
+/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_window(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    inputs: [Vec<u8>; 2],
+    public_outputs: [Vec<u8>; 2],
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_window(
+        device,
+        source,
+        descriptor,
+        rows,
+        inputs,
+        public_outputs,
+        device_cap,
+    )
+    .await
+}
+
+/// Separate fresh process only. Actual unknown completion must stay fenced.
+#[cfg(all(feature = "cuda", target_os = "linux"))]
+pub async fn qualify_cpu_profile_bf16_abandon(
+    device: cubecl::Device,
+    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
+    descriptor: zerogravity_core::accelerator::program::SourceTensor,
+    rows: std::ops::Range<usize>,
+    input: Vec<u8>,
+    device_cap: u64,
+) -> zerogravity_core::Result<()> {
+    crate::cpu_bf16_projection::qualification::qualify_abandon(
+        device, source, descriptor, rows, input, device_cap,
+    )
+    .await
+}
--- a/crates/zerogravity-gpu/src/resident_leases.rs
+++ b/crates/zerogravity-gpu/src/resident_leases.rs
@@ -814,6 +814,15 @@
     armed: bool,
 }
 impl ResidentInitialization<'_> {
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub(crate) fn retain_completion_owner(
+        &mut self,
+        owner: crate::physical_quarantine::CompletionOwner,
+    ) -> Result<()> {
+        // Attach first: refusal must retain owners of previously queued work.
+        self.completion_owners.push(owner);
+        self.service.ensure_usable()
+    }
     pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
         self.service.ensure_usable()?;
         let allocation_generation = issue(&self.service.next_allocation)?;

````

