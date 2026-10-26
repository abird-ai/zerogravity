//! Source-authenticated SYNTHETIC generic operator adversaries. Never a model.
//! Launches the exact candidate kernel on the original resident queue/leases.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{Device, bytes::Bytes, server::Handle};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

pub const ADVERSARY_WIDTHS: [usize; 6] = [64, 128, 1024, 2048, 5120, 8192];
pub const SYNTHETIC_ROWS: usize = 5;
const FIRST: usize = 1;
const LAST: usize = 4;
const OUTPUT_ROWS: usize = LAST - FIRST;
const GUARD_WORDS: usize = 4;
const GUARD_BYTES: usize = GUARD_WORDS * 4;
const PROVIDER: usize = 64 * 1024 * 1024;
const RAM: usize = 4 * 1024 * 1024;
const EXPECTED_REFUSAL: &str = "synthetic BF16 arithmetic intentionally refuses Ready";

pub fn bad_geometry_name(columns: usize) -> Result<String> {
    if ![65, 8193].contains(&columns) {
        return Err(malformed("explicit synthetic refused geometry"));
    }
    Ok(format!("synthetic.cpu64.cold-refuse-width{columns}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdversaryKind {
    SignedZero,
    EmptyTailZero,
    GradualPositive,
    GradualNegative,
    SubnormalProduct,
    SubnormalTieEven,
    SubnormalTieOdd,
    ReductionTree,
    RoundTieEven,
    RoundTieOdd,
    RoundNegativeEven,
    RoundNegativeOdd,
    FmaUnderflow,
    FmaOverflowCancellation,
    Fp32Overflow,
    Bf16Overflow,
    SourceNan,
    SourceInfinity,
}
impl AdversaryKind {
    pub fn all(columns: usize) -> Result<Vec<Self>> {
        if !ADVERSARY_WIDTHS.contains(&columns) {
            return Err(malformed("synthetic fixture closed width"));
        }
        let mut kinds = vec![
            Self::SignedZero,
            Self::EmptyTailZero,
            Self::GradualPositive,
            Self::GradualNegative,
            Self::SubnormalProduct,
            Self::SubnormalTieEven,
            Self::SubnormalTieOdd,
            Self::ReductionTree,
            Self::RoundTieEven,
            Self::RoundTieOdd,
            Self::RoundNegativeEven,
            Self::RoundNegativeOdd,
            Self::Fp32Overflow,
            Self::Bf16Overflow,
            Self::SourceNan,
            Self::SourceInfinity,
        ];
        if columns >= 128 {
            kinds.extend([Self::FmaUnderflow, Self::FmaOverflowCancellation]);
        }
        Ok(kinds)
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::SignedZero => "signed-zero-input",
            Self::EmptyTailZero => "mandatory-empty-tail-positive-zero",
            Self::GradualPositive => "positive-bf16-subnormal",
            Self::GradualNegative => "negative-bf16-subnormal",
            Self::SubnormalProduct => "f32-subnormal-product",
            Self::SubnormalTieEven => "bf16-subnormal-tie-even",
            Self::SubnormalTieOdd => "bf16-subnormal-tie-odd",
            Self::ReductionTree => "cpu64-horizontal-cancellation",
            Self::RoundTieEven => "bf16-normal-tie-even",
            Self::RoundTieOdd => "bf16-normal-tie-odd",
            Self::RoundNegativeEven => "bf16-negative-tie-even",
            Self::RoundNegativeOdd => "bf16-negative-tie-odd",
            Self::FmaUnderflow => "fused-subnormal-cancellation",
            Self::FmaOverflowCancellation => "fused-overflow-cancellation",
            Self::Fp32Overflow => "finite-input-f32-overflow",
            Self::Bf16Overflow => "finite-accumulation-bf16-overflow",
            Self::SourceNan => "nonfinite-source-nan-cold-refusal",
            Self::SourceInfinity => "nonfinite-source-inf-cold-refusal",
        }
    }
    pub fn source_name(self, columns: usize) -> String {
        format!("synthetic.cpu64.width{columns}.{}", self.name())
    }
    fn fault(self) -> bool {
        matches!(self, Self::Fp32Overflow | Self::Bf16Overflow)
    }
    fn nonfinite_source(self) -> bool {
        matches!(self, Self::SourceNan | Self::SourceInfinity)
    }

    /// Integer BF16 recipes only. ROOT seals these generic synthetic matrices;
    /// expected arithmetic is later obtained from the ORIGINAL CPU operator.
    pub fn fixture(self, columns: usize) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
        if !Self::all(columns)?.contains(&self) {
            return Err(malformed("synthetic case requires at least128 columns"));
        }
        let mut weights = vec![0u16; columns];
        let mut input = vec![0u16; columns];
        match self {
            Self::SignedZero => {
                weights.fill(0x3f80);
                input.fill(0x8000);
            }
            Self::EmptyTailZero => {
                weights.fill(0x0080);
                input.fill(0x8080);
            }
            Self::GradualPositive => {
                weights[0] = 0x0001;
                input[0] = 0x3f80;
            }
            Self::GradualNegative => {
                weights[0] = 0x8001;
                input[0] = 0x3f80;
            }
            Self::SubnormalProduct => {
                weights[0] = 0x0001;
                input[0] = 0x3b00;
            }
            Self::SubnormalTieEven => {
                weights[0] = 0x0001;
                input[0] = 0x3f00;
            }
            Self::SubnormalTieOdd => {
                weights[0] = 0x0003;
                input[0] = 0x3f00;
            }
            Self::ReductionTree => {
                weights[0] = 0x3f80;
                weights[2] = 0x3f80;
                weights[4] = 0x3f80;
                input[0] = 0x4b80;
                input[2] = 0x3f80;
                input[4] = 0xcb80;
            }
            Self::RoundTieEven
            | Self::RoundTieOdd
            | Self::RoundNegativeEven
            | Self::RoundNegativeOdd => {
                weights[0] = 0x3f80;
                weights[1] = 0x3f80;
                input[0] = match self {
                    Self::RoundTieEven => 0x3f80,
                    Self::RoundTieOdd => 0x3f81,
                    Self::RoundNegativeEven => 0xbf80,
                    _ => 0xbf81,
                };
                input[1] = if matches!(self, Self::RoundNegativeEven | Self::RoundNegativeOdd) {
                    0xbb80
                } else {
                    0x3b80
                };
            }
            Self::FmaUnderflow => {
                weights[0] = 0x0001;
                input[0] = 0x3780;
                weights[64] = 0x8001;
                input[64] = 0x3700;
            }
            Self::FmaOverflowCancellation => {
                weights[0] = 0x7f7f;
                input[0] = 0xbf80;
                weights[64] = 0x7f7f;
                input[64] = 0x4000;
            }
            Self::Fp32Overflow => {
                weights[0] = 0x7f7f;
                input[0] = 0x4000;
            }
            Self::Bf16Overflow => {
                weights[0] = 0x7f7f;
                input[0] = 0x3f80;
                weights[1] = 0x7b00;
                input[1] = 0x3f80;
            }
            Self::SourceNan => {
                weights[0] = 0x7fc1;
                input[0] = 0x3f80;
            }
            Self::SourceInfinity => {
                weights[0] = 0x7f80;
                input[0] = 0x3f80;
            }
        }
        let marker = columns - 1;
        let mut full = Vec::with_capacity(SYNTHETIC_ROWS * columns * 2);
        for row in 0..SYNTHETIC_ROWS {
            for (column, &weight) in weights.iter().enumerate() {
                let word = match row {
                    0 => 0x3e23u16 + (column % 16) as u16,
                    1 => weight,
                    2 => {
                        if column == marker {
                            0x3f80
                        } else {
                            0
                        }
                    }
                    3 => {
                        if column == marker {
                            0xbf80
                        } else {
                            0
                        }
                    }
                    _ => 0xbe4du16 + (column % 16) as u16,
                };
                full.extend_from_slice(&word.to_le_bytes());
            }
        }
        let inputs = (0..8)
            .map(|request| {
                let mut own = input.clone();
                // Member0 keeps the exact arithmetic witness. Other members have
                // distinct marker outputs, catching request/row broadcast aliases.
                if request > 0 {
                    own[marker] = 0x3f80 + request as u16;
                }
                own.into_iter().flat_map(u16::to_le_bytes).collect()
            })
            .collect();
        Ok((full, inputs))
    }
    fn witness(self) -> Option<(u32, u32)> {
        // Analytic source-free witnesses, not trained or reconstructed targets.
        Some(match self {
            Self::SignedZero | Self::EmptyTailZero | Self::FmaUnderflow => (0, 0),
            Self::GradualPositive => (0x0001_0000, 0x0001_0000),
            Self::GradualNegative => (0x8001_0000, 0x8001_0000),
            Self::SubnormalProduct => (0x0000_0080, 0),
            Self::SubnormalTieEven => (0x0000_8000, 0),
            Self::SubnormalTieOdd => (0x0001_8000, 0x0002_0000),
            Self::ReductionTree => (0x3f80_0000, 0x3f80_0000),
            Self::RoundTieEven => (0x3f80_8000, 0x3f80_0000),
            Self::RoundTieOdd => (0x3f81_8000, 0x3f82_0000),
            Self::RoundNegativeEven => (0xbf80_8000, 0xbf80_0000),
            Self::RoundNegativeOdd => (0xbf81_8000, 0xbf82_0000),
            Self::FmaOverflowCancellation => (0x7f7f_0000, 0x7f7f_0000),
            Self::Bf16Overflow => (0x7f7f_8000, 0x7f80_0000),
            _ => return None,
        })
    }
}

fn rne(bits: u32) -> u32 {
    bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000
}
fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
fn guard(binding: usize, side: usize, at: usize) -> u32 {
    0x4a10_0041 + (binding as u32) * 0x10000 + (side as u32) * 0x100 + at as u32
}
fn guarded(binding: usize, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(payload.len() + 2 * GUARD_BYTES);
    for at in 0..GUARD_WORDS {
        bytes.extend_from_slice(&guard(binding, 0, at).to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    for at in 0..GUARD_WORDS {
        bytes.extend_from_slice(&guard(binding, 1, at).to_le_bytes());
    }
    bytes
}
fn check_guards(binding: usize, actual: &[u8], payload_bytes: usize) -> Result<()> {
    if actual.len() != payload_bytes + 2 * GUARD_BYTES {
        return Err(malformed("guarded physical readback length"));
    }
    for at in 0..GUARD_WORDS {
        if word(&actual[4 * at..4 * at + 4]) != guard(binding, 0, at)
            || word(
                &actual[GUARD_BYTES + payload_bytes + 4 * at
                    ..GUARD_BYTES + payload_bytes + 4 * at + 4],
            ) != guard(binding, 1, at)
        {
            return Err(malformed("actual physical prefix/suffix canary changed"));
        }
    }
    Ok(())
}
fn descriptor(source: &VerifiedTensorSource, name: &str) -> Result<SourceTensor> {
    let matrix = source.matrix(name, MatrixAxes::OutputInput)?;
    Ok(SourceTensor {
        source: source.source_identity(),
        name: name.into(),
        kind: matrix.tensor().kind,
        rows: matrix.rows(),
        columns: matrix.columns(),
        axes: matrix.axes(),
    })
}

struct ProbeOwner {
    _window: Arc<TensorWindow>,
    initial: [Vec<u8>; 6],
    _ram: WindowWorkspace,
}
fn lease_view(
    service: &Arc<ResidentService>,
    lease: &ResidentLease,
    payload: usize,
) -> Result<Handle> {
    let full = payload
        .checked_add(2 * GUARD_BYTES)
        .ok_or_else(|| malformed("guarded lease byte overflow"))?;
    let handle = lease.handle();
    if !Arc::ptr_eq(lease.service(), service)
        || lease.source_identity() != service.source_identity()
        || lease.service_generation() != service.generation()
        || lease.bytes() != full
        || !lease.is_initialized()
        || handle.service != service.client().service_id()
        || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
        || handle.size_in_used() < full as u64
    {
        return Err(malformed("guarded original lease/service/whole range"));
    }
    let trim = handle.size_in_used() - GUARD_BYTES as u64 - payload as u64;
    let view = handle.offset_start(GUARD_BYTES as u64).offset_end(trim);
    if view.size_in_used() != payload as u64 {
        return Err(malformed("actual guarded subview byte range"));
    }
    Ok(view)
}

/// Direct qualifier of the SAME compiled kernel with guarded original leases.
/// This is not a second operator implementation or a production binding.
async fn probe(
    service: &Arc<ResidentService>,
    source: &VerifiedTensorSource,
    kind: AdversaryKind,
    columns: usize,
    batch: usize,
) -> Result<usize> {
    let ram = source.reserve_workspace(RAM)?; // BEFORE recipes, CPU arrays or copies.
    let (full, inputs) = kind.fixture(columns)?;
    let desc = descriptor(source, &kind.source_name(columns))?;
    if desc.rows != SYNTHETIC_ROWS || desc.columns != columns || desc.kind != QuantKind::Bf16 {
        return Err(malformed("synthetic full original descriptor differs"));
    }
    let matrix = source.matrix(&desc.name, MatrixAxes::OutputInput)?;
    let window = Arc::new(source.read_rows(&matrix, FIRST..LAST)?);
    if window.as_slice() != &full[FIRST * columns * 2..LAST * columns * 2] {
        return Err(malformed(
            "sealed synthetic words differ from declared integer recipe",
        ));
    }
    let bound = Bf16Window::bind(
        source.clone(),
        window.clone(),
        desc.clone(),
        FIRST..LAST,
        PROFILE,
    )?;
    bound.identity.require_window(&window)?;
    let raw = inputs[..batch].concat();
    let bound_input = Bf16Inputs::bind(source.clone(), PROFILE, OUTPUT_ROWS, columns, batch, &raw)?;
    if !source.same_issuer(&bound.source) || !source.same_issuer(&bound_input.source) {
        return Err(malformed("synthetic source issuer mismatch"));
    }
    let mut expected = Vec::new();
    for input in inputs.iter().take(batch) {
        let x = input
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect::<Vec<_>>();
        let cpu = zerogravity_cpu::source::window_projection::project_bf16_window(
            &desc,
            FIRST..LAST,
            &window,
            &x,
            None,
            None,
        );
        if kind == AdversaryKind::Fp32Overflow {
            if !matches!(cpu,Err(Error::Malformed(ref message)) if message=="nonfinite original window projection output")
            {
                return Err(malformed(
                    "CPU oracle did not refuse genuine finite-input FP32 overflow",
                ));
            }
            // Read the two independent finite marker rows through ORIGINAL CPU math.
            let marker = source.read_rows(&matrix, 2..LAST)?;
            let finite = zerogravity_cpu::source::window_projection::project_bf16_window(
                &desc,
                2..LAST,
                &marker,
                &x,
                None,
                None,
            )?;
            let mut row = vec![f32::INFINITY];
            row.extend_from_slice(finite.accumulations());
            expected.push(row);
        } else {
            expected.push(cpu?.accumulations().to_vec());
        }
    }
    if expected.len() != batch || expected.iter().any(|row| row.len() != OUTPUT_ROWS) {
        return Err(malformed("complete original CPU oracle output coverage"));
    }
    if let Some((raw, rounded)) = kind.witness()
        && (expected[0][0].to_bits() != raw || rne(raw) != rounded)
    {
        return Err(malformed(
            "original CPU64 analytic synthetic witness failed before CUDA",
        ));
    }
    let shape = bound_input.shape;
    let hardware = &service.client().properties().hardware;
    service.ensure_usable()?;
    service.pinned_cuda_queue()?;
    if service.source_identity() != source.source_identity()
        || !crate::quantization::activation_fused::is_cuda_service(service.client())
        || hardware.max_units_per_cube < 64
        || hardware.max_cube_dim.0 < 64
        || hardware.max_shared_memory_size < 512
        || hardware.max_bindings < 6
        || hardware.max_cube_count.0 < shape.outputs as u32
    {
        return Err(malformed(
            "guarded native original source/workgroup capability",
        ));
    }
    let payloads = [
        shape.weight_bytes,
        shape.input_bytes,
        4,
        shape.outputs * 4,
        shape.outputs * 4,
        shape.outputs * 4,
    ];
    let poison = vec![0x7fc0_1234u32.to_le_bytes(); shape.outputs].concat();
    let initial = [
        guarded(0, window.as_slice()),
        guarded(1, &raw),
        guarded(2, &0u32.to_le_bytes()),
        guarded(3, &poison),
        guarded(4, &poison),
        guarded(5, &poison),
    ];
    let owner = Arc::new(ProbeOwner {
        _window: window,
        initial,
        _ram: ram,
    });
    let retained = Arc::downgrade(&owner);
    let total = owner.initial.iter().map(Vec::len).sum::<usize>();
    if service.remaining_capacity()? < total {
        return Err(Error::Limit("guarded probe capacity".into()));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: total as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    if service.remaining_capacity()? < total {
        return Err(Error::Limit("guarded probe queued capacity".into()));
    }
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(owner.clone())?;
    let roles = [
        BufferRole::ImmutableWeights,
        BufferRole::Activation,
        BufferRole::ImmutableWeights,
        BufferRole::Output,
        BufferRole::Output,
        BufferRole::Scratch,
    ];
    let mut leases = Vec::new();
    for (raw, role) in owner.initial.iter().zip(roles) {
        leases.push(init.upload_bytes(raw, role)?);
    }
    let initialized = init.complete_sync().await?;
    slot.finish()?;
    drop(initialized);
    let mut views = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for (lease, payload) in leases.iter().zip(payloads) {
        if !ids.insert(lease.id()) {
            return Err(malformed("guarded lease alias"));
        }
        views.push(lease_view(service, lease, payload)?);
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: total as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    service.ensure_usable()?;
    let owners_before =
        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    slot.begin_submission()?;
    let accesses = leases
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (
                l.clone(),
                if i < 3 {
                    DeviceAccess::Read
                } else {
                    DeviceAccess::Write
                },
            )
        })
        .collect::<Vec<_>>();
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(owner.clone())?;
    let arg = |index: usize, n: usize| -> BufferArg {
        // SAFETY: six original whole leases are retained by the armed flight;
        // exact aligned interior views exclude both checked physical canaries.
        unsafe { BufferArg::from_raw_parts(views[index].clone(), n) }
    };
    // SAFETY: exact native shape, one64-lane cube/output, original kernel/barriers.
    unsafe {
        project_cpu_bf16::launch_unchecked(
            service.client(),
            CubeCount::Static(shape.outputs as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(0, shape.weight_bytes / 4),
            arg(1, shape.input_bytes / 4),
            arg(2, 1),
            arg(3, shape.outputs),
            arg(4, shape.outputs),
            arg(5, shape.outputs),
            shape.rows,
            shape.columns,
        );
    }
    let outputs = leases[3..].to_vec();
    let reads = leases
        .iter()
        .map(|lease| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let validated = AtomicBool::new(false);
    let validate = |actual: &[Bytes]| -> Result<()> {
        if actual.len() != 6 {
            return Err(malformed("guarded full physical readback coverage"));
        }
        for (index, (actual, payload)) in actual.iter().zip(payloads).enumerate() {
            check_guards(index, actual, payload)?;
            if index < 3 && &actual[..] != owner.initial[index].as_slice() {
                return Err(malformed(
                    "actual readonly BF16 source/input/zero bits changed",
                ));
            }
        }
        for (request, expected_row) in expected.iter().enumerate() {
            for (row, expected_value) in expected_row.iter().enumerate() {
                let at = GUARD_BYTES + (request * OUTPUT_ROWS + row) * 4;
                let accumulator = word(&actual[3][at..at + 4]);
                let rounded = word(&actual[4][at..at + 4]);
                let status = word(&actual[5][at..at + 4]);
                let cpu = expected_value.to_bits();
                let rounded_cpu = rne(cpu);
                let expected_status = if row == 0 && kind == AdversaryKind::Fp32Overflow {
                    24
                } else if row == 0 && kind == AdversaryKind::Bf16Overflow {
                    16
                } else {
                    0
                };
                if accumulator != cpu || rounded != rounded_cpu || status != expected_status {
                    return Err(Error::State(format!(
                        "synthetic BF16 {} C{columns} B{batch} request{request} row{row}: actual={accumulator:08x}/{rounded:08x}/{status:08x} expected={cpu:08x}/{rounded_cpu:08x}/{expected_status:08x}",
                        kind.name()
                    )));
                }
            }
        }
        validated.store(true, Ordering::Release);
        if kind.fault() {
            Err(Error::State(EXPECTED_REFUSAL.into()))
        } else {
            Ok(())
        }
    };
    let attempt = flight
        .complete_with_readback(&outputs, &reads, validate)
        .await;
    if kind.fault() {
        if !matches!(attempt,Err(Error::State(ref message)) if message==EXPECTED_REFUSAL)
            || !validated.load(Ordering::Acquire)
            || service.ensure_usable().is_ok()
            || !ResourceArbiter::process_shared()
                .snapshot(&service.physical_key())?
                .fenced
            || service.pending_events()? == 0
        {
            return Err(malformed(
                "actual nonfinite status did not refuse Ready into original quarantine",
            ));
        }
        drop(slot);
        drop(reads);
        drop(outputs);
        drop(views);
        drop(accesses);
        drop(leases);
        drop(owner);
        drop(bound);
        drop(bound_input);
        if retained.upgrade().is_none()
            || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
                <= owners_before
            || ResourceArbiter::process_shared()
                .snapshot(&service.physical_key())?
                .reserved
                < PROVIDER as u64
        {
            return Err(malformed(
                "nonfinite status lost original source/completion owner",
            ));
        }
        println!(
            "CPU_BF16_SYNTHETIC_STATUS_PASS case={} columns={columns} B={batch} actual_kernel_dispatches=1 canaries=exact readonly_bits=exact CPU_refusal_or_round_overflow=exact Ready=false original_source_and_quarantine=true final_zero=false",
            kind.name()
        );
        return Ok(shape.outputs);
    }
    let (ready, readback) = attempt?;
    slot.finish()?;
    if !validated.load(Ordering::Acquire)
        || ready.outputs().len() != 3
        || ready.event_ids().len() != 1
    {
        return Err(malformed("guarded native physical Ready receipt"));
    }
    drop(readback);
    drop(ready);
    drop(reads);
    drop(outputs);
    drop(views);
    drop(accesses);
    drop(leases);
    drop(owner);
    drop(bound);
    drop(bound_input);
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(malformed(
            "guarded native output/event leases did not retire",
        ));
    }
    println!(
        "CPU_BF16_SYNTHETIC_OPERATOR_PASS case={} columns={columns} B={batch} source_rows=5 range=1..4 actual_kernel_dispatches=1 accumulation_and_RNE=CPU64_exact status=0 canaries=exact readonly_bits=exact Ready=true trained_model=false",
        kind.name()
    );
    Ok(shape.outputs)
}

/// Full normal run or ONE failure probe in a fresh process. Never promote HF.
pub async fn qualify(
    device: Device,
    source: VerifiedTensorSource,
    device_cap: u64,
    fault: Option<(AdversaryKind, usize)>,
) -> Result<()> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "synthetic physical BF16 proof requires release CUDA".into(),
        ));
    }
    if fault.is_some_and(|(kind, batch)| !kind.fault() || ![1, 2, 4, 8].contains(&batch)) {
        return Err(malformed(
            "one explicit fresh-process nonfinite failure mode",
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("synthetic-cpu64-bf16", "generic-operator-only")?,
        PROVIDER,
        device_cap,
    )?;
    if ResourceArbiter::process_shared()
        .snapshot(&service.physical_key())?
        .reserved
        != PROVIDER as u64
    {
        return Err(malformed(
            "synthetic proof requires otherwise idle original physical budget",
        ));
    }
    let key = service.physical_key();
    if let Some((kind, batch)) = fault {
        probe(&service, &source, kind, 128, batch).await?;
        return Ok(()); // Deliberately fenced; no further device work/final-zero.
    }
    let mut launches = 0usize;
    let mut words = 0usize;
    let mut nonfinite_sources = 0usize;
    let mut nonfinite_inputs = 0usize;
    for columns in [65, 8193] {
        let desc = descriptor(&source, &bad_geometry_name(columns)?)?;
        if desc.rows != SYNTHETIC_ROWS || desc.columns != columns {
            return Err(malformed("actual cold geometry source shape"));
        }
        let matrix = source.matrix(&desc.name, MatrixAxes::OutputInput)?;
        let window = Arc::new(source.read_rows(&matrix, FIRST..LAST)?);
        if Bf16Window::bind(source.clone(), window, desc, FIRST..LAST, PROFILE).is_ok()
            || service.resident_bytes()? != 0
            || service.uploaded_value_bytes() != 0
        {
            return Err(malformed(
                "actual authenticated unsupported geometry crossed cold boundary",
            ));
        }
    }
    for columns in ADVERSARY_WIDTHS {
        let _ram = source.reserve_workspace(RAM)?;
        let uploads = service.uploaded_value_bytes();
        for bad in [0x7fc1u16, 0x7f80] {
            let mut raw = vec![0u8; columns * 2];
            raw[0..2].copy_from_slice(&bad.to_le_bytes());
            if Bf16Inputs::bind(source.clone(), PROFILE, OUTPUT_ROWS, columns, 1, &raw).is_ok()
                || service.resident_bytes()? != 0
                || service.uploaded_value_bytes() != uploads
            {
                return Err(malformed("nonfinite explicit input crossed cold boundary"));
            }
            nonfinite_inputs += 1;
        }
        if Bf16Inputs::bind(
            source.clone(),
            PROFILE,
            OUTPUT_ROWS,
            columns,
            3,
            &vec![0; columns * 6],
        )
        .is_ok()
        {
            return Err(malformed("unpadded unsupported B3 crossed cold boundary"));
        }
        for kind in AdversaryKind::all(columns)? {
            if kind.fault() {
                continue;
            }
            if kind.nonfinite_source() {
                let desc = descriptor(&source, &kind.source_name(columns))?;
                let matrix = source.matrix(&desc.name, MatrixAxes::OutputInput)?;
                let window = Arc::new(source.read_rows(&matrix, FIRST..LAST)?);
                let uploads = service.uploaded_value_bytes();
                if Bf16Window::bind(source.clone(), window, desc, FIRST..LAST, PROFILE).is_ok()
                    || service.resident_bytes()? != 0
                    || service.uploaded_value_bytes() != uploads
                {
                    return Err(malformed(
                        "authenticated nonfinite synthetic source crossed cold boundary",
                    ));
                }
                nonfinite_sources += 1;
                continue;
            }
            for batch in [1, 2, 4, 8] {
                words += probe(&service, &source, kind, columns, batch).await?;
                launches += 1;
            }
        }
    }
    if launches != 328 || words != 3690 || nonfinite_sources != 12 || nonfinite_inputs != 12 {
        return Err(malformed(
            "synthetic complete case/native-band/cold-refusal coverage",
        ));
    }
    drop(service);
    let final_row = ResourceArbiter::process_shared().snapshot(&key)?;
    if final_row.reserved != 0 || final_row.reservations != 0 || final_row.fenced {
        return Err(malformed(
            "synthetic normal original physical budget final zero",
        ));
    }
    println!(
        "CPU_BF16_SYNTHETIC_NORMAL_PASS actual_kernel_dispatches={launches} accumulation_words={words} rounded_words={words} source_nonfinite_cold_refusals=12 input_nonfinite_cold_refusals=12 actual_source_geometry_refusals=2 B3_refusals=6 original_GPU_reservations=0 trained_model=false HF_or_vLLM_profile_promotion=false"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_source_range_and_member_markers_remain_distinct() {
        for columns in ADVERSARY_WIDTHS {
            for kind in AdversaryKind::all(columns).unwrap() {
                let (full, inputs) = kind.fixture(columns).unwrap();
                assert_eq!(full.len(), SYNTHETIC_ROWS * columns * 2);
                assert_eq!(inputs.len(), 8);
                assert!(inputs.iter().all(|raw| raw.len() == columns * 2));
                assert_ne!(
                    &full[..columns * 2],
                    &full[(SYNTHETIC_ROWS - 1) * columns * 2..]
                );
                for pair in inputs.windows(2) {
                    assert_ne!(pair[0], pair[1]);
                }
                let source_word = u16::from_le_bytes([full[columns * 2], full[columns * 2 + 1]]);
                if kind.nonfinite_source() {
                    assert_eq!(source_word & 0x7f80, 0x7f80);
                } else {
                    assert!(finite_bf16(&full));
                }
                assert!(inputs.iter().all(|raw| finite_bf16(raw)));
            }
        }
    }
    #[test]
    fn every_physical_guard_word_mutation_is_detected() {
        for binding in 0..6 {
            let bytes = guarded(binding, &[0x52u8; 12]);
            check_guards(binding, &bytes, 12).unwrap();
            for at in (0..GUARD_BYTES).chain(GUARD_BYTES + 12..bytes.len()) {
                let mut changed = bytes.clone();
                changed[at] ^= 1;
                assert!(check_guards(binding, &changed, 12).is_err());
            }
            assert!(check_guards(binding, &bytes, 16).is_err());
            // Payload is checked against CPU/status separately, not by canaries.
            let mut changed = bytes.clone();
            changed[GUARD_BYTES] ^= 1;
            check_guards(binding, &changed, 12).unwrap();
        }
    }
    #[test]
    fn recipes_do_not_add_scalar_tails_or_fake_source_widths() {
        for columns in [0, 63, 65, 8193] {
            assert!(AdversaryKind::all(columns).is_err());
        }
        assert!(AdversaryKind::FmaUnderflow.fixture(64).is_err());
        assert!(AdversaryKind::FmaOverflowCancellation.fixture(64).is_err());
        assert_eq!(
            ADVERSARY_WIDTHS
                .iter()
                .map(|&c| AdversaryKind::all(c)
                    .unwrap()
                    .into_iter()
                    .filter(|kind| !kind.fault() && !kind.nonfinite_source())
                    .count()
                    * 4)
                .sum::<usize>(),
            328
        );
    }
}
