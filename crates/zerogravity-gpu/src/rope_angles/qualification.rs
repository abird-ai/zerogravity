//! Original physical angle kernels are the oracle; no fitted expectations.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::Device;
use std::sync::Arc;
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

const PAD: usize = 4;
const SENTINEL: u32 = 0xa5a55a5a;
const POSITIONS: [u32; 24] = [
    0, 1, 2, 3, 7, 31, 119, 120, 121, 127, 128, 255, 256, 511, 512, 1023, 4095, 4096, 8191, 8192,
    16383, 16384, 32767, 32768,
];

enum Input {
    Single(f32),
    Rows(Vec<u32>),
}

struct Fixture {
    label: String,
    input: Input,
    half: usize,
    theta: f32,
    cpu_source: bool,
}

impl Fixture {
    fn rows(&self) -> usize {
        match &self.input {
            Input::Single(_) => 1,
            Input::Rows(positions) => positions.len(),
        }
    }
    fn scalar(&self, row: usize) -> f32 {
        match &self.input {
            Input::Single(p) => *p,
            Input::Rows(p) => p[row] as f32,
        }
    }
}

fn fixtures(theta: f32) -> Vec<Fixture> {
    let mut cases = Vec::new();
    for position in POSITIONS {
        cases.push(Fixture {
            label: format!("trained-theta-B1-position-{position}"),
            input: Input::Single(position as f32),
            half: 32,
            theta,
            cpu_source: true,
        });
    }
    for rows in [2, 4, 8] {
        for chunk in POSITIONS.chunks(rows) {
            for reverse in [false, true] {
                let mut positions = chunk.to_vec();
                if reverse {
                    positions.reverse();
                }
                cases.push(Fixture {
                    label: format!("trained-theta-B{rows}-reverse-{reverse}-{}", chunk[0]),
                    input: Input::Rows(positions),
                    half: 32,
                    theta,
                    cpu_source: true,
                });
            }
        }
    }
    // Preserve the whole existing resident rotate2..64 bound, not only the
    // approved trained rotate64. These smaller widths are synthetic geometry.
    for half in 1..32 {
        cases.push(Fixture {
            label: format!("synthetic-B1-half-{half}"),
            input: Input::Single(32768.0),
            half,
            theta,
            cpu_source: true,
        });
        cases.push(Fixture {
            label: format!("synthetic-half-{half}"),
            input: Input::Rows(vec![0, 1, 511, 32768]),
            half,
            theta,
            cpu_source: true,
        });
    }
    for scale in [
        theta,
        1.0,
        0.5,
        f32::from_bits(0x3f7fffff),
        f32::MIN_POSITIVE,
        f32::from_bits(0x007fffff),
        f32::from_bits(1),
    ] {
        for seed in [
            0.0f32,
            -0.0,
            f32::from_bits(1),
            f32::from_bits(0x007fffff),
            f32::MIN_POSITIVE,
            f32::from_bits(0x3f800001),
        ] {
            cases.push(Fixture {
                label: format!(
                    "synthetic-seed-{:08x}-theta-{:08x}",
                    seed.to_bits(),
                    scale.to_bits()
                ),
                input: Input::Single(seed),
                half: 32,
                theta: scale,
                // Synthetic subnormal/zero policy is the old physical
                // kernel's contract, not a newly assumed host-FP policy.
                cpu_source: false,
            });
        }
    }
    cases
}

fn view(lease: &ResidentLease, start: usize, count: usize) -> Handle {
    lease
        .handle()
        .offset_start((start * 4) as u64)
        .offset_end((lease.bytes() - (start + count) * 4) as u64)
}

fn original_rows(client: &Client, fixture: &Fixture, angles: &Handle, positions: &Handle) -> usize {
    let rows = fixture.rows();
    // SAFETY: original full views and geometry were checked before submission;
    // the original attempt pins both input and output leases through Ready.
    unsafe {
        for stage in 0..fixture.half {
            crate::rope_kv::angle_step_rows::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(angles.clone(), rows * fixture.half),
                BufferArg::from_raw_parts(positions.clone(), rows),
                fixture.theta,
                fixture.half,
                rows,
                stage,
            );
        }
    }
    fixture.half
}

fn original_single(client: &Client, fixture: &Fixture, lease: &ResidentLease) -> usize {
    // SAFETY: each independent original scalar launch writes only its checked
    // row view. Same outer attempt retains every original allocation.
    unsafe {
        for row in 0..fixture.rows() {
            let handle = view(lease, PAD + row * fixture.half, fixture.half);
            for stage in 0..fixture.half {
                crate::rope_kv::angle_step::launch_unchecked(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(handle.clone(), fixture.half),
                    fixture.scalar(row),
                    fixture.theta,
                    stage,
                );
            }
        }
    }
    fixture.rows() * fixture.half
}

#[derive(Default)]
struct Counts {
    words: usize,
    legacy_row: usize,
    legacy_scalar: usize,
    candidate: usize,
}

fn compare_readbacks(fixture: &Fixture, read: &[cubecl::bytes::Bytes]) -> Result<()> {
    let count = fixture.rows() * fixture.half;
    let expected_bytes = (count + PAD * 2) * 4;
    let readbacks = if fixture.rows() == 1 { 2 } else { 3 };
    if read.len() != readbacks || read.iter().any(|r| r.len() != expected_bytes) {
        return Err(Error::State("ordered angle full readback coverage".into()));
    }
    for (index, bytes) in read.iter().enumerate() {
        if bytes[..] != read[0][..] {
            let word = bytes
                .chunks_exact(4)
                .zip(read[0].chunks_exact(4))
                .position(|(a, b)| a != b)
                .ok_or_else(|| Error::State("ordered angle unequal byte geometry".into()))?;
            return Err(Error::State(format!(
                "ordered angles {} readback{index} word{word} differs from original physical bits",
                fixture.label,
            )));
        }
        for word in bytes[..PAD * 4]
            .chunks_exact(4)
            .chain(bytes[(PAD + count) * 4..].chunks_exact(4))
        {
            if word != SENTINEL.to_ne_bytes() {
                return Err(Error::State(
                    "ordered angle wrote outside original angle view".into(),
                ));
            }
        }
    }
    for row in 0..fixture.rows() {
        let at = (PAD + row * fixture.half) * 4;
        let actual = u32::from_ne_bytes(read[0][at..at + 4].try_into().expect("stage0 word"));
        if actual != fixture.scalar(row).to_bits() {
            return Err(Error::State(
                "ordered angle stage0 did not preserve source bits".into(),
            ));
        }
    }
    if fixture.cpu_source {
        for row in 0..fixture.rows() {
            let mut angle = fixture.scalar(row);
            for stage in 0..fixture.half {
                let at = (PAD + row * fixture.half + stage) * 4;
                let actual =
                    u32::from_ne_bytes(read[0][at..at + 4].try_into().expect("angle word"));
                if actual != angle.to_bits() {
                    return Err(Error::State(format!(
                        "ordered angles {} row{row} stage{stage} independent source-FP32 sequence",
                        fixture.label,
                    )));
                }
                angle *= fixture.theta;
            }
        }
    }
    Ok(())
}

async fn compare_case(service: &Arc<ResidentService>, fixture: &Fixture) -> Result<Counts> {
    let rows = fixture.rows();
    let count = words(fixture.half, rows)?;
    let output_count = if rows == 1 { 2 } else { 3 };
    let output_bytes = (count + PAD * 2) * 4;
    let position_bytes = if rows == 1 { 0 } else { rows * 4 };
    let total = output_bytes * output_count + position_bytes;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: total as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    if total > service.remaining_capacity()? {
        return Err(Error::Limit(
            "ordered angle fixture physical capacity".into(),
        ));
    }
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let sentinels = vec![SENTINEL; count + PAD * 2];
    let mut outputs = Vec::with_capacity(output_count);
    for _ in 0..output_count {
        outputs.push(init.upload_bytes(u32::as_bytes(&sentinels), BufferRole::Scratch)?);
    }
    let positions = match &fixture.input {
        Input::Rows(p) => Some(init.upload_bytes(u32::as_bytes(p), BufferRole::Scratch)?),
        Input::Single(_) => None,
    };
    let ready = init.complete_sync().await?;
    if ready.outputs().len() != output_count + usize::from(positions.is_some()) {
        return Err(Error::State("ordered angle initialization coverage".into()));
    }
    drop(ready);
    let client = service.client();
    let candidate_handle = view(outputs.last().expect("candidate lease"), PAD, count);
    let position_handle = positions.as_ref().map(ResidentLease::handle);
    let candidate = match &fixture.input {
        Input::Single(p) => PreparedAngles::single_for_cuda(
            client,
            &candidate_handle,
            *p,
            fixture.theta,
            fixture.half,
        )?,
        Input::Rows(_) => PreparedAngles::rows_for_cuda(
            client,
            &candidate_handle,
            position_handle.as_ref().expect("positions"),
            fixture.theta,
            fixture.half,
            rows,
        )?,
    }
    .ok_or_else(|| Error::State("ordered angle CUDA selection returned None".into()))?;
    let short = candidate_handle.clone().offset_end(4);
    let short_result = match &fixture.input {
        Input::Single(p) => {
            PreparedAngles::single_for_cuda(client, &short, *p, fixture.theta, fixture.half)
        }
        Input::Rows(_) => PreparedAngles::rows_for_cuda(
            client,
            &short,
            position_handle.as_ref().expect("positions"),
            fixture.theta,
            fixture.half,
            rows,
        ),
    };
    if short_result.is_ok()
        || PreparedAngles::single_for_cuda(client, &candidate_handle, 0.0, fixture.theta, 0).is_ok()
        || PreparedAngles::single_for_cuda(client, &candidate_handle, 0.0, f32::NAN, fixture.half)
            .is_ok()
    {
        return Err(Error::State(
            "ordered angle accepted invalid geometry/short view".into(),
        ));
    }
    let mut foreign = candidate_handle.clone();
    foreign.service.service = std::any::TypeId::of::<()>();
    if PreparedAngles::single_for_cuda(client, &foreign, 0.0, fixture.theta, fixture.half).is_ok() {
        return Err(Error::State(
            "ordered angle accepted foreign metadata".into(),
        ));
    }
    if rows > 1
        && PreparedAngles::rows_for_cuda(
            client,
            &candidate_handle,
            &candidate_handle,
            fixture.theta,
            fixture.half,
            rows,
        )
        .is_ok()
    {
        return Err(Error::State(
            "ordered angle accepted aliased positions".into(),
        ));
    }
    let mut accesses = outputs
        .iter()
        .cloned()
        .map(|l| (l, DeviceAccess::Write))
        .collect::<Vec<_>>();
    if let Some(p) = &positions {
        accesses.push((p.clone(), DeviceAccess::Read));
    }
    let readbacks = outputs
        .iter()
        .map(|l| l.readback(0, output_bytes))
        .collect::<Result<Vec<_>>>()?;
    let uploaded = service.uploaded_value_bytes();
    let flight = service.submit(0, &accesses)?;
    let reference = if let Some(p) = &position_handle {
        original_rows(client, fixture, &view(&outputs[0], PAD, count), p)
    } else {
        original_single(client, fixture, &outputs[0])
    };
    let scalar_reference = if rows == 1 {
        0
    } else {
        original_single(client, fixture, &outputs[1])
    };
    let selected = candidate.launch(client)?;
    let (ready, bytes) = flight
        .complete_with_readback(&outputs, &readbacks, |read| {
            compare_readbacks(fixture, read)
        })
        .await?;
    slot.finish()?;
    if reference != fixture.half
        || selected != 1
        || (rows > 1 && scalar_reference != rows * fixture.half)
        || service.uploaded_value_bytes() != uploaded
    {
        return Err(Error::State(
            "ordered angle physical launch/upload accounting".into(),
        ));
    }
    let result = Counts {
        words: bytes.iter().map(|b| b.len() / 4).sum(),
        legacy_row: reference,
        legacy_scalar: scalar_reference,
        candidate: selected,
    };
    drop(ready);
    Ok(result)
}

pub async fn qualify_angles_with_plan(
    device: Device,
    model: Arc<dyn zerogravity_core::source::TensorSource>,
    plan: zerogravity_core::accelerator::program::ForwardPlan,
    source_path: &std::path::Path,
    device_cap: u64,
) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "physical angle qualification requires --release".into(),
        ));
    }
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "ordered angle qualifier requires CUDA".into(),
        ));
    }
    if plan.source() != model.source_identity() {
        return Err(Error::State(
            "angle source/recorded plan identity differs".into(),
        ));
    }
    let geometry = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.operation {
            PlanOperation::Rope { geometry, .. } => Some(*geometry),
            _ => None,
        })
        .collect::<Vec<_>>();
    if geometry.len() != 6
        || geometry
            .iter()
            .any(|g| g.rotate != 64 || g.theta_scale.to_bits() != geometry[0].theta_scale.to_bits())
    {
        return Err(Error::Unsupported(
            "angle qualifier requires approved six-layer rotate64 graph".into(),
        ));
    }
    let service = ResidentService::new(
        device,
        model.source_identity(),
        ResourceOwner::new("rope-angles", "original-physical-differential")?,
        64 * 1024,
        device_cap,
    )?;
    let key = service.physical_key();
    let cases = fixtures(geometry[0].theta_scale);
    let mut totals = Counts::default();
    for fixture in &cases {
        let actual = compare_case(&service, fixture).await?;
        totals.words += actual.words;
        totals.legacy_row += actual.legacy_row;
        totals.legacy_scalar += actual.legacy_scalar;
        totals.candidate += actual.candidate;
        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
            return Err(Error::State(
                "ordered angle retained lease/event after fixture".into(),
            ));
        }
    }
    drop(service);
    let final_budget = ResourceArbiter::process_shared().snapshot(&key)?;
    if final_budget.reserved != 0 || final_budget.fenced {
        return Err(Error::State("ordered angle final physical envelope".into()));
    }
    println!(
        "CUDA_ORDERED_ROPE_ANGLES_PASS source={} theta_bits={:08x} cases={} batches=1,2,4,8 half=1..32 trained_max_position=32768 compared_words={} old_row_or_single_dispatches={} independent_scalar_dispatches={} candidate_dispatches={} exact_signed_bits=1 guard_words=unchanged original_leases_and_Ready=1 warm_value_uploads=0 final_reserved=0 published_roots=0 fullmodel_performance=not_qualified",
        source_path.display(),
        geometry[0].theta_scale.to_bits(),
        cases.len(),
        totals.words,
        totals.legacy_row,
        totals.legacy_scalar,
        totals.candidate
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_scope_has_actual_positions_all_shapes_and_explicit_synthetic_edges() {
        let all = fixtures(0.5);
        for half in 1..=32 {
            assert!(all.iter().any(|f| f.half == half));
        }
        for rows in [1, 2, 4, 8] {
            assert!(all.iter().any(|f| f.rows() == rows));
        }
        for position in POSITIONS {
            assert!(all.iter().any(|f| matches!(&f.input, Input::Single(p)
                if f.cpu_source && p.to_bits() == (position as f32).to_bits())));
        }
        assert!(
            all.iter()
                .any(|f| !f.cpu_source && f.scalar(0).to_bits() == 0x80000000)
        );
        assert!(all.iter().any(|f| !f.cpu_source && f.theta.to_bits() == 1));
        assert!(
            all.iter()
                .all(|f| f.half <= 32 && words(f.half, f.rows()).is_ok())
        );
    }
}
