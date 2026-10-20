//! ROOT-only dual CPU-reference / authentic public-output physical proof.
//! External caller must authenticate the original source and actual archived
//! teacher inputs/outputs. Synthetic words alone cannot qualify trained math.
use super::*;
use crate::resident_leases::ResidentService;
use cubecl::Device;
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

const PROVIDER_CAP: usize = 64 * 1024 * 1024;

fn bytes_word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
fn check_exact(
    result: &ProjectionReadback,
    expected: &[Vec<f32>],
    torch: &[Vec<u8>],
    shape: Shape,
) -> Result<usize> {
    if expected.len() != 2
        || expected.iter().any(|values| values.len() != shape.rows)
        || result.batch != shape.batch
        || result.kernel_dispatches != 1
        || result.ready_outputs() != 3
        || result.completion_events() != 1
        || result.accumulation().len() != shape.outputs * 4
        || result.rounded().len() != shape.outputs * 4
    {
        return Err(malformed("physical single-dispatch/Ready/output coverage"));
    }
    for request in 0..shape.batch {
        let teacher = request % 2;
        for (row, expected_value) in expected[teacher].iter().enumerate().take(shape.rows) {
            let at = request * shape.rows + row;
            let actual = bytes_word(&result.accumulation()[4 * at..4 * at + 4]);
            let rounded = bytes_word(&result.rounded()[4 * at..4 * at + 4]);
            let expected_bits = expected_value.to_bits();
            let full_row = result.identity.rows.start + row;
            let reference = u32::from(u16::from_le_bytes([
                torch[teacher][2 * full_row],
                torch[teacher][2 * full_row + 1],
            ])) << 16;
            if actual != expected_bits || rounded != reference {
                return Err(Error::State(format!(
                    "CPU-profile BF16 CUDA B{} request{request} row{full_row}: accumulation {actual:08x}!={expected_bits:08x}, final {rounded:08x}!={reference:08x}",
                    shape.batch
                )));
            }
        }
    }
    Ok(shape.outputs)
}

/// Test-only public entry receives actual source and ROOT-authenticated raw
/// records; no model factory/provider fallback or per-token foreign runtime.
pub async fn qualify_window(
    device: Device,
    source: VerifiedTensorSource,
    descriptor: SourceTensor,
    rows: Range<usize>,
    inputs: [Vec<u8>; 2],
    public_outputs: [Vec<u8>; 2],
    device_cap: u64,
) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "physical BF16 projection requires release".into(),
        ));
    }
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "physical BF16 projection requires CUDA".into(),
        ));
    }
    Shape::new(rows.len(), descriptor.columns, 1)?;
    if descriptor.rows == 0 || descriptor.rows > 1_048_576 {
        return Err(malformed("bounded original full row count"));
    }
    let _host = source.reserve_workspace(2 * 1024 * 1024)?;
    if descriptor.source != source.source_identity()
        || inputs
            .iter()
            .any(|raw| raw.len() != descriptor.columns * 2 || !finite_bf16(raw))
        || public_outputs
            .iter()
            .any(|raw| raw.len() != descriptor.rows * 2 || !finite_bf16(raw))
    {
        return Err(malformed(
            "complete actual original two-teacher input/public output geometry",
        ));
    }
    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
    Bf16Window::bind(
        source.clone(),
        window.clone(),
        descriptor.clone(),
        rows.clone(),
        PROFILE,
    )?;
    // Real sealed original window, deliberately wrong full-descriptor tags.
    // None may become a partial/fake source or reach CUDA initialization.
    for changed in 0..4 {
        let mut wrong = descriptor.clone();
        match changed {
            0 => {
                wrong.source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation()?
            }
            1 => wrong.axes = MatrixAxes::InputOutput,
            2 => wrong.rows = wrong.rows.saturating_add(1),
            _ => wrong.name.push_str(".foreign"),
        }
        if Bf16Window::bind(source.clone(), window.clone(), wrong, rows.clone(), PROFILE).is_ok() {
            return Err(malformed("wrong original full descriptor was accepted"));
        }
    }
    let mut expected = Vec::new();
    for raw in &inputs {
        let values = raw
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect::<Vec<_>>();
        let cpu = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
            &descriptor,
            rows.clone(),
            &window,
            &values,
            None,
            None,
        )?;
        if cpu.identity().descriptor() != &descriptor || cpu.identity().row_range() != rows {
            return Err(malformed("original CPU oracle full identity/range"));
        }
        expected.push(cpu.accumulations().to_vec());
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("original-cpu-bf16-reference", &descriptor.name)?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != PROVIDER_CAP as u64 {
        return Err(Error::State(
            "BF16 diagnostic requires otherwise idle physical budget".into(),
        ));
    }
    // Real original physical allocation, leaving four bytes less than required.
    // The new binding must refuse BEFORE H2D, allocation or model dispatch.
    let shape = Shape::new(rows.len(), descriptor.columns, 8)?;
    let need = shape.owned_bytes()?;
    let filler_bytes = PROVIDER_CAP
        .checked_sub(need)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| malformed("pressure fixture cap"))?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: filler_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(99)?;
    let filler = init.allocate_empty(filler_bytes, BufferRole::Scratch)?;
    let initialized = init.complete_sync().await?;
    slot.finish()?;
    drop(initialized);
    let raw = inputs[0].repeat(8);
    let uploads = service.uploaded_value_bytes();
    let pressure = PreparedProjection::prepare(
        service.clone(),
        Bf16Window::bind(
            source.clone(),
            window.clone(),
            descriptor.clone(),
            rows.clone(),
            PROFILE,
        )?,
        Bf16Inputs::bind(
            source.clone(),
            PROFILE,
            rows.len(),
            descriptor.columns,
            8,
            &raw,
        )?,
        99,
        None,
    )
    .await;
    if !matches!(pressure, Err(Error::Limit(ref message)) if message == "private BF16 operator capacity before launch")
        || service.resident_bytes()? != filler_bytes
        || service.uploaded_value_bytes() != uploads
        || service.pending_events()? != 0
    {
        return Err(Error::State(
            "BF16 real capacity refusal crossed prelaunch boundary".into(),
        ));
    }
    drop(filler);
    if service.resident_bytes()? != 0 {
        return Err(Error::State(
            "BF16 pressure allocation did not retire".into(),
        ));
    }
    let mut compared = 0usize;
    let mut actual_dispatches = 0usize;
    for (case, batch) in [1usize, 1usize, 2usize, 4usize, 8usize]
        .into_iter()
        .enumerate()
    {
        let shape = Shape::new(rows.len(), descriptor.columns, batch)?;
        let selected = if case == 1 { 1usize } else { 0usize };
        let raw = (0..batch)
            .flat_map(|request| inputs[(request + selected) % 2].iter().copied())
            .collect::<Vec<_>>();
        let bind_input = || {
            Bf16Inputs::bind(
                source.clone(),
                PROFILE,
                rows.len(),
                descriptor.columns,
                batch,
                &raw,
            )
        };
        let source_window = || {
            Bf16Window::bind(
                source.clone(),
                window.clone(),
                descriptor.clone(),
                rows.clone(),
                PROFILE,
            )
        };
        // Queued/prelaunch cancellation cannot create a lease, H2D or kernel.
        let uploads = service.uploaded_value_bytes();
        if PreparedProjection::prepare(
            service.clone(),
            source_window()?,
            bind_input()?,
            case as u64,
            Some(&AtomicBool::new(true)),
        )
        .await
        .is_ok()
            || service.resident_bytes()? != 0
            || service.uploaded_value_bytes() != uploads
        {
            return Err(Error::State(
                "BF16 pre-cancel performed physical work".into(),
            ));
        }
        // Cold initialized leases may also cancel before their first kernel.
        let prepared = PreparedProjection::prepare(
            service.clone(),
            source_window()?,
            bind_input()?,
            case as u64,
            None,
        )
        .await?;
        let uploads = service.uploaded_value_bytes();
        if prepared
            .execute(case as u64, Some(&AtomicBool::new(true)))
            .await
            .is_ok()
            || service.resident_bytes()? != 0
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != uploads
        {
            return Err(Error::State(
                "BF16 initialized prelaunch cancellation leaked or submitted work".into(),
            ));
        }
        let prepared = PreparedProjection::prepare(
            service.clone(),
            source_window()?,
            bind_input()?,
            case as u64,
            None,
        )
        .await?;
        let result = prepared.execute(case as u64, None).await?;
        result.identity.require_window(&window)?;
        if result.identity.descriptor() != &descriptor || result.identity.rows() != rows {
            return Err(malformed("actual GPU full descriptor/range"));
        }
        // Second independent B1 uses actual teacher353, never padded/relabelled.
        if selected == 0 {
            compared += check_exact(&result, &expected, &public_outputs, shape)?;
        } else {
            let expected_swapped = [expected[1].clone(), expected[0].clone()];
            let public_swapped = [public_outputs[1].clone(), public_outputs[0].clone()];
            compared += check_exact(&result, &expected_swapped, &public_swapped, shape)?;
        }
        actual_dispatches += result.kernel_dispatches;
        drop(result);
        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
            return Err(Error::State(
                "BF16 physical outputs/events failed retirement".into(),
            ));
        }
        println!(
            "CPU_PROFILE_BF16_CUDA_OPERATOR_PASS name={} full_rows={} range={}..{} columns={} B={batch} accumulation_bits=exact archived_public_BF16_bits=exact actual_native_dispatches=1 physical_pressure_and_prelaunch_cancel=true state_or_HF_profile_promotion=false",
            descriptor.name, descriptor.rows, rows.start, rows.end, descriptor.columns
        );
    }
    if compared != 16 * rows.len() || actual_dispatches != 5 {
        return Err(Error::State(
            "BF16 physical all-row dual reference coverage incomplete".into(),
        ));
    }
    drop(service);
    let final_row = ResourceArbiter::process_shared().snapshot(&key)?;
    if final_row.reserved != 0 || final_row.reservations != 0 || final_row.fenced {
        return Err(Error::State(
            "BF16 original GPU budget did not finish at zero".into(),
        ));
    }
    Ok(())
}

/// Separate fresh process only: ACTUAL queued kernel, no trusted completion.
/// Success here means the original uncertainty quarantine held, never Ready.
pub async fn qualify_abandon(
    device: Device,
    source: VerifiedTensorSource,
    descriptor: SourceTensor,
    rows: Range<usize>,
    input: Vec<u8>,
    device_cap: u64,
) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "physical BF16 abort requires release".into(),
        ));
    }
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "physical BF16 abort requires CUDA".into(),
        ));
    }
    let matrix = source.matrix(&descriptor.name, MatrixAxes::OutputInput)?;
    let window = Arc::new(source.read_rows(&matrix, rows.clone())?);
    let witness = Arc::downgrade(&window);
    let inputs = Bf16Inputs::bind(
        source.clone(),
        PROFILE,
        rows.len(),
        descriptor.columns,
        1,
        &input,
    )?;
    let original = Bf16Window::bind(source.clone(), window.clone(), descriptor, rows, PROFILE)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("original-cpu-bf16-reference", "physical-abandon")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    let owners_before =
        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    let prepared = PreparedProjection::prepare(service.clone(), original, inputs, 0, None).await?;
    prepared.abandon_after_launch(0).await?;
    drop(window);
    drop(source);
    let actual = ResourceArbiter::process_shared().snapshot(&key)?;
    if !actual.fenced
        || actual.reserved < PROVIDER_CAP as u64
        || service.ensure_usable().is_ok()
        || witness.upgrade().is_none()
        || service.pending_events()? == 0
        || crate::physical_quarantine::quarantined_completion_owners_for(service.client())?
            <= owners_before
    {
        return Err(Error::State(
            "BF16 actual unknown work lost original leases/source/fence".into(),
        ));
    }
    println!(
        "CPU_PROFILE_BF16_CUDA_ABANDON_PASS actual_kernel_submitted=true trusted_completion=false Ready=false original_source_and_leases_quarantined=true final_zero_not_claimed=true"
    );
    Ok(())
}
