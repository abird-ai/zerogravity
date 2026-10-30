//! Ignored actual-HIP operator comparison, not model or serving qualification.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
#[path = "../../../../../tests/support/torch_cpu_attention.rs"]
mod archive;
mod source;

#[test]
#[ignore = "approved gfx1151 only; pinned original HF attention archive required"]
fn original_hf_attention_on_hip() {
    futures_lite::future::block_on(run()).unwrap();
}
async fn run() -> Result<()> {
    // Refuse the held local Renoir device before creating any native context.
    if std::env::var("ZG_SOURCE_ATTENTION_GFX1151").as_deref() != Ok("1")
        || std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/vendor")?.trim() != "0x1002"
        || std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/device")?.trim() != "0x1586"
    {
        return Err(Error::Unsupported(
            "explicit x2 gfx1151 operator diagnostic required".into(),
        ));
    }
    let (anchor, source) = source::open()?;
    let host = source.reserve_backend_workspace(64 * 1024 * 1024)?;
    let cases = archive::read()?;
    let cold = crate::hip::native::begin_qwen_qualification(
        cubecl::Device::Hip(cubecl::device::AmdDevice::new(0)),
        &source,
        anchor.clone(),
    )?;
    let mut catalogue = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for case in &cases {
        for kernel in kernels(cold.client(), case.geometry, case.position)? {
            if ids.insert(kernel.id()) {
                catalogue.push(kernel);
            }
        }
    }
    let service = cold.service();
    let native = cold
        .prepare(catalogue, source.reserve_workspace(1024 * 1024)?)
        .await?;
    let probe = native.final_owner_probe();
    let initial = probe.observe()?;
    if initial.modules != ids.len() || initial.modules == 0 {
        return Err(Error::State(
            "attention exact prepared module ownership".into(),
        ));
    }
    let baseline = service.resident_bytes()?;
    let mut words = 0;
    let mut failures = 0;
    for (index, case) in cases.iter().enumerate() {
        native.require_live(service.client(), source.source_identity())?;
        let (compared, mismatches) = execute(&service, &native, case, index).await?;
        words += compared;
        failures += mismatches;
        if service.resident_bytes()? != baseline {
            return Err(Error::State(
                "attention case retained physical buffers".into(),
            ));
        }
    }
    // Numerical failures still get independently checked known retirement.
    drop(native);
    let retired = probe.observe()?;
    if retired.retained || retired.modules != 0 {
        return Err(Error::State("attention native final owner retained".into()));
    }
    service.ensure_usable()?;
    service.close_model_namespace()?;
    if service.ensure_usable().is_ok() {
        return Err(Error::State(
            "attention closed namespace remained usable".into(),
        ));
    }
    drop((service, cases, host, source, anchor));
    eprintln!(
        "SOURCE_ATTENTION_HIP_RESULT cases=40 words={words} mismatches={failures} native_modules={} native_retired=true namespace_closed=true trained_model=false",
        initial.modules
    );
    if failures != 0 {
        return Err(Error::State(
            "original HF attention exact comparison failed".into(),
        ));
    }
    Ok(())
}
async fn execute(
    service: &Arc<ResidentService>,
    native: &Arc<crate::hip::native::PreparedSourceNative>,
    case: &archive::Case,
    index: usize,
) -> Result<(usize, usize)> {
    let mut slot = service
        .queue()
        .submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 4 * 1024 * 1024,
            }],
            None,
        )?
        .wait()
        .await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let inputs = case
        .inputs
        .iter()
        .map(|bytes| init.upload_bytes(bytes, BufferRole::Activation))
        .collect::<Result<Vec<_>>>()?;
    let outputs = case
        .expected
        .iter()
        .map(|bytes| init.allocate_empty(bytes.len(), BufferRole::Output))
        .collect::<Result<Vec<_>>>()?;
    let scratch = scratch_bytes(case.geometry, case.position)?
        .into_iter()
        .map(|bytes| init.allocate_empty(bytes, BufferRole::Scratch))
        .collect::<Result<Vec<_>>>()?;
    let frequencies = (0..32)
        .flat_map(|i| (1.0f32 / 10_000_000f32.powf((2 * i) as f32 / 64.0)).to_le_bytes())
        .collect::<Vec<_>>();
    let trig = crate::source_trig::TRIG_TABLES
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let scalar = crate::source_transcendentals::TABLES
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let tables = [frequencies, trig, scalar, vec![0; 4]]
        .iter()
        .map(|bytes| init.upload_bytes(bytes, BufferRole::Scratch))
        .collect::<Result<Vec<_>>>()?;
    let status = init.upload_zeroed(4, BufferRole::Scratch)?;
    drop(init.complete_sync().await?);
    let accesses = inputs
        .iter()
        .chain(&tables)
        .map(|l| (l.clone(), DeviceAccess::Read))
        .chain(
            outputs
                .iter()
                .chain(&scratch)
                .chain([&status])
                .map(|l| (l.clone(), DeviceAccess::Write)),
        )
        .collect::<Vec<_>>();
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(native.clone())?;
    let input = inputs.iter().map(ResidentLease::handle).collect::<Vec<_>>();
    let output = outputs
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    let scratch_handles = scratch
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    let launches = launch(
        service.client(),
        &scratch_handles,
        &input.iter().collect::<Vec<_>>(),
        output
            .iter()
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Error::State("nine attention outputs".into()))?,
        &tables[0].handle(),
        &tables[1].handle(),
        &tables[2].handle(),
        &status.handle(),
        &tables[3].handle(),
        case.geometry,
        case.position,
    )?;
    let mut produced = outputs.clone();
    produced.push(status.clone());
    let readbacks = produced
        .iter()
        .map(|l| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let mut result = (0, 0);
    let (ready, read) = flight.complete_with_readback(&produced, &readbacks, |read| {
        if read.len() != 10 || read[9].as_ref() != [0; 4] {
            return Err(Error::State("attention device extent/nonfinite status".into()));
        }
        let mut compared = 0;
        let mut mismatches = 0;
        for (stage, (actual, expected)) in read[..9].iter().zip(&case.expected).enumerate() {
            if actual.len() != expected.len() { return Err(Error::State("attention full field extent".into())); }
            let mut differences = 0;
            for (word, (a, e)) in actual.chunks_exact(4).zip(expected.chunks_exact(4)).enumerate() {
                if a != e {
                    if differences == 0 { eprintln!("SOURCE_ATTENTION_DIFFERENCE case={index} stage={stage} word={word} actual={:08x} expected={:08x}", u32::from_le_bytes(a.try_into().unwrap()), u32::from_le_bytes(e.try_into().unwrap())); }
                    differences += 1;
                }
            }
            compared += expected.len() / 4;
            mismatches += differences;
            eprintln!("SOURCE_ATTENTION_STAGE case={index} stage={stage} words={} mismatches={differences}", expected.len()/4);
        }
        result = (compared, mismatches);
        Ok(())
    }).await?;
    slot.finish()?;
    drop((ready, read));
    eprintln!(
        "SOURCE_ATTENTION_CASE case={index} tokens={} heads={} kv_heads={} launches={launches} words={} mismatches={}",
        case.position + 1,
        case.geometry.query_heads,
        case.geometry.kv_heads,
        result.0,
        result.1
    );
    Ok(result)
}
