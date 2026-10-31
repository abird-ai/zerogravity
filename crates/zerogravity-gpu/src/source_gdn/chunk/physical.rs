//! Actual-HIP operator diagnostic through original source/queue/flight owners.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{client::Client, server::Handle};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest},
};
#[path = "../../../../../tests/support/torch_cpu_gdn.rs"]
mod archive;
mod prepared;
mod source;

#[test]
#[ignore = "approved gfx1151 only; authenticated original chunk-GDN transport required"]
fn original_hf_chunk_gdn_on_hip() {
    futures_lite::future::block_on(run()).unwrap();
}
async fn run() -> Result<()> {
    if std::env::var("ZG_SOURCE_GDN_GFX1151").as_deref() != Ok("1")
        || std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/vendor")?.trim() != "0x1002"
        || std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/device")?.trim() != "0x1586"
    {
        return Err(Error::Unsupported(
            "explicit x2 gfx1151 chunk-GDN diagnostic required".into(),
        ));
    }
    let (anchor, source) = source::open()?;
    // Archive is streamed per case; expected data never becomes a device operand.
    let host = source.reserve_backend_workspace(256 * 1024 * 1024)?;
    let mut archive = archive::Archive::open()?;
    let cold = crate::hip::native::begin_qwen_qualification(
        cubecl::Device::Hip(cubecl::device::AmdDevice::new(0)),
        &source,
        anchor.clone(),
    )?;
    let mut catalogue = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for heads in archive.heads() {
        for kernel in prepared::kernels(cold.client(), heads) {
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
            "GDN original prepared module inventory".into(),
        ));
    }
    let baseline = service.resident_bytes()?;
    let mut words = 0;
    let mut failures = 0;
    for index in 0..archive.len() {
        let case = archive.case(index)?;
        native.require_live(service.client(), source.source_identity())?;
        let (count, bad) = execute(&service, &native, &case, index, false).await?;
        words += count;
        failures += bad;
        if service.resident_bytes()? != baseline {
            return Err(Error::State("GDN case retained physical buffers".into()));
        }
    }
    for (index, (input, value)) in [
        (0, f32::from_bits(0x7f7f0000)),
        (3, 2.0),
        (4, 0.125),
        (5, f32::INFINITY),
    ]
    .into_iter()
    .enumerate()
    {
        let mut case = archive.case(0)?;
        case.inputs[input][..4].copy_from_slice(&value.to_le_bytes());
        execute(&service, &native, &case, index, true).await?;
        if service.resident_bytes()? != baseline {
            return Err(Error::State("GDN refusal retained physical buffers".into()));
        }
    }
    archive.require_stable()?;
    drop(native);
    let retired = probe.observe()?;
    if retired.retained
        || retired.modules != 0
        || retired.code_bytes != 0
        || retired.host_bytes != 0
        || retired.device_bytes != 0
    {
        return Err(Error::State("GDN native final owner retained".into()));
    }
    service.ensure_usable()?;
    service.close_model_namespace()?;
    if service.ensure_usable().is_ok() {
        return Err(Error::State("GDN closed namespace remained usable".into()));
    }
    drop((service, archive, host, source, anchor));
    eprintln!(
        "SOURCE_GDN_HIP_RESULT cases=19 fields=380 words={words} mismatches={failures} native_modules={} native_retired=true namespace_closed=true refusal_controls=4 inputs_unchanged=true trained_model=false model_qualified=false",
        initial.modules
    );
    if failures != 0 {
        return Err(Error::State(
            "original chunk-GDN exact comparison failed".into(),
        ));
    }
    Ok(())
}

async fn execute(
    service: &Arc<ResidentService>,
    native: &Arc<crate::hip::native::PreparedSourceNative>,
    case: &archive::Case,
    index: usize,
    refusal: bool,
) -> Result<(usize, usize)> {
    let h = case.heads;
    let rows = case.rows as u32;
    let mut slot = service
        .queue()
        .submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 64 * 1024 * 1024,
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
    let sizes = archive::output_words(case.rows, h);
    let outputs = sizes[..19]
        .iter()
        .map(|words| init.allocate_empty(words * 4, BufferRole::Output))
        .collect::<Result<Vec<_>>>()?;
    let scratch = (0..2)
        .map(|_| init.allocate_empty(h * 64 * 4, BufferRole::Scratch))
        .collect::<Result<Vec<_>>>()?;
    use zerogravity_core::accelerator::program::standalone_exp::{
        MKL_EXHA_EXP2_64_F64, MKL_EXHA_SSE2_EXP2_128,
    };
    let normal = MKL_EXHA_SSE2_EXP2_128
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let rare = MKL_EXHA_EXP2_64_F64
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let tables = [normal, rare]
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
    // All handles come from this original service; writes are distinct allocations.
    let i = inputs.iter().map(ResidentLease::handle).collect::<Vec<_>>();
    let o = outputs
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    let s = scratch
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    let normal = tables[0].handle();
    let rare = tables[1].handle();
    let status_handle = status.handle();
    let client = service.client();
    let dim = CubeDim::new_1d(32);
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let arg = |handle: &Handle, n: usize, word_bytes: usize| -> BufferArg {
        assert_eq!(handle.service, client.service_id());
        let available = handle
            .size()
            .checked_sub(handle.offset_start.unwrap_or(0))
            .and_then(|bytes| bytes.checked_sub(handle.offset_end.unwrap_or(0)))
            .unwrap();
        assert!(available >= (n * word_bytes) as u64);
        // Original initialization owns each checked, non-overlapping allocation.
        unsafe { BufferArg::from_raw_parts(handle.clone(), n) }
    };
    // BufferArg lengths are elements, not bytes.
    let f = |handle: &Handle, n: usize| arg(handle, n, 4);
    let status_arg = || f(&status_handle, 1);
    unsafe {
        normalize::launch_unchecked(
            client,
            grid(h * 64),
            dim,
            f(&i[0], case.rows * h * 128),
            f(&i[1], case.rows * h * 128),
            f(&o[0], h * 8192),
            f(&o[1], h * 8192),
            status_arg(),
            rows,
            h,
        );
        beta_vectors::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(&o[1], h * 8192),
            f(&i[2], case.rows * h * 128),
            f(&i[3], case.rows * h),
            f(&o[2], h * 8192),
            f(&o[3], h * 8192),
            status_arg(),
            rows,
            h,
        );
        cumulative::launch_unchecked(
            client,
            grid(h),
            dim,
            f(&i[4], case.rows * h),
            arg(&rare, 128, 8),
            f(&o[4], h * 64),
            status_arg(),
            rows,
            h,
        );
        decays::launch_unchecked(
            client,
            grid(h * 4096),
            dim,
            f(&o[4], h * 64),
            f(&normal, 256),
            arg(&rare, 128, 8),
            f(&o[5], h * 4096),
            f(&s[0], h * 64),
            f(&s[1], h * 64),
            f(&o[13], h),
            status_arg(),
            h,
        );
        gram::launch_unchecked(
            client,
            grid(h * 4096),
            dim,
            f(&o[0], h * 8192),
            f(&o[1], h * 8192),
            f(&o[2], h * 8192),
            f(&o[5], h * 4096),
            f(&o[6], h * 4096),
            f(&o[7], h * 4096),
            status_arg(),
            h,
        );
        decay_vectors::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(&o[0], h * 8192),
            f(&o[1], h * 8192),
            f(&o[2], h * 8192),
            f(&s[0], h * 64),
            f(&s[1], h * 64),
            f(&o[8], h * 8192),
            f(&o[11], h * 8192),
            f(&o[12], h * 8192),
            status_arg(),
            h,
        );
        triangular::launch_unchecked(
            client,
            grid(h * 128),
            dim,
            f(&o[6], h * 4096),
            f(&o[3], h * 8192),
            f(&o[8], h * 8192),
            f(&o[9], h * 8192),
            f(&o[10], h * 8192),
            status_arg(),
            h,
        );
        prior_products::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(&o[10], h * 8192),
            f(&o[11], h * 8192),
            f(&i[5], h * 16384),
            f(&o[9], h * 8192),
            f(&o[14], h * 8192),
            f(&o[15], h * 8192),
            status_arg(),
            h,
        );
        core_product::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(&o[7], h * 4096),
            f(&o[14], h * 8192),
            f(&o[15], h * 8192),
            f(&o[16], h * 8192),
            f(&o[18], case.rows * h * 128),
            status_arg(),
            rows,
            h,
        );
        state_product::launch_unchecked(
            client,
            grid(h * 16384),
            dim,
            f(&o[12], h * 8192),
            f(&o[14], h * 8192),
            f(&i[5], h * 16384),
            f(&o[13], h),
            f(&o[17], h * 16384),
            status_arg(),
            h,
        );
    }
    // Known physical writes complete normally; refusals export only status.
    let mut produced = if refusal { Vec::new() } else { outputs.clone() };
    produced.push(status.clone());
    let readbacks = produced
        .iter()
        .chain(&inputs)
        .map(|l| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let mut result = (0usize, 0usize);
    let (ready,read)=flight.complete_with_readback(&produced,&readbacks,|read|{
        let count = if refusal { 1 } else { 20 };
        let expected_status = if refusal { NONFINITE_VALUE } else { 0u32 }.to_le_bytes();
        if read.len()!=count+6 || read[count-1].as_ref()!=expected_status {return Err(Error::State("GDN device numerical status".into()));}
        for (actual,expected) in read[count..].iter().zip(&case.inputs) {
            let actual: &[u8] = actual.as_ref();
            if actual != expected.as_slice() {return Err(Error::State("GDN mutated original input/prior state".into()));}
        }
        if refusal { return Ok(()); }
        for (stage,expected) in case.expected.iter().enumerate() {
            let actual=&read[if stage==19 {17}else{stage}];
            if actual.len()!=expected.len() {return Err(Error::State("GDN complete stage extent".into()));}
            let mut bad=0;
            for (word,(a,e)) in actual.chunks_exact(4).zip(expected.chunks_exact(4)).enumerate() {
                if a!=e {
                    if bad==0 {eprintln!("SOURCE_GDN_DIFFERENCE case={index} stage={} word={word} actual={:08x} expected={:08x}",archive::STAGES[stage],u32::from_le_bytes(a.try_into().unwrap()),u32::from_le_bytes(e.try_into().unwrap()));}
                    bad+=1;
                }
            }
            result.0+=expected.len()/4;result.1+=bad;
            eprintln!("SOURCE_GDN_STAGE case={index} stage={} words={} mismatches={bad}",archive::STAGES[stage],expected.len()/4);
        }
        Ok(())
    }).await?;
    slot.finish()?;
    if !Arc::ptr_eq(ready.service(), service)
        || ready.expected_root() != 0
        || ready.root_binding().is_some()
        || ready.event_ids().len() != 1
        || !ready
            .outputs()
            .iter()
            .map(ResidentLease::id)
            .eq(produced.iter().map(ResidentLease::id))
        || !outputs.iter().all(ResidentLease::is_initialized)
    {
        return Err(Error::State(
            "GDN original completion/output authority".into(),
        ));
    }
    if refusal {
        if ready.outputs().len() != 1
            || ready.outputs()[0].id() != status.id()
            || outputs
                .iter()
                .any(|output| ready.outputs().iter().any(|held| held.id() == output.id()))
        {
            return Err(Error::State("GDN refused math exported by Ready".into()));
        }
        eprintln!(
            "SOURCE_GDN_REFUSAL control={index} status={NONFINITE_VALUE} inputs_unchanged=true physical_writes_completed=true math_outputs_ready=false"
        );
    } else {
        eprintln!(
            "SOURCE_GDN_CASE case={index} rows={} heads={h} initialized={} launches=10 words={} mismatches={} inputs_unchanged=true",
            case.rows, case.initialized, result.0, result.1
        );
    }
    drop((ready, read));
    Ok(result)
}
