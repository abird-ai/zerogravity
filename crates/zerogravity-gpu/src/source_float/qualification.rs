//! Actual original shared rounding primitive under a caller's prepared service.
use super::*;
use crate::resident_leases::ResidentService;
use cubecl::{client::Client, prelude::BufferCompilationArg};
use cubecl_server::kernel::CubeKernel;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
use zerogravity_core::{Error, Result};

const N: usize = 16;
#[cube(launch_unchecked)]
fn round_source_probe(
    x: &[f64],
    a: &[u32],
    b: &[u32],
    sign: &[u32],
    cancel: &[u32],
    output: &mut [u32],
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let mut value = x[i];
        if cancel[i] != 0u32 {
            value = widen_source(f32::reinterpret(a[i])) + widen_source(f32::reinterpret(b[i]));
        }
        output[i] = u32::reinterpret(round_source(value, sign[i]));
    }
}
fn arg() -> BufferCompilationArg {
    BufferCompilationArg { inplace: None }
}
pub(crate) fn kernel(client: &Client) -> Box<dyn CubeKernel> {
    Box::new(round_source_probe::RoundSourceProbe::new(
        KernelSettings::new(
            CubeDim::new_1d(32).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        ),
        client.properties_shared(),
        client.target_properties_shared(),
        arg(),
        arg(),
        arg(),
        arg(),
        arg(),
        arg(),
    ))
}
struct Cases {
    x: [f64; N],
    a: [u32; N],
    b: [u32; N],
    sign: [u32; N],
    cancel: [u32; N],
    expected: [u32; N],
}
fn cases() -> Cases {
    let unit = f64::from(f32::from_bits(1));
    let x = [
        -unit,
        unit,
        -unit * 0.25,
        unit * 0.25,
        -unit * 0.5,
        unit * 0.5,
        -unit * 1.5,
        unit * 1.5,
        -unit * 2.5,
        unit * 2.5,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let mut a = [0; N];
    let mut b = [0; N];
    let mut cancel = [0; N];
    a[12..].copy_from_slice(&[0x007fffff, 0x00800000, 0x00800000, 0x80000000]);
    b[12..].copy_from_slice(&[0x80800000, 0x807fffff, 0x80800000, 0x80000000]);
    cancel[12..].fill(1);
    let sign = [
        0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0x80000000, 0,
        0, 0, 0, 0x80000000,
    ];
    let expected = [
        0x80000001, 1, 0x80000000, 0, 0x80000000, 0, 0x80000002, 2, 0x80000002, 2, 0x80000000, 0,
        0x80000001, 1, 0, 0x80000000,
    ];
    Cases {
        x,
        a,
        b,
        sign,
        cancel,
        expected,
    }
}
/// Caller has sealed this exact generated constructor in its genuine original
/// cache. Source admits host copies; same ledger/queue owns every physical byte.
pub(crate) async fn run(
    service: &Arc<ResidentService>,
    source: &VerifiedTensorSource,
) -> Result<()> {
    let _host = source.reserve_backend_workspace(64 * 1024)?;
    let Cases {
        x,
        a,
        b,
        sign,
        cancel,
        expected,
    } = cases();
    let required = N * (8 + 5 * 4);
    let mut slot = service
        .queue()
        .submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: required as u64,
            }],
            None,
        )?
        .wait()
        .await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let x = init.upload_bytes(f64::as_bytes(&x), BufferRole::Activation)?;
    let a = init.upload_bytes(u32::as_bytes(&a), BufferRole::Activation)?;
    let b = init.upload_bytes(u32::as_bytes(&b), BufferRole::Activation)?;
    let sign = init.upload_bytes(u32::as_bytes(&sign), BufferRole::Activation)?;
    let cancel = init.upload_bytes(u32::as_bytes(&cancel), BufferRole::Activation)?;
    let output = init.allocate_empty(N * 4, BufferRole::Output)?;
    let initialized = init.complete_sync().await?;
    drop(initialized);
    let accesses = [
        (&x, DeviceAccess::Read),
        (&a, DeviceAccess::Read),
        (&b, DeviceAccess::Read),
        (&sign, DeviceAccess::Read),
        (&cancel, DeviceAccess::Read),
        (&output, DeviceAccess::Write),
    ]
    .into_iter()
    .map(|(lease, access)| (lease.clone(), access))
    .collect::<Vec<_>>();
    let flight = service.submit(0, &accesses)?;
    // SAFETY: all16 input/output entries admitted and initialized; original
    // submission retains every lease until actual completion or quarantine.
    unsafe {
        round_source_probe::launch_unchecked(
            service.client(),
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(x.handle(), N),
            BufferArg::from_raw_parts(a.handle(), N),
            BufferArg::from_raw_parts(b.handle(), N),
            BufferArg::from_raw_parts(sign.handle(), N),
            BufferArg::from_raw_parts(cancel.handle(), N),
            BufferArg::from_raw_parts(output.handle(), N),
        );
    }
    let (ready, read) = flight
        .complete_with_readback(
            std::slice::from_ref(&output),
            &[output.readback(0, N * 4)?],
            |read| {
                if read.len() != 1 || read[0].len() != N * 4 {
                    return Err(Error::State("shared source-round physical extent".into()));
                }
                for (i, (word, expected)) in read[0].chunks_exact(4).zip(expected).enumerate() {
                    let actual = u32::from_ne_bytes(word.try_into().expect("checked word"));
                    if actual != expected {
                        return Err(Error::State(format!(
                            "shared source-round word{i}: {actual:08x} != {expected:08x}"
                        )));
                    }
                }
                Ok(())
            },
        )
        .await?;
    slot.finish()?;
    drop((ready, read, accesses, x, a, b, sign, cancel, output));
    eprintln!(
        "SHARED_SOURCE_ROUND_PHYSICAL_PASS cases={N} nonzero_sign=actual_result zero_sign=explicit ties=nearest_even cancellation=actual_device shared_primitive=original prepared_cache=sealed roots=0"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_shared_body_handles_result_sign_and_cancellation() {
        // Execute unchanged shared CubeCL scalar bodies; substitute only the
        // typed cast/bitcast intrinsics. Physical expansion remains separate.
        let source = include_str!("../source_float.rs");
        let (body, _) = source
            .split_once("/// Correct a physically stored quotient")
            .unwrap();
        assert_eq!(body.matches("#[cube]").count(), 2);
        let body = body
            .replace("//!", "//")
            .replace("use cubecl::prelude::*;", "")
            .replace("#[cube]", "")
            .replace("f32::reinterpret", "f32::from_bits")
            .replace("u32::reinterpret", "f32::to_bits")
            .replace("f64::cast_from", "to_f64")
            .replace("f32::cast_from", "to_f32")
            .replace("u32::cast_from", "to_u32");
        assert!(!body.contains("cubecl") && !body.contains("::cast_from"));
        let c = cases();
        let cases = (0..N)
            .map(|i| {
                format!(
                    "(f64::from_bits({}),{},{},{},{},{}),",
                    c.x[i].to_bits(),
                    c.a[i],
                    c.b[i],
                    c.sign[i],
                    c.cancel[i],
                    c.expected[i]
                )
            })
            .collect::<String>();
        let code = format!(
            "fn to_f64<T:Into<f64>>(x:T)->f64{{x.into()}}fn to_f32(x:f64)->f32{{x as f32}}fn to_u32(x:f64)->u32{{x as u32}}\n{body}\nfn main(){{for(x,a,b,sign,cancel,expected)in[{cases}]{{let x=if cancel!=0{{widen_source(f32::from_bits(a))+widen_source(f32::from_bits(b))}}else{{x}};let actual=round_source(x,sign).to_bits();assert_eq!(actual,expected,\"x={{x:?}} sign={{sign:08x}}\");let ieee=if x==0.0{{sign}}else{{(x as f32).to_bits()}};assert_eq!(actual,ieee);}}}}"
        );
        let dir = std::env::temp_dir().join(format!(
            "zg-source-round-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let src = dir.join("round.rs");
        let bin = dir.join("round");
        std::fs::write(&src, code).unwrap();
        let compiled = std::process::Command::new("rustc")
            .args(["--edition=2024", "-O"])
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let output = std::process::Command::new(&bin).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
