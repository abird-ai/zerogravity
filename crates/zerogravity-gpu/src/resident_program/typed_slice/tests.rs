use super::*;
use sha2::{Digest, Sha256};
use std::{fs, fs::File, path::PathBuf};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};
use zerogravity_core::{
    accelerator::program::{
        EmbeddingContract, MathematicalBackend, MatrixAxes, OrdinaryRmsContract, PlanBuilder,
        ProjectionContract, SourceTensor,
    },
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner},
};
#[path = "reference.rs"]
mod reference;

struct Fixture {
    path: PathBuf,
    index: Vec<u8>,
    digest: String,
    bytes: u64,
    admission: TransferAdmission,
    _metadata: zerogravity_core::resource_arbiter::GlobalResourceLease,
}
impl Fixture {
    fn new(width: usize, rows: usize, arbiter: ResourceArbiter) -> Result<Self> {
        let resource = PhysicalDeviceKey::new("host", "system-ram")?;
        arbiter.register_matching(resource.clone(), 128 * 1024 * 1024)?;
        let admission = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("portable-bf16-arithmetic", "bounded-source-fixture")?,
        };
        let ticket = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[zerogravity_core::resource_arbiter::AllocationRequest {
                kind: zerogravity_core::resource_arbiter::AllocationClass::Scratch,
                bytes: 2 * 1024 * 1024,
            }],
        )?;
        let lengths = [17 * width * 2, width * 2, rows * width * 2];
        if lengths.iter().sum::<usize>() > 1024 * 1024 {
            return Err(Error::Limit("portable arithmetic fixture bound".into()));
        }
        let header = format!(
            "{{\"embedding\":{{\"dtype\":\"BF16\",\"shape\":[17,{width}],\"data_offsets\":[0,{}]}},\"gamma\":{{\"dtype\":\"BF16\",\"shape\":[{width}],\"data_offsets\":[{},{}]}},\"projection\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{width}],\"data_offsets\":[{},{}]}}}}",
            lengths[0],
            lengths[0],
            lengths[0] + lengths[1],
            lengths[0] + lengths[1],
            lengths.iter().sum::<usize>()
        );
        let mut payload = Vec::with_capacity(8 + header.len() + lengths.iter().sum::<usize>());
        payload.extend_from_slice(&(header.len() as u64).to_le_bytes());
        payload.extend_from_slice(header.as_bytes());
        for (member, bytes) in lengths.iter().enumerate() {
            for i in 0..bytes / 2 {
                let q = (i as u32)
                    .wrapping_mul(1664525)
                    .wrapping_add((member as u32).wrapping_mul(1013904223))
                    .wrapping_add(123);
                let sign = ((q >> 31) as u16) << 15;
                let exponent = if member == 1 {
                    119 + ((q >> 8) % 5)
                } else if i % 37 == 0 {
                    0
                } else {
                    112 + ((q >> 8) % 24)
                };
                let rounded = sign | ((exponent as u16) << 7) | ((q >> 16) as u16 & 127);
                payload.extend_from_slice(&rounded.to_le_bytes());
            }
        }
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-portable-bf16-arithmetic-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        fs::write(path.join("source.safetensors"), &payload)?;
        let bytes = payload.len() as u64;
        let digest = format!("{:x}", Sha256::digest(&payload));
        drop(payload);
        drop(ticket);
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[zerogravity_core::resource_arbiter::AllocationRequest {
                kind: zerogravity_core::resource_arbiter::AllocationClass::Scratch,
                bytes: 64 * 1024,
            }],
        )?;
        let index=format!(r#"{{"metadata":{{"total_size":{}}},"weight_map":{{"embedding":"source.safetensors","gamma":"source.safetensors","projection":"source.safetensors"}}}}"#,lengths.iter().sum::<usize>()).into_bytes();
        Ok(Self {
            path,
            index,
            digest,
            bytes,
            admission,
            _metadata: metadata,
        })
    }
    fn source(&self) -> Result<VerifiedTensorSource> {
        let artifact = Artifact::from_file(
            File::open(self.path.join("source.safetensors"))?,
            std::path::Path::new("source.safetensors"),
        )?;
        SourceWeights::prepare_verified_windows(
            &self.index,
            &format!("{:x}", Sha256::digest(&self.index)),
            vec![ExpectedSourceShard::new(
                "source.safetensors".into(),
                artifact,
                self.bytes,
                self.digest.clone(),
            )?],
            WindowOpenLimits {
                max_source_bytes: 2 * 1024 * 1024,
                max_metadata_bytes: 16 * 1024 * 1024,
                max_shards: 1,
                max_tensors: 3,
                scan_chunk_bytes: 1024 * 1024,
            },
            &self.admission,
        )?
        .open()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
fn descriptor(source: &VerifiedTensorSource, name: &str) -> Result<SourceTensor> {
    let m = source.matrix(name, MatrixAxes::OutputInput)?;
    Ok(SourceTensor {
        source: source.source_identity(),
        name: name.into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        rows: m.rows(),
        columns: m.columns(),
    })
}
fn record(source: &VerifiedTensorSource, extra: bool) -> Result<RecordedOperatorSlice> {
    let mut b = PlanBuilder::new_operator_slice(source.source_identity(), bindings::PROFILE)?;
    let x = b.embed_profiled(
        descriptor(source, "embedding")?,
        PlanBuilder::inputs().token,
        EmbeddingContract::Bf16OutputInput,
    )?;
    let n = b.ordinary_rms_profiled(
        source.parameter("gamma")?.source_parameter(),
        &x,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    let y = b.project_profiled(
        descriptor(source, "projection")?,
        &n,
        ProjectionContract::TorchCpuBf16Avx2,
    )?;
    let out = if extra {
        b.torch_cpu_bf16_add("unexpected extra math", &y, &y)?
    } else {
        y
    };
    b.finish_operator_slice(out)
}
#[test]
fn whole_source_three_nodes_and_foreign_issuer_refusals() -> Result<()> {
    let f = Fixture::new(1024, 33, ResourceArbiter::new())?;
    let source = f.source()?;
    let slice = record(&source, false)?;
    let shape = Shape::bind(&slice, &source, 16)?;
    assert_eq!((shape.width, shape.rows), (1024, 33));
    assert!(Shape::bind(&slice, &source, 17).is_err());
    assert!(Shape::bind(&record(&source, true)?, &source, 0).is_err());
    let foreign = f.source()?;
    assert!(Shape::bind(&slice, &foreign, 0).is_err());
    assert!(ResidentProgram::require_legacy_contract(slice.plan()).is_err());
    assert_eq!(slice.plan().frontier_delta(), 0);
    assert!(slice.plan().required_domains().is_empty());
    Ok(())
}
#[test]
fn exact_device_layout_has_no_hidden_output_credit() -> Result<()> {
    for h in [1024, 2048] {
        let f = Fixture::new(h, 97, ResourceArbiter::new())?;
        let source = f.source()?;
        let s = Shape::bind(&record(&source, false)?, &source, 0)?;
        let layout = s.layout();
        let ids = layout.iter().map(|(b, _)| *b).collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), layout.len());
        assert!(layout.iter().all(|(_, n)| *n > 0));
        let actual = layout.iter().map(|(_, n)| *n).sum::<usize>();
        assert_eq!(actual, 34 * h + 12 * 97 + 28);
        assert!(actual + 97 * h * 2 < 40 * 1024 * 1024);
    }
    Ok(())
}

fn compare_cpu(
    source: &VerifiedTensorSource,
    token: u32,
    done: &qualification::CompletedSlice<'_>,
) -> Result<()> {
    let wm = source.matrix("projection", MatrixAxes::OutputInput)?;
    let h = wm.columns();
    let _oracle = source.reserve_workspace(2 * 1024 * 1024)?;
    let em = source.matrix("embedding", MatrixAxes::OutputInput)?;
    let e = source.read_rows(&em, token as usize..token as usize + 1)?;
    let x = e
        .as_slice()
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect::<Vec<_>>();
    let gp = source.parameter("gamma")?;
    let g = source.read_parameter(&gp, 0..h)?;
    let reference = zerogravity_cpu::numerics::torch_cpu::norm::ordinary_rms(
        source.source_identity(),
        zerogravity_core::source::SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[h as u64],
            packed_bytes: g.as_slice(),
        },
        &x,
        h,
    )?;
    for (name, values) in [
        ("embedding", x.as_slice()),
        ("raw_gamma", reference.weight_f32()),
        ("runtime_gamma", reference.runtime_gamma_f32()),
        ("squares", reference.squares_f32()),
        ("sum", reference.sums_f32()),
        ("mean", reference.mean_f32()),
        ("denominator", reference.denominator_f32()),
        ("inverse", reference.inverse_f32()),
        ("normalized", reference.normalized_f32()),
        ("weighted", reference.weighted_f32()),
        ("rms_bf16", reference.output_bf16()),
    ] {
        let actual = done.value(name)?;
        for (i, (a, b)) in actual.chunks_exact(4).zip(values).enumerate() {
            if a != b.to_bits().to_le_bytes() {
                return Err(Error::State(format!(
                    "portable {name} first differing word{i}: actual={:08x} expected={:08x} words={}",
                    u32::from_le_bytes(a.try_into().expect("whole observed word")),
                    b.to_bits(),
                    values.len()
                )));
            }
        }
        if actual.len() != values.len() * 4 {
            return Err(Error::State(format!("portable {name} full extent")));
        }
    }
    let packed = done.value("rms_packed")?;
    if packed.len() != h * 2
        || packed
            .chunks_exact(2)
            .zip(reference.output_bf16())
            .any(|(a, b)| a != ((b.to_bits() >> 16) as u16).to_le_bytes())
    {
        return Err(Error::State(
            "portable exact packed RMS words differ".into(),
        ));
    }
    let sqrt = reference
        .denominator_f32()
        .iter()
        .map(|v| v.sqrt())
        .collect::<Vec<_>>();
    if done
        .value("sqrt")?
        .chunks_exact(4)
        .zip(&sqrt)
        .any(|(a, b)| a != b.to_bits().to_le_bytes())
    {
        return Err(Error::State("portable source sqrt differs".into()));
    }
    let row_capacity = source.staging_chunk_bytes() / (h * 2);
    if row_capacity == 0 {
        return Err(Error::Limit("CPU reference source row capacity".into()));
    }
    for first in (0..wm.rows()).step_by(row_capacity) {
        let end = (first + row_capacity).min(wm.rows());
        let window = source.read_rows(&wm, first..end)?;
        let output = zerogravity_cpu::source::window_projection::project_bf16_window(
            &descriptor(source, "projection")?,
            first..end,
            &window,
            reference.output_bf16(),
            None,
            None,
        )?;
        for (local, v) in output.accumulations().iter().enumerate() {
            let at = (first + local) * 4;
            let bits = v.to_bits();
            let rounded = (bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff0000;
            for (name, expected) in [("projection_f32", bits), ("projection_bf16", rounded)] {
                let actual = done.value(name)?;
                if actual.len() != wm.rows() * 4 || actual[at..at + 4] != expected.to_le_bytes() {
                    return Err(Error::State(format!(
                        "portable exact {name} word{} differs",
                        first + local
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(super) async fn cold_physical_guards(
    device: &cubecl::Device,
    inputs: &PreparedInputs,
    service: &Arc<ResidentService>,
) -> Result<()> {
    let key = service.physical_key();
    let before = ResourceArbiter::process_shared().snapshot(&key)?.reserved;
    let tiny = ResidentService::new(
        device.clone(),
        inputs.source.source_identity(),
        ResourceOwner::new("typed-provider-pressure", "before-commands")?,
        128 * 1024,
        256 * 1024 * 1024,
    )?;
    let refused = TypedBody::prepare_inputs(
        tiny.clone(),
        inputs.source.clone(),
        inputs.slice.clone(),
        inputs.token,
        None,
    );
    assert!(matches!(refused, Err(Error::Limit(_))));
    assert_eq!(tiny.resident_bytes()?, 0);
    assert_eq!(tiny.pending_events()?, 0);
    drop(tiny);
    assert_eq!(
        ResourceArbiter::process_shared().snapshot(&key)?.reserved,
        before
    );
    let requests = [zerogravity_core::resource_arbiter::AllocationRequest {
        kind: zerogravity_core::resource_arbiter::AllocationClass::Scratch,
        bytes: 1,
    }];
    let active = service
        .queue()
        .submit(service.budget(), service.owner().clone(), &requests, None)?
        .wait()
        .await?;
    let old = service.queue().stats()?.cancelled_queued;
    let mut waiting = Box::pin(
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?
            .wait(),
    );
    assert!(
        futures_lite::future::poll_once(waiting.as_mut())
            .await
            .is_none()
    );
    drop(waiting);
    assert_eq!(service.queue().stats()?.cancelled_queued, old + 1);
    // Acquired only: no Ready/submit is fabricated for this queue reservation.
    drop(active);
    service.ensure_usable()?;
    assert_eq!(service.pending_events()?, 0);
    Ok(())
}

fn selected_device(backend: &str) -> Result<cubecl::Device> {
    use cubecl::device::{WgpuBackend, WgpuDevice};
    let device = match backend {
        "hip" => cubecl::Device::Hip(Default::default()),
        "vulkan" => {
            cubecl::Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Vulkan))
        }
        _ => {
            return Err(Error::Unsupported(
                "explicit arithmetic hip|vulkan required".into(),
            ));
        }
    };
    crate::backend::validate_provider(&device)?;
    let client = device.client();
    if client
        .properties()
        .identity
        .physical
        .as_ref()
        .and_then(|p| p.pci_address)
        .map(|p| p.to_string())
        .as_deref()
        != Some("0000:c6:00.0")
    {
        return Err(Error::Unsupported(
            "portable arithmetic actual pvl-x2 PCI required".into(),
        ));
    }
    Ok(device)
}
#[test]
#[ignore = "real explicit pvl-x2 HIP/Vulkan numerical/source cohort; coordinate GPU owner"]
fn actual_portable_typed_bf16_slice() {
    futures_lite::future::block_on(async {
        let backend=std::env::var("ZG_ARITHMETIC_BACKEND").map_err(|_|Error::Unsupported("explicit arithmetic backend".into()))?;
        let device=selected_device(&backend)?;
        let mode=std::env::var("ZG_ARITHMETIC_MODE").map_err(|_|Error::Unsupported("explicit arithmetic mode".into()))?;
        let fault=match mode.as_str() {"healthy"=>qualification::SliceFault::None,"rms-unknown"|"cross-rms"=>qualification::SliceFault::AfterRms,"projection-unknown"|"cross-projection"=>qualification::SliceFault::AfterProjection,_=>return Err(Error::Unsupported("explicit arithmetic mode required".into()))};
        for h in [1024,2048] {
            let f=Fixture::new(h,97,ResourceArbiter::process_shared())?;
            let source=f.source()?;
            // Real source admission pressure and cancellation precede submission.
            assert!(source.reserve_workspace(128*1024*1024).is_err());
            let precancel=AtomicBool::new(true);let callbacks=AtomicUsize::new(0);
            assert!(qualification::qualify_portable_bf16_slice(device.clone(),source.clone(),record(&source,false)?,16,qualification::SliceFault::None,Some(&precancel),|_| {callbacks.fetch_add(1,Ordering::AcqRel);Ok(())}).await.is_err());
            assert_eq!(callbacks.load(Ordering::Acquire),0);
            let result=if mode.starts_with("cross-") {
                let peer=selected_device(if backend=="hip" {"vulkan"}else{"hip"})?;
                qualification::qualify_portable_bf16_slice_cross_provider(device.clone(),peer,source.clone(),record(&source,false)?,16,fault).await?
            } else {
                let expected_source=source.clone();
                qualification::qualify_portable_bf16_slice(device.clone(),source.clone(),record(&source,false)?,16,fault,None,|done| {
                    compare_cpu(&expected_source,16,&done)?;
                    let _expected=expected_source.reserve_workspace(1024*1024)?;
                    reference::compare(&done,&f.digest,h,97)
                }).await?
            };
            println!("PORTABLE_BF16_ARITHMETIC_RESULT width={h} result={result:?} learned=false public_torch_reference={} model=false",fault==qualification::SliceFault::None);
            drop(source);
            if fault!=qualification::SliceFault::None { return Ok(()); }
            if f.admission.arbiter.snapshot(&f.admission.resource)?.reserved!=64*1024 {return Err(Error::State("portable arithmetic original host owners remain".into()));}
            let admission=f.admission.clone();drop(f);
            if admission.arbiter.snapshot(&admission.resource)?.reserved!=0 {return Err(Error::State("portable arithmetic fixture host final0".into()));}
        }
        Ok::<(),Error>(())
    }).unwrap();
}
