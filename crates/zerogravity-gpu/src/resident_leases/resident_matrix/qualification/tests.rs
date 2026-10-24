use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};

const HOST: u64 = 32 * 1024 * 1024;
const CONSTRUCTION: u64 = 8 * 1024 * 1024;
const FIXTURE: u64 = 64 * 1024;
struct Fixture {
    path: PathBuf,
    index: Vec<u8>,
    digest: String,
    bytes: u64,
    admission: TransferAdmission,
    _control: zerogravity_core::resource_arbiter::GlobalResourceLease,
}
impl Fixture {
    fn new(rows: usize, width: usize, arbiter: ResourceArbiter) -> Result<Self> {
        let raw_bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Limit("matrix fixture byte extent".into()))?;
        if raw_bytes > 3 * 1024 * 1024 - 1024 {
            return Err(Error::Limit("matrix fixture source bound".into()));
        }
        let key = PhysicalDeviceKey::new("host", "system-ram")?;
        arbiter.register_matching(key.clone(), HOST)?;
        let owner = ResourceOwner::new("verified-matrix-fixture", "source-and-oracle")?;
        let control = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: CONSTRUCTION,
            }],
        )?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-bf16-matrix-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        let header = format!(
            "{{\"matrix\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{width}],\"data_offsets\":[0,{raw_bytes}]}}}}"
        );
        let mut bytes = Vec::with_capacity(8 + header.len() + raw_bytes);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for i in 0..rows * width {
            bytes.extend_from_slice(&(i as u16).to_le_bytes());
        }
        let digest = format!("{:x}", Sha256::digest(&bytes));
        fs::write(path.join("model.safetensors"), &bytes)?;
        let len = bytes.len() as u64;
        drop(bytes);
        // Full file payload is gone; retain only bounded path/index/hash data.
        let retained = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: FIXTURE,
            }],
        )?;
        drop(control);
        let index = format!("{{\"metadata\":{{\"total_size\":{raw_bytes}}},\"weight_map\":{{\"matrix\":\"model.safetensors\"}}}}").into_bytes();
        Ok(Self {
            path,
            index,
            digest,
            bytes: len,
            admission: TransferAdmission {
                arbiter,
                resource: key,
                owner,
            },
            _control: retained,
        })
    }
    fn source(&self) -> Result<VerifiedTensorSource> {
        self.source_with_metadata(16 * 1024 * 1024)
    }
    fn source_with_metadata(&self, metadata: u64) -> Result<VerifiedTensorSource> {
        let opened = Artifact::from_file(
            File::open(self.path.join("model.safetensors"))?,
            std::path::Path::new("model.safetensors"),
        )?;
        let shard = ExpectedSourceShard::new(
            "model.safetensors".into(),
            opened,
            self.bytes,
            self.digest.clone(),
        )?;
        SourceWeights::prepare_verified_windows(
            &self.index,
            &format!("{:x}", Sha256::digest(&self.index)),
            vec![shard],
            WindowOpenLimits {
                max_source_bytes: 3 * 1024 * 1024,
                max_metadata_bytes: metadata,
                max_shards: 1,
                max_tensors: 1,
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

#[test]
fn original_fixture_covers_all_bf16_patterns_and_valid_held_rename() {
    let f = Fixture::new(685, 1536, ResourceArbiter::new()).unwrap();
    let source = f.source().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let first = source.read_rows(&matrix, 0..43).unwrap();
    let mut seen = vec![false; 65536];
    for w in first.as_slice().chunks_exact(2) {
        seen[u16::from_le_bytes([w[0], w[1]]) as usize] = true;
    }
    assert!(seen.iter().all(|v| *v));
    drop(first);
    let foreign = f.source_with_metadata(9 * 1024 * 1024).unwrap();
    assert!(foreign.read_rows(&matrix, 0..1).is_err());
    drop(foreign);
    // Rename/unlink changes cache stamps,not the original chunk byte authority.
    let renamed = f.path.join("held.safetensors");
    fs::rename(f.path.join("model.safetensors"), &renamed).unwrap();
    fs::write(f.path.join("model.safetensors"), b"unrelated replacement").unwrap();
    let row = source.read_rows(&matrix, 684..685).unwrap();
    assert_eq!(row.as_slice().len(), 3072);
    drop(row);
    fs::remove_file(&renamed).unwrap();
    source.read_rows(&matrix, 0..1).unwrap();
    let budget = source.reserve_workspace(HOST as usize);
    assert!(matches!(budget, Err(Error::Limit(_))));
    drop((source, matrix));
    assert_eq!(
        f.admission
            .arbiter
            .snapshot(&f.admission.resource)
            .unwrap()
            .reserved,
        FIXTURE
    );
}

#[test]
fn original_corrupt_next_window_refuses_without_reminting_source() {
    let f = Fixture::new(685, 1536, ResourceArbiter::new()).unwrap();
    let source = f.source().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    source.read_rows(&matrix, 0..1).unwrap();
    let mut write = OpenOptions::new()
        .write(true)
        .open(f.path.join("model.safetensors"))
        .unwrap();
    write
        .seek(SeekFrom::Start(matrix.tensor().byte_offset + 1_100_000))
        .unwrap();
    write.write_all(&[0xff]).unwrap();
    write.sync_all().unwrap();
    assert!(source.read_rows(&matrix, 341..682).is_err());
    assert!(Geometry::new(685, 1535).is_err());
}

fn selected_device() -> Result<cubecl::Device> {
    use cubecl::device::{WgpuBackend, WgpuDevice};
    let device = match std::env::var("ZG_MATRIX_BACKEND").as_deref() {
        Ok("hip") => cubecl::Device::Hip(Default::default()),
        Ok("vulkan") => {
            cubecl::Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Vulkan))
        }
        _ => {
            return Err(Error::Unsupported(
                "set explicit ZG_MATRIX_BACKEND=hip|vulkan".into(),
            ));
        }
    };
    crate::backend::validate_provider(&device)?;
    let client = device.client();
    let pci = client
        .properties()
        .identity
        .physical
        .as_ref()
        .and_then(|p| p.pci_address)
        .map(|p| p.to_string());
    if pci.as_deref() != Some("0000:c6:00.0") {
        return Err(Error::Unsupported(
            "matrix probe requires actual pvl-x2 PCI".into(),
        ));
    }
    Ok(device)
}

#[test]
#[ignore = "real owned pvl-x2 HIP/Vulkan; fresh process and coordinated launch"]
fn actual_verified_bf16_matrix_transport() {
    futures_lite::future::block_on(async {
        let mode = match std::env::var("ZG_MATRIX_MODE").as_deref() {
            Ok("healthy") => MatrixTransportMode::Healthy,
            Ok("upload-first") => MatrixTransportMode::UnknownFirstUpload,
            Ok("upload-last") => MatrixTransportMode::UnknownLastUpload,
            Ok("rows-unknown") => MatrixTransportMode::UnknownRows,
            Ok("invalid-device-id") => MatrixTransportMode::InvalidDeviceId,
            _ => return Err(Error::Unsupported("explicit matrix mode required".into())),
        };
        let device = selected_device()?;
        for (rows,width) in [(1029,1024),(685,1536),(519,2048)] {
            let f = Fixture::new(rows,width,ResourceArbiter::process_shared())?;
            let source = f.source()?;
            let ids: Arc<[u32]> = vec![(rows-1) as u32,0,(rows/2) as u32,(rows-1) as u32].into();
            let result = qualify_bf16_matrix_transport(device.clone(),source,"matrix",ids,mode).await?;
            println!("MATRIX_TRANSPORT_PASS mode={mode:?} shape={rows}x{width} outcome={result:?} learned=false");
            if mode != MatrixTransportMode::Healthy { return Ok(()); }
            if f.admission.arbiter.snapshot(&f.admission.resource)?.reserved != FIXTURE { return Err(Error::State("matrix source/host owners not retired".into())); }
            let admission = f.admission.clone(); drop(f);
            if admission.arbiter.snapshot(&admission.resource)?.reserved != 0 { return Err(Error::State("matrix host final0".into())); }
        }
        Ok::<(),Error>(())
    }).unwrap();
}

#[test]
#[ignore = "real two-provider same-PCI unknown; fresh owned process"]
fn actual_verified_bf16_matrix_cross_provider() {
    futures_lite::future::block_on(async {
        use cubecl::device::{WgpuBackend,WgpuDevice};
        let device = selected_device()?;
        let peer = match device {
            cubecl::Device::Hip(_) => cubecl::Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Vulkan)),
            _ => cubecl::Device::Hip(Default::default()),
        };
        let f = Fixture::new(685,1536,ResourceArbiter::process_shared())?;
        let result = qualify_bf16_matrix_cross_provider(device,peer,f.source()?,"matrix",vec![684,0,342,684].into()).await?;
        println!("MATRIX_CROSS_PROVIDER_PASS outcome={result:?} learned=false; process retirement required");
        Ok::<(),Error>(())
    }).unwrap();
}

#[cfg(feature = "gemma-bf16-qualification")]
#[test]
fn prepared_projection_keeps_real_source_issuer_and_storage_initialization_separate() {
    use crate::resident_program::bf16_projection::{Bf16ProjectionStorage, PreparedBf16Projection};
    let f = Fixture::new(685, 1536, ResourceArbiter::new()).unwrap();
    let source = f.source().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let foreign = f.source_with_metadata(9 * 1024 * 1024).unwrap();
    let foreign_matrix = foreign.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    assert!(!matrix.same_binding(&foreign_matrix));
    assert!(
        PreparedBf16Projection::new(
            foreign.clone(),
            matrix.clone(),
            Bf16ProjectionStorage::Resident
        )
        .is_err()
    );
    assert!(
        PreparedBf16Projection::new(
            source.clone(),
            foreign_matrix.clone(),
            Bf16ProjectionStorage::Streamed
        )
        .is_err()
    );
    let streamed = PreparedBf16Projection::new(
        source.clone(),
        matrix.clone(),
        Bf16ProjectionStorage::Streamed,
    )
    .unwrap();
    let resident = PreparedBf16Projection::new(
        source.clone(),
        matrix.clone(),
        Bf16ProjectionStorage::Resident,
    )
    .unwrap();
    assert_eq!(streamed.launch_rows().collect::<Vec<_>>(), [341, 341, 3]);
    assert_eq!(resident.launch_rows().collect::<Vec<_>>(), [685]);
    assert_eq!(streamed.max_launch_rows(), 341);
    assert_eq!(resident.max_launch_rows(), 685);
    assert!(resident.bind_initialized(None).is_err());
    assert!(resident.resident().is_none());
    streamed.bind_initialized(None).unwrap();
    drop((source, matrix, foreign, foreign_matrix));
    assert_eq!(
        f.admission
            .arbiter
            .snapshot(&f.admission.resource)
            .unwrap()
            .reserved,
        FIXTURE + 16 * 1024 * 1024
    );
    assert!(resident.source().same_issuer(streamed.source()));
    let window = resident
        .source()
        .read_rows(resident.matrix(), 684..685)
        .unwrap();
    assert_eq!(window.as_slice().len(), 3072);
    drop(window);
    drop((resident, streamed));
    assert_eq!(
        f.admission
            .arbiter
            .snapshot(&f.admission.resource)
            .unwrap()
            .reserved,
        FIXTURE
    );
}

#[cfg(feature = "gemma-bf16-qualification")]
#[test]
fn streamed_projection_refuses_a_full_k_row_larger_than_staging_before_iterator() {
    use crate::resident_program::bf16_projection::{Bf16ProjectionStorage, PreparedBf16Projection};
    let f = Fixture::new(1, 524544, ResourceArbiter::new()).unwrap();
    let source = f.source().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    assert!(
        PreparedBf16Projection::new(
            source.clone(),
            matrix.clone(),
            Bf16ProjectionStorage::Streamed
        )
        .is_err()
    );
    assert!(PreparedBf16Projection::new(source, matrix, Bf16ProjectionStorage::Resident).is_err());
}

#[cfg(feature = "gemma-bf16-qualification")]
#[test]
#[ignore = "real owned pvl-x2 HIP/Vulkan; authentic source, no numerical/model eligibility"]
fn actual_prepared_projection_alias_refuses_before_pack_or_source_copy() {
    use crate::resident_leases::source_page::ResidentSourcePage;
    use crate::resident_program::bf16_projection::{
        Bf16ProjectionBindings, Bf16ProjectionStorage, PreparedBf16Projection,
    };
    futures_lite::future::block_on(async {
        let f = Fixture::new(685, 1536, ResourceArbiter::process_shared())?;
        let source = f.source()?;
        let original = source.matrix("matrix", MatrixAxes::OutputInput)?;
        let operand = PreparedBf16Projection::new(source.clone(), original.clone(), Bf16ProjectionStorage::Streamed)?;
        let service = ResidentService::new(selected_device()?, source.source_identity(), ResourceOwner::new("projection-negative", "original-source")?, PROVIDER, DEVICE)?;
        let mut queue = slot(&service).await?; queue.begin_submission()?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(Arc::new(source.reserve_workspace(3*crate::resident_leases::source_page::PAGE_BYTES + 64*1024)?))?;
        let page = ResidentSourcePage::allocate(source.clone(), service.clone(), &mut init)?;
        init.upload_zeroed_into(page.lease())?;
        let input_bytes = vec![1.0f32; 1536].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        let input = init.upload_bytes(&input_bytes, BufferRole::Activation)?;
        let output = init.upload_zeroed(685*4, BufferRole::Activation)?;
        let accumulation = init.upload_zeroed(operand.max_launch_rows()*4, BufferRole::Scratch)?;
        let flags = init.upload_zeroed(operand.max_launch_rows()*4, BufferRole::Scratch)?;
        let status = init.upload_zeroed(4, BufferRole::Scratch)?;
        let zero = init.upload_zeroed(4, BufferRole::ImmutableWeights)?;
        drop(init.complete_sync().await?); queue.finish()?;
        let access = [(input.clone(),DeviceAccess::Read),(output.clone(),DeviceAccess::Write),(page.lease().clone(),DeviceAccess::Write),(accumulation.clone(),DeviceAccess::Write),(flags.clone(),DeviceAccess::Write),(status.clone(),DeviceAccess::Write),(zero.clone(),DeviceAccess::Read)];
        let mut queue = slot(&service).await?; queue.begin_submission()?;
        let mut flight = service.submit(0, &access)?;
        let result = operand.enqueue(&mut flight, &page, Bf16ProjectionBindings { input:&input.handle(), input_lease:&input, output:&output.handle(), output_lease:&output, packed:page.lease(), accumulation:&accumulation, flags:&flags, status:&status, zero:&zero }, None).await;
        if !matches!(result,Err(Error::State(ref text)) if text == "BF16 original projection buffers alias") { return Err(Error::State("actual alias was not refused before pack".into())); }
        let staging = Arc::new(source.reserve_workspace(3*crate::resident_leases::source_page::PAGE_BYTES)?);
        flight.retain_completion_owner(staging)?;
        let (ready, raw) = flight.complete_with_readback(&[output.clone(),page.lease().clone()], &[page.lease().readback(0,crate::resident_leases::source_page::PAGE_BYTES)?,output.readback(0,output.bytes())?], |raw| {
            if raw.iter().any(|bytes| bytes.iter().any(|value| *value != 0)) { return Err(Error::State("refused alias changed actual page/output".into())); } Ok(())
        }).await?;
        drop((ready,raw)); queue.finish()?;
        if service.ensure_usable().is_err() { return Err(Error::State("pre-effect refusal fenced healthy source".into())); }
        drop((access,input,output,accumulation,flags,status,zero,page,operand,original,source,service));
        if f.admission.arbiter.snapshot(&f.admission.resource)?.reserved != FIXTURE { return Err(Error::State("negative projection source owners did not retire".into())); }
        println!("PREPARED_PROJECTION_ALIAS_REFUSED actual_page_output_unchanged=true published_roots=0 learned=false");
        Ok::<(),Error>(())
    }).unwrap();
}
