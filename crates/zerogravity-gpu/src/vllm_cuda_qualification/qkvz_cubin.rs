//! Closed original-source RMS→selected native N1 GEMV, qualification only.
//! No provider constructor, arbitrary manifest, reference input or model permit.
use super::cubin_manifests::{Projection, ProjectionBinding, ProjectionImage};
use super::layer_common::qkvz_weights;
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ForeignProviderCharge, ResidentReadback};
use crate::vllm_cuda_input_rms::RmsPrepared;
use cubecl::cuda::{CubinKernelManifest, OwnedCudaKernel, PinnedBf16Cubin};
use std::path::Path;
use zerogravity_artifacts::source_weights::TensorWindow;

#[cfg(test)]
mod tests;

const FOREIGN_BYTES: usize = 64 * 1024 * 1024;
const HOST_BYTES: usize = 32 * 1024 * 1024;
const IMAGE_BYTES: usize = 2_818_208;
const WEIGHT_BYTES: usize = 8192 * 1024 * 2;
const INPUT_BYTES: usize = 1024 * 2;
const OUTPUT_BYTES: usize = 8192 * 2;
const TEACHERS: [u32; 3] = [11, 353, 353];

/// Complete expected records from actual oracle calls0/1/4, postflight only.
pub struct QkvzCubinFixture {
    pub teacher_id: u32,
    pub normalized: Vec<u8>,
    pub qkvz: Vec<u8>,
}

/// Caller reserves from this SAME source before allocating the extracted image.
/// The closed vendor registry validates its exact original DSO derivation/SHA.
pub struct QkvzCubinImage {
    pub bytes: Vec<u8>,
    pub admission: WindowWorkspace,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QkvzCubinComparison {
    pub calls: usize,
    pub producer_flights: usize,
    pub rms_stages: usize,
    pub cubin_launches: usize,
    pub bf16_words: usize,
}

struct SourceOwner {
    _source: VerifiedTensorSource,
    _windows: Vec<Arc<TensorWindow>>,
    weights: Vec<u8>,
    fixtures: [QkvzCubinFixture; 3],
    // Field drop order keeps admission until the actual allocations are freed.
    _host: [WindowWorkspace; 2],
}

// This exact Arc is sealed into both image and original actor module set.
struct NativeOwner {
    _source: Arc<SourceOwner>,
    _image_admission: WindowWorkspace,
    _native_charge: Arc<ForeignProviderCharge>,
}

struct Call {
    rms: Arc<RmsPrepared>,
    output: ResidentLease,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
    reads: Vec<ResidentReadback>,
}

fn native_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("closed original QKVZ cubin: {error}"))
}

fn closed_manifest() -> Result<CubinKernelManifest> {
    let mut launches = Projection::Qkvz1.launches();
    if launches.len() != 1
        || launches[0].image() != ProjectionImage::Bf16Gemv2TSm86
        || launches[0].bindings()
            != [
                ProjectionBinding::W,
                ProjectionBinding::X,
                ProjectionBinding::Y,
            ]
    {
        return Err(Error::State(
            "closed original N1 QKVZ launch/image/bindings".into(),
        ));
    }
    let observed = launches[0].image();
    let approved = PinnedBf16Cubin::Bf16Gemv2TSm86.derivation();
    if observed.bytes() != IMAGE_BYTES
        || observed.bytes() != approved.image_bytes
        || observed.sha256() != approved.image_sha256
        || observed.binary_sm() != approved.binary_sm
    {
        return Err(Error::State(
            "observed QKVZ image differs from original closed registry".into(),
        ));
    }
    Ok(launches.remove(0).into_manifest())
}

fn validate_image_capacity(bytes: usize, capacity: usize, admission: usize) -> Result<()> {
    if bytes != IMAGE_BYTES || capacity < bytes || capacity > HOST_BYTES || admission < capacity {
        return Err(Error::Limit(
            "original cubin full owned image capacity admission".into(),
        ));
    }
    Ok(())
}

fn validate_fixtures(fixtures: &[QkvzCubinFixture; 3]) -> Result<()> {
    for (fixture, teacher) in fixtures.iter().zip(TEACHERS) {
        if fixture.teacher_id != teacher
            || fixture.normalized.len() != INPUT_BYTES
            || fixture.qkvz.len() != OUTPUT_BYTES
            || !finite_bf16(&fixture.normalized)
            || !finite_bf16(&fixture.qkvz)
            || fixture
                .normalized
                .capacity()
                .checked_add(fixture.qkvz.capacity())
                .is_none_or(|bytes| bytes > 1024 * 1024)
        {
            return Err(Error::Malformed(
                "closed actual N1 teachers/finite complete expectations".into(),
            ));
        }
    }
    Ok(())
}

fn compare_output(index: usize, actual: &[u8], expected: &[u8]) -> Result<()> {
    if actual.len() != expected.len() || !finite_bf16(actual) {
        return Err(Error::State(format!(
            "closed QKVZ call{index} finite complete output"
        )));
    }
    if let Some(first) = actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .position(|(a, b)| a != b)
    {
        return Err(Error::State(format!(
            "closed QKVZ call{index} differs at BF16 word{first}"
        )));
    }
    Ok(())
}

/// Three actual N1 teachers: original embedding/RMS then original selected GEMV.
/// The fixed template comes from the executed driver ABI, not caller addresses.
/// Original native-module metadata uses a nested64MiB reservation inside the
/// existing128MiB envelope; ELF length is not a driver-memory measurement.
pub async fn qualify_qkvz_cubin(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [QkvzCubinFixture; 3],
    image: QkvzCubinImage,
    cublas: &Path,
    cublas_lt: &Path,
    device_cap: u64,
) -> Result<QkvzCubinComparison> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for closed QKVZ".into(),
        ));
    }
    // Refuse substituted issuers or uncharged spare capacity before owning the
    // image in the native source/set and before our first allocation.
    if !image.admission.source().same_issuer(&source) {
        return Err(Error::State(
            "original cubin host/source issuer differs".into(),
        ));
    }
    validate_image_capacity(
        image.bytes.len(),
        image.bytes.capacity(),
        image.admission.bytes(),
    )?;
    validate_fixtures(&fixtures)?;
    let manifest = closed_manifest()?;
    let host = [
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
    ];
    let (windows, weights) = qkvz_weights(&source)?;
    if weights.len() != WEIGHT_BYTES || weights.capacity() > HOST_BYTES {
        return Err(Error::State(
            "closed original packed QKVZ weight extent".into(),
        ));
    }
    let owner = Arc::new(SourceOwner {
        _source: source.clone(),
        _windows: windows,
        weights,
        fixtures,
        _host: host,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-cubin", "source-rms-qkvz-n1")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(cublas, cublas_lt)
        .map_err(native_error)?;
    let charge = service.reserve_foreign_provider(FOREIGN_BYTES)?;
    if charge.bytes() != FOREIGN_BYTES {
        return Err(Error::State(
            "closed original native module reservation".into(),
        ));
    }
    let QkvzCubinImage {
        bytes: image_bytes,
        admission,
    } = image;
    let native_owner: CompletionOwner = Arc::new(NativeOwner {
        _source: owner.clone(),
        _image_admission: admission,
        _native_charge: charge,
    });
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut prepared_rms = Vec::with_capacity(3);
    for (index, fixture) in owner.fixtures.iter().enumerate() {
        prepared_rms.push(
            RmsPrepared::prepare(
                service.clone(),
                source.clone(),
                &[fixture.teacher_id],
                100 + index as u64,
            )
            .await?,
        );
    }
    let mut init = service.begin_initialization(110)?;
    init.retain_completion_owner(native_owner.clone())?;
    for rms in &prepared_rms {
        init.retain_completion_owner(rms.completion_owner())?;
    }
    let weight = init.upload_bytes(&owner.weights, BufferRole::ImmutableWeights)?;
    let mut calls = Vec::with_capacity(3);
    for rms in prepared_rms {
        let output = init.allocate_empty(OUTPUT_BYTES, BufferRole::Output)?;
        let mut accesses = rms.accesses().to_vec();
        accesses.extend([
            (weight.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ]);
        let mut outputs = rms.outputs().to_vec();
        outputs.push(output.clone());
        let mut reads = Vec::from(rms.readbacks()?);
        reads.push(output.readback(0, OUTPUT_BYTES)?);
        let call = Arc::new(Call {
            rms,
            output,
            accesses,
            outputs,
            reads,
        });
        init.retain_completion_owner(call.clone())?;
        calls.push(call);
    }
    // SAFETY: original composite owner already retained by the armed real W
    // upload. The only image arm is hash-pinned SM86 original module154.
    let image = unsafe {
        sources.derived_cubin(
            native_owner.clone(),
            PinnedBf16Cubin::Bf16Gemv2TSm86,
            image_bytes,
        )
    }
    .map_err(native_error)?;
    init.retain_completion_owner(image.clone())?;
    // SAFETY: identical original admitted Arc retained before any module load.
    let set = unsafe { queue.admit_cubin_set(native_owner.clone()) }.map_err(native_error)?;
    init.retain_completion_owner(set.clone())?;
    // SAFETY: exact captured N1 ABI/scalars; original W/RMS/Y storage only.
    let kernel: Arc<OwnedCudaKernel> =
        unsafe { set.prepare(image.clone(), manifest) }.map_err(native_error)?;
    init.retain_completion_owner(kernel.clone())?;
    drop(init.complete_sync().await?);
    slot.finish()?;

    let mut compared = QkvzCubinComparison::default();
    for (index, call) in calls.iter().enumerate() {
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: FOREIGN_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(200 + index as u64, &call.accesses)?;
        flight.retain_completion_owner(native_owner.clone())?;
        flight.retain_completion_owner(call.clone())?;
        flight.retain_completion_owner(kernel.clone())?;
        flight.retain_completion_owner(set.clone())?;
        flight.retain_completion_owner(image.clone())?;
        call.rms.enqueue()?;
        // SAFETY: same service, exact complete handles and exclusive outputs
        // retained by this original flight. RMS writes X before GEMV reads it.
        unsafe {
            kernel.enqueue(vec![
                weight.handle(),
                call.rms.normalized().handle(),
                call.output.handle(),
            ])
        }
        .map_err(native_error)?;
        let (ready, raw) = flight
            .complete_with_readback(&call.outputs, &call.reads, |raw: &[Bytes]| {
                if raw.len() != 4 {
                    return Err(Error::State("closed QKVZ complete producer cohort".into()));
                }
                let normalized = call.rms.validate_readback(&raw[..3])?;
                compare_output(index, normalized, &owner.fixtures[index].normalized)?;
                compare_output(index, &raw[3], &owner.fixtures[index].qkvz)
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.producer_flights += 1;
        compared.rms_stages += 10;
        compared.cubin_launches += 1;
        compared.bf16_words += (INPUT_BYTES + OUTPUT_BYTES) / 2;
        drop((ready, raw));
    }

    // All preceding producer flights physically completed. Native unload is
    // guarded by this real initialized-W flight, never a fabricated status.
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(999, &[(weight.clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(native_owner.clone())?;
    flight.retain_completion_owner(image.clone())?;
    flight.retain_completion_owner(set.clone())?;
    flight.retain_completion_owner(kernel.clone())?;
    // SAFETY: all actual flights completed, no concurrent prepare/enqueue.
    // Any record/unload error drops this ARMED original flight and quarantines
    // every retained admission/source/module owner; no retirement/Ready claim.
    unsafe { kernel.retire_after_physical_completion() }.map_err(native_error)?;
    unsafe { set.retire_after_physical_completion() }.map_err(native_error)?;
    let raw = flight
        .complete_read_only_readback(&[weight.readback(0, WEIGHT_BYTES)?], |raw| {
            if raw.len() != 1 || &raw[0][..] != owner.weights.as_slice() {
                return Err(Error::State("closed QKVZ original weights changed".into()));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((
        raw,
        calls,
        weight,
        kernel,
        set,
        image,
        native_owner,
        sources,
        queue,
        owner,
        source,
    ));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "closed QKVZ buffers/events/native admission not retired".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State(
            "closed QKVZ original physical envelope not retired".into(),
        ));
    }
    Ok(compared)
}
