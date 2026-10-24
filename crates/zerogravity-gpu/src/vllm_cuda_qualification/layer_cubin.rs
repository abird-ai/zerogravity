//! Original source→RMS→closed native projections→existing GDN, fixed cuts.
//! No observed intermediate can enter this connected interface.
use super::activation_cubin::resident::ActivationPrograms;
use super::activation_manifest::Activation;
use super::cubin_manifests::{Projection, ProjectionBinding, WORKSPACE_BYTES};
use super::layer_common::{Snapshot, finite_f32, load_parameters, projection_weight};
use super::mlp_cubin::{
    mlp_weight,
    resident::{MlpBindings, MlpPrograms},
};
use super::mlp_manifests::MlpProjection;
use super::projection_cubin::resident::{AdmittedImages, ProjectionBindings, ProjectionPrograms};
use super::projection_cubin::{
    CubinActivationImage, CubinQualificationInputs, native_error, not_cancelled,
    validate_activation_image, validate_images,
};
use super::*;
use crate::resident_leases::ResidentReadback;
use crate::vllm_cuda_gdn::{GdnCacheSlot, GdnInputs, GdnPrepared};
use crate::vllm_cuda_input_rms::{FusedRmsPrepared, RmsPrepared};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicBool,
};
use zerogravity_artifacts::source_weights::TensorWindow;
use zerogravity_core::device_events::BufferLeaseId;

const FOREIGN_BYTES: usize = 64 * 1024 * 1024;
const MLP_PROVIDER_CAP: usize = 144 * 1024 * 1024;
mod faults;
mod mlp_faults;
pub use faults::{
    PostnormCubinFault, PostnormCubinFaultOutcome, qualify_layer_postnorm_cubin_fault,
};
pub use mlp_faults::{
    LayerMlpCubinFault, LayerMlpCubinFaultOutcome, qualify_layer_mlp_cubin_fault,
};
#[cfg(test)]
mod tests;
const PHASES: [VllmGdnPhase; 5] = [
    VllmGdnPhase::InitialOne,
    VllmGdnPhase::CachedDecode,
    VllmGdnPhase::Initial64,
    VllmGdnPhase::ContinuedTwo,
    VllmGdnPhase::CachedDecode,
];
#[derive(Clone, Copy)]
enum LayerScope {
    FirstSequenceThroughGated,
    FivePhases,
    FivePhasesThroughPostNorm,
    FivePhasesThroughMlp,
}
impl LayerScope {
    fn phases(self) -> &'static [VllmGdnPhase] {
        match self {
            Self::FirstSequenceThroughGated => &PHASES[..2],
            Self::FivePhases | Self::FivePhasesThroughPostNorm | Self::FivePhasesThroughMlp => {
                &PHASES
            }
        }
    }
    fn projection_count(self) -> usize {
        match self {
            Self::FirstSequenceThroughGated => 2,
            Self::FivePhases | Self::FivePhasesThroughPostNorm | Self::FivePhasesThroughMlp => 3,
        }
    }
    fn postnorm(self) -> bool {
        matches!(
            self,
            Self::FivePhasesThroughPostNorm | Self::FivePhasesThroughMlp
        )
    }
    fn mlp(self) -> bool {
        matches!(self, Self::FivePhasesThroughMlp)
    }
    fn weight_count(self) -> usize {
        self.projection_count() + 2 * usize::from(self.mlp())
    }
    fn provider_cap(self) -> usize {
        if self.mlp() {
            MLP_PROVIDER_CAP
        } else {
            PROVIDER_CAP
        }
    }
    fn projections(self, phase: VllmGdnPhase) -> Result<impl Iterator<Item = Projection>> {
        Ok(Projection::for_phase(phase)?
            .into_iter()
            .take(self.projection_count()))
    }
    fn required(self, phase: VllmGdnPhase) -> BTreeSet<LayerOutput> {
        let mut required: BTreeSet<_> = LayerOutput::required(phase)
            .into_iter()
            .filter(|output| self.projection_count() == 3 || *output != LayerOutput::Projected)
            .collect();
        if self.postnorm() {
            required.extend([
                LayerOutput::PostNormSumF32,
                LayerOutput::PostNormalized,
                LayerOutput::PostResidual,
            ]);
        }
        if self.mlp() {
            required.extend([
                LayerOutput::GateUp,
                LayerOutput::Activation,
                LayerOutput::MlpOutput,
            ]);
        }
        required
    }
}
enum LayerFixtures {
    FirstSequence([LayerFixture; 2]),
    FivePhases([LayerFixture; 5]),
    FivePhasesPostNorm([LayerFixture; 5]),
    FivePhasesMlp([LayerFixture; 5]),
}
impl LayerFixtures {
    fn scope(&self) -> LayerScope {
        match self {
            Self::FirstSequence(_) => LayerScope::FirstSequenceThroughGated,
            Self::FivePhases(_) => LayerScope::FivePhases,
            Self::FivePhasesPostNorm(_) => LayerScope::FivePhasesThroughPostNorm,
            Self::FivePhasesMlp(_) => LayerScope::FivePhasesThroughMlp,
        }
    }
    fn as_slice(&self) -> &[LayerFixture] {
        match self {
            Self::FirstSequence(values) => values,
            Self::FivePhases(values)
            | Self::FivePhasesPostNorm(values)
            | Self::FivePhasesMlp(values) => values,
        }
    }
}
const INITIAL64_IDS: [u32; 64] = [
    6269, 220, 15, 15, 15, 25, 357, 22486, 10695, 449, 75124, 908, 25659, 3294, 279, 8964, 13, 561,
    22839, 7189, 2234, 11, 14984, 11, 9039, 321, 9849, 1324, 220, 16, 15, 15, 15, 13, 357, 28336,
    13205, 1754, 1622, 579, 3555, 1528, 9209, 1345, 491, 944, 28712, 13, 79091, 864, 20934, 513,
    4221, 1518, 9722, 279, 9412, 13, 198, 8556, 5887, 62, 15, 19490,
];

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerCubinComparison {
    pub calls: usize,
    pub producer_flights: usize,
    pub projection_calls: usize,
    pub native_launches: usize,
    pub rms_stages: usize,
    pub postnorm_stages: usize,
    pub mlp_projection_calls: usize,
    pub activation_launches: usize,
    pub workspace_clears: usize,
    pub gdn_launches: usize,
    pub bf16_words: usize,
    pub f32_words: usize,
}
struct LayerOwner {
    _source: VerifiedTensorSource,
    _windows: Vec<Arc<TensorWindow>>,
    _parameters: Vec<Arc<ParameterWindow>>,
    weights: [Vec<u8>; 5],
    parameters: [Vec<u8>; 4],
    fixtures: LayerFixtures,
    zero_cache: Vec<u8>,
    _host: [WindowWorkspace; 4],
}
struct MlpCall {
    gate_up: ResidentLease,
    activation: ResidentLease,
    output: ResidentLease,
}
struct Call {
    rms: Arc<RmsPrepared>,
    gdn: Arc<GdnPrepared>,
    postnorm: Option<Arc<FusedRmsPrepared>>,
    mlp: Option<MlpCall>,
    qkvz: ResidentLease,
    ba: ResidentLease,
    projected: Option<ResidentLease>,
    cache: ResidentLease,
    slot: GdnCacheSlot,
    snapshot: Option<Snapshot>,
    phase: VllmGdnPhase,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
    reads: Vec<ResidentReadback>,
}
fn validate_fixtures(fixtures: &[LayerFixture], scope: LayerScope) -> Result<()> {
    if fixtures.len() != scope.phases().len() {
        return Err(Error::Malformed("closed layer call count".into()));
    }
    let ids: [&[u32]; 5] = [&[11], &[353], &INITIAL64_IDS, &[25, 43112], &[353]];
    let mut retained = 0usize;
    for ((fixture, phase), ids) in fixtures.iter().zip(scope.phases().iter().copied()).zip(ids) {
        if fixture.teacher_ids != ids || fixture.teacher_ids.capacity() > 64 {
            return Err(Error::Malformed(
                "closed actual layer teacher IDs/phase".into(),
            ));
        }
        let required = scope.required(phase);
        let mut actual = BTreeSet::new();
        retained = retained
            .checked_add(
                fixture
                    .expected
                    .capacity()
                    .checked_mul(std::mem::size_of::<LayerExpectedRecord>())
                    .ok_or_else(|| Error::Limit("layer metadata capacity overflow".into()))?,
            )
            .ok_or_else(|| Error::Limit("layer expected capacity overflow".into()))?;
        for record in &fixture.expected {
            retained = retained
                .checked_add(record.bytes.capacity())
                .ok_or_else(|| Error::Limit("layer expected capacity overflow".into()))?;
            let finite = if record.output.word_bytes() == 4 {
                finite_f32(&record.bytes)
            } else {
                finite_bf16(&record.bytes)
            };
            if !actual.insert(record.output)
                || record.bytes.len() != record.output.bytes(phase)
                || !finite
            {
                return Err(Error::Malformed(
                    "complete finite unique layer output records".into(),
                ));
            }
        }
        if actual != required {
            return Err(Error::Malformed(
                "exact records for the selected closed layer cut".into(),
            ));
        }
    }
    if retained > 32 * 1024 * 1024 {
        return Err(Error::Limit(
            "closed layer expected capacity exceeds original host admission".into(),
        ));
    }
    Ok(())
}
fn insert_access(
    all: &mut BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)>,
    lease: ResidentLease,
    access: DeviceAccess,
) -> Result<()> {
    if let Some((prior, mode)) = all.get_mut(&lease.id()) {
        if !Arc::ptr_eq(prior.service(), lease.service()) || prior.bytes() != lease.bytes() {
            return Err(Error::State(
                "connected original lease ID changed owner/extent".into(),
            ));
        }
        if access == DeviceAccess::Write {
            *mode = DeviceAccess::Write;
        }
    } else {
        all.insert(lease.id(), (lease, access));
    }
    Ok(())
}
impl Call {
    fn prepare_accesses(
        &self,
        weights: &[ResidentLease],
        workspace: Option<&ResidentLease>,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut all = BTreeMap::new();
        for (lease, access) in self
            .rms
            .accesses()
            .iter()
            .cloned()
            .chain(self.gdn.accesses())
            .chain(
                self.postnorm
                    .iter()
                    .flat_map(|postnorm| postnorm.accesses().iter().cloned()),
            )
        {
            insert_access(&mut all, lease, access)?;
        }
        for lease in weights {
            insert_access(&mut all, lease.clone(), DeviceAccess::Read)?;
        }
        for lease in [&self.qkvz, &self.ba]
            .into_iter()
            .chain(self.projected.as_ref())
        {
            insert_access(&mut all, lease.clone(), DeviceAccess::Write)?;
        }
        if let Some(mlp) = &self.mlp {
            insert_access(
                &mut all,
                self.postnorm()?.normalized().clone(),
                DeviceAccess::Read,
            )?;
            for lease in [&mlp.gate_up, &mlp.activation, &mlp.output] {
                insert_access(&mut all, lease.clone(), DeviceAccess::Write)?;
            }
            for lease in [&mlp.gate_up, &mlp.activation] {
                insert_access(&mut all, lease.clone(), DeviceAccess::Read)?;
            }
        }
        if self.mlp.is_some()
            || Projection::for_phase(self.phase)?
                .iter()
                .take(2 + usize::from(self.projected.is_some()))
                .any(|p| {
                    p.launches()
                        .iter()
                        .any(|launch| launch.bindings().contains(&ProjectionBinding::Workspace))
                })
        {
            let workspace = workspace.ok_or_else(|| {
                Error::State("required original projection workspace absent".into())
            })?;
            insert_access(&mut all, workspace.clone(), DeviceAccess::Write)?;
        }
        if let Some(snapshot) = &self.snapshot {
            insert_access(&mut all, snapshot.lease.clone(), DeviceAccess::Write)?;
        }
        Ok(all.into_values().collect())
    }
    fn readback(&self, output: LayerOutput) -> Result<ResidentReadback> {
        let bytes = output.bytes(self.phase);
        let (lease, offset) = match output {
            LayerOutput::Normalized => (self.rms.normalized(), 0),
            LayerOutput::Qkvz => (
                self.snapshot
                    .as_ref()
                    .map(|copy| &copy.lease)
                    .unwrap_or(&self.qkvz),
                0,
            ),
            LayerOutput::Convolution => (
                if self.phase == VllmGdnPhase::CachedDecode {
                    &self.qkvz
                } else {
                    self.gdn.lease(GdnBuffer::ConvOutput)?
                },
                0,
            ),
            LayerOutput::Projected => (
                self.projected.as_ref().ok_or_else(|| {
                    Error::State("projected output absent from this layer cut".into())
                })?,
                0,
            ),
            LayerOutput::ConvolutionState => (&self.cache, self.slot.convolution_offset()),
            LayerOutput::RecurrentState => (&self.cache, self.slot.recurrent_offset()),
            LayerOutput::PostNormSumF32 => (self.postnorm()?.sum_f32(), 0),
            LayerOutput::PostNormalized => (self.postnorm()?.normalized(), 0),
            LayerOutput::PostResidual => (self.postnorm()?.residual(), 0),
            LayerOutput::GateUp => (&self.mlp()?.gate_up, 0),
            LayerOutput::Activation => (&self.mlp()?.activation, 0),
            LayerOutput::MlpOutput => (&self.mlp()?.output, 0),
            _ => (
                self.gdn.lease(
                    output
                        .role()
                        .ok_or_else(|| Error::State("closed layer output role".into()))?,
                )?,
                0,
            ),
        };
        lease.readback(offset, bytes)
    }
    fn mlp(&self) -> Result<&MlpCall> {
        self.mlp
            .as_ref()
            .ok_or_else(|| Error::State("MLP output absent from this layer cut".into()))
    }
    fn postnorm(&self) -> Result<&FusedRmsPrepared> {
        self.postnorm
            .as_deref()
            .ok_or_else(|| Error::State("postnorm output absent from this layer cut".into()))
    }
}

/// Fixed five actual phases, including two independent original initial caches.
/// Every mathematical input comes from original teacher/source producers.
/// Expected records are reachable only by post-arithmetic validators.
pub async fn qualify_layer_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 5],
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<LayerCubinComparison> {
    qualify_layer_scope(
        device,
        source,
        LayerFixtures::FivePhases(fixtures),
        inputs,
        cancelled,
    )
    .await
}

/// Fixed five source-produced layer0 calls through post-attention dual RMS.
/// The original embedding is the residual; the actual native output projection
/// is the branch. This cut stops before MLP or any full-model admission.
pub async fn qualify_layer_postnorm_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 5],
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<LayerCubinComparison> {
    qualify_layer_scope(
        device,
        source,
        LayerFixtures::FivePhasesPostNorm(fixtures),
        inputs,
        cancelled,
    )
    .await
}

/// Distinct complete layer0 scope; original six projection images plus one
/// separately admitted selected activation resource. No full-model permit.
pub struct LayerMlpCubinInputs {
    pub projections: CubinQualificationInputs,
    pub activation: CubinActivationImage,
}

pub async fn qualify_layer_mlp_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 5],
    inputs: LayerMlpCubinInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<LayerCubinComparison> {
    match qualify_layer_inner_with_activation(
        device,
        source,
        LayerFixtures::FivePhasesMlp(fixtures),
        inputs.projections,
        Some(inputs.activation),
        Mode::Compare,
        cancelled,
    )
    .await?
    {
        Outcome::Compared(compared) => Ok(compared),
        Outcome::Fault(_) | Outcome::MlpFault(_) => Err(Error::State(
            "MLP comparison returned a fault outcome".into(),
        )),
    }
}

/// Actual InitialOne[11] then CachedDecode[353], through the original GDN gate.
/// The second call consumes the first call's computed cache. Output projection
/// is outside this cut; its weights, workspace and native images are not loaded.
pub async fn qualify_first_sequence_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 2],
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<LayerCubinComparison> {
    qualify_layer_scope(
        device,
        source,
        LayerFixtures::FirstSequence(fixtures),
        inputs,
        cancelled,
    )
    .await
}

async fn qualify_layer_scope(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: LayerFixtures,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<LayerCubinComparison> {
    match qualify_layer_inner(device, source, fixtures, inputs, Mode::Compare, cancelled).await? {
        Outcome::Compared(compared) => Ok(compared),
        Outcome::Fault(_) | Outcome::MlpFault(_) => Err(Error::State(
            "layer comparison returned fault outcome".into(),
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compare,
    Fault(PostnormCubinFault),
    MlpFault(LayerMlpCubinFault),
}
enum Outcome {
    Compared(LayerCubinComparison),
    Fault(PostnormCubinFaultOutcome),
    MlpFault(Box<LayerMlpCubinFaultOutcome>),
}

async fn qualify_layer_inner(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: LayerFixtures,
    inputs: CubinQualificationInputs,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    qualify_layer_inner_with_activation(device, source, fixtures, inputs, None, mode, cancelled)
        .await
}

async fn qualify_layer_inner_with_activation(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: LayerFixtures,
    inputs: CubinQualificationInputs,
    activation: Option<CubinActivationImage>,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    not_cancelled(cancelled)?;
    let scope = fixtures.scope();
    if scope.mlp() != activation.is_some() {
        return Err(Error::Malformed(
            "exact MLP activation resource scope".into(),
        ));
    }
    mlp_faults::validate_mode(scope, mode)?;
    if let Some(image) = &activation {
        validate_activation_image(&source, image)?;
    }
    if matches!(mode, Mode::Fault(_)) {
        faults::validate_scope(scope)?;
    }
    let phases = scope.phases();
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for connected native layer".into(),
        ));
    }
    validate_fixtures(fixtures.as_slice(), scope)?;
    if inputs.images.capacity() > 8 {
        return Err(Error::Limit("closed layer image metadata capacity".into()));
    }
    let mut projections = BTreeSet::new();
    for phase in phases.iter().copied() {
        projections.extend(scope.projections(phase)?);
    }
    let projections = projections.into_iter().collect::<Vec<_>>();
    let needs_workspace = scope.mlp()
        || projections.iter().any(|p| {
            p.launches()
                .iter()
                .any(|launch| launch.bindings().contains(&ProjectionBinding::Workspace))
        });
    // Both original executable source categories retain fixed identities.
    // Validate the exact complete image set before host/device allocations.
    validate_images(&source, &projections, &inputs.images)?;
    let host = [
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
    ];
    let mut windows = Vec::new();
    let mut weights: [Vec<u8>; 5] = std::array::from_fn(|_| Vec::new());
    for (index, value) in weights
        .iter_mut()
        .take(scope.projection_count())
        .enumerate()
    {
        not_cancelled(cancelled)?;
        let (original, raw) = projection_weight(&source, index)?;
        windows.extend(original);
        *value = raw;
    }
    if scope.mlp() {
        for (index, projection) in MlpProjection::for_phase(VllmGdnPhase::InitialOne)?
            .into_iter()
            .enumerate()
        {
            not_cancelled(cancelled)?;
            let (original, raw) = mlp_weight(&source, projection)?;
            windows.extend(original);
            weights[index + 3] = raw;
        }
    }
    let (parameters, parameter_values) = load_parameters(&source)?;
    let owner = Arc::new(LayerOwner {
        _source: source.clone(),
        _windows: windows,
        _parameters: parameters,
        weights,
        parameters: parameter_values,
        fixtures,
        zero_cache: vec![0; GdnBuffer::CacheArena.bytes(VllmGdnPhase::InitialOne)],
        _host: host,
    });
    not_cancelled(cancelled)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-cubin", "connected-original-layer0")?,
        scope.provider_cap(),
        inputs.device_cap,
    )?;
    let key = service.physical_key();
    if mode == Mode::MlpFault(LayerMlpCubinFault::AdmissionPressure) {
        let witness = mlp_faults::admission_pressure(&owner, &service)?;
        drop((owner, source, inputs, activation, service));
        return witness
            .inspect()
            .map(|result| Outcome::MlpFault(Box::new(result)));
    }
    if mode == Mode::Fault(PostnormCubinFault::AdmissionPressure) {
        let witness = faults::admission_pressure(&owner, &service)?;
        drop((owner, source, inputs, service));
        return witness.inspect().map(Outcome::Fault);
    }
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let admitted = AdmittedImages::reserve_with_activation(
        &service,
        owner.clone(),
        inputs.images,
        activation,
    )?;
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
    let mut all_rms = Vec::with_capacity(phases.len());
    for (index, fixture) in owner.fixtures.as_slice().iter().enumerate() {
        not_cancelled(cancelled)?;
        all_rms.push(
            RmsPrepared::prepare(
                service.clone(),
                source.clone(),
                &fixture.teacher_ids,
                100 + index as u64,
            )
            .await?,
        );
    }
    // GDN needs initialized original parameters/cache, but not projection W.
    // This first real source upload allows all GDN programs to be prepared
    // before the second real matrix-W upload guards native module loading.
    let mut init = service.begin_initialization(110)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    for rms in &all_rms {
        init.retain_completion_owner(rms.completion_owner())?;
    }
    let parameters = owner
        .parameters
        .iter()
        .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
        .collect::<Result<Vec<_>>>()?;
    let cache_one = init.upload_bytes(&owner.zero_cache, BufferRole::RecurrentState)?;
    let cache_four = if phases.len() == 5 {
        Some(init.upload_bytes(&owner.zero_cache, BufferRole::RecurrentState)?)
    } else {
        None
    };
    if cache_four
        .as_ref()
        .is_some_and(|cache| cache_one.id() == cache.id())
    {
        return Err(Error::State("independent native layer caches alias".into()));
    }
    let workspace = if needs_workspace {
        Some(init.allocate_empty(WORKSPACE_BYTES, BufferRole::Scratch)?)
    } else {
        None
    };
    let mut ordinary = Vec::with_capacity(phases.len());
    for phase in phases.iter().copied() {
        not_cancelled(cancelled)?;
        let [qkvz, ba, out] = Projection::for_phase(phase)?;
        ordinary.push((
            init.allocate_empty(qkvz.output_bytes(), BufferRole::Output)?,
            init.allocate_empty(ba.output_bytes(), BufferRole::Output)?,
            if scope.projection_count() == 3 {
                Some(init.allocate_empty(out.output_bytes(), BufferRole::Output)?)
            } else {
                None
            },
            if scope.mlp() {
                let [gate_up, down] = MlpProjection::for_phase(phase)?;
                Some(MlpCall {
                    gate_up: init.allocate_empty(gate_up.output_bytes(), BufferRole::Output)?,
                    activation: init.allocate_empty(down.input_bytes(), BufferRole::Activation)?,
                    output: init.allocate_empty(down.output_bytes(), BufferRole::Output)?,
                })
            } else {
                None
            },
            if phase == VllmGdnPhase::CachedDecode {
                Some(init.allocate_empty(16384, BufferRole::Output)?)
            } else {
                None
            },
        ));
    }
    drop(init.complete_sync().await?);
    let mut prepared = Vec::with_capacity(phases.len());
    for (index, ((phase, rms), (qkvz, ba, projected, mlp, snapshot))) in phases
        .iter()
        .copied()
        .zip(all_rms)
        .zip(ordinary)
        .enumerate()
    {
        not_cancelled(cancelled)?;
        let cache_slot = if index < 2 {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let cache = if index < 2 {
            cache_one.clone()
        } else {
            cache_four
                .as_ref()
                .ok_or_else(|| Error::State("second original cache absent".into()))?
                .clone()
        };
        let gdn = GdnPrepared::prepare_with_cancel(
            service.clone(),
            phase,
            cache_slot,
            GdnInputs {
                qkvz: qkvz.clone(),
                ba: ba.clone(),
                cache: cache.clone(),
                conv_weights: parameters[0].clone(),
                a_log: parameters[1].clone(),
                dt_bias: parameters[2].clone(),
                gamma: parameters[3].clone(),
            },
            cancelled,
        )
        .await?;
        let snapshot = snapshot
            .map(|lease| Snapshot::prepare(&service, &qkvz, lease))
            .transpose()?;
        let postnorm = if scope.postnorm() {
            Some(
                FusedRmsPrepared::prepare(
                    service.clone(),
                    source.clone(),
                    phase.tokens(),
                    projected
                        .as_ref()
                        .ok_or_else(|| Error::State("postnorm needs original projection".into()))?
                        .clone(),
                    rms.embedding().clone(),
                    130 + index as u64,
                    cancelled,
                )
                .await?,
            )
        } else {
            None
        };
        prepared.push(Call {
            rms,
            gdn,
            postnorm,
            mlp,
            qkvz,
            ba,
            projected,
            cache,
            slot: cache_slot,
            snapshot,
            phase,
            accesses: Vec::new(),
            outputs: Vec::new(),
            reads: Vec::new(),
        });
    }
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(120)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    for call in &prepared {
        init.retain_completion_owner(call.rms.completion_owner())?;
        init.retain_completion_owner(call.gdn.clone())?;
        if let Some(postnorm) = &call.postnorm {
            init.retain_completion_owner(postnorm.completion_owner())?;
        }
    }
    let weights = owner
        .weights
        .iter()
        .take(scope.weight_count())
        .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
        .collect::<Result<Vec<_>>>()?;
    let mut calls = Vec::with_capacity(phases.len());
    for (index, mut call) in prepared.into_iter().enumerate() {
        call.accesses = call.prepare_accesses(&weights, workspace.as_ref())?;
        call.outputs = call
            .accesses
            .iter()
            .filter(|(_, access)| *access == DeviceAccess::Write)
            .map(|(lease, _)| lease.clone())
            .collect();
        call.reads = Vec::from(call.rms.readbacks()?);
        if let Some(postnorm) = &call.postnorm {
            call.reads.extend(postnorm.readbacks()?);
        }
        for record in &owner.fixtures.as_slice()[index].expected {
            call.reads.push(call.readback(record.output)?);
        }
        let call = Arc::new(call);
        init.retain_completion_owner(call.clone())?;
        calls.push(call);
    }
    let images = if scope.mlp() {
        admitted.prepare_connected_images(sources.clone(), &mut init, cancelled)?
    } else {
        admitted.prepare_images(sources.clone(), &mut init, cancelled)?
    };
    let programs = ProjectionPrograms::prepare(images.clone(), &projections, &mut init, cancelled)?;
    let mlp_programs = if scope.mlp() {
        let mut operations = BTreeSet::new();
        for phase in phases.iter().copied() {
            operations.extend(MlpProjection::for_phase(phase)?);
        }
        let operations = operations.into_iter().collect::<Vec<_>>();
        Some(MlpPrograms::prepare(
            images.clone(),
            &operations,
            &mut init,
            cancelled,
        )?)
    } else {
        None
    };
    let activation_programs = if scope.mlp() {
        let operations = phases
            .iter()
            .copied()
            .map(Activation::for_phase)
            .collect::<Result<BTreeSet<_>>>()?
            .into_iter()
            .collect::<Vec<_>>();
        Some(ActivationPrograms::prepare(
            images.clone(),
            &operations,
            &mut init,
            cancelled,
        )?)
    } else {
        None
    };
    drop(init.complete_sync().await?);
    slot.finish()?;
    let mut compared = LayerCubinComparison::default();
    let mut index = 0;
    while let Some(call) = calls.get(index).cloned() {
        let mlp_abandoned = if let Mode::MlpFault(fault) = mode {
            Some(mlp_faults::UnknownWitness::new(
                fault,
                &owner,
                &call,
                &calls,
                &programs,
                mlp_programs
                    .as_ref()
                    .ok_or_else(|| Error::State("MLP fault programs absent".into()))?,
                activation_programs
                    .as_ref()
                    .ok_or_else(|| Error::State("MLP fault activation absent".into()))?,
                &weights,
                &parameters,
                workspace
                    .as_ref()
                    .ok_or_else(|| Error::State("MLP fault workspace absent".into()))?,
                &service,
            )?)
        } else {
            None
        };
        let abandoned = if let Mode::Fault(fault) = mode {
            Some(faults::UnknownWitness::new(
                fault, &owner, &call, &programs, &service,
            )?)
        } else {
            None
        };
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
        let mut flight = service.submit(300 + index as u64, &call.accesses)?;
        flight.retain_completion_owner(programs.clone())?;
        if scope.mlp() {
            flight.retain_completion_owner(images.clone())?;
        }
        if let Some(programs) = &mlp_programs {
            flight.retain_completion_owner(programs.clone())?;
        }
        if let Some(programs) = &activation_programs {
            flight.retain_completion_owner(programs.clone())?;
        }
        flight.retain_completion_owner(call.clone())?;
        flight.retain_completion_owner(call.rms.completion_owner())?;
        flight.retain_completion_owner(call.gdn.clone())?;
        if let Some(postnorm) = &call.postnorm {
            flight.retain_completion_owner(postnorm.completion_owner())?;
        }
        not_cancelled(cancelled)?;
        call.rms.enqueue()?;
        let operations = Projection::for_phase(call.phase)?;
        let mut native_launches = 0;
        for (projection, output) in [(0, &call.qkvz), (1, &call.ba)] {
            native_launches += programs.enqueue(
                operations[projection],
                ProjectionBindings {
                    weights: &weights[projection],
                    input: call.rms.normalized(),
                    output,
                    workspace: workspace.as_ref(),
                },
                cancelled,
            )?;
        }
        if let Some(snapshot) = &call.snapshot {
            not_cancelled(cancelled)?;
            snapshot.launch(&service)?;
        }
        let gdn_launches = call.gdn.launch(cancelled)?;
        if let Some(projected) = &call.projected {
            native_launches += programs.enqueue(
                operations[2],
                ProjectionBindings {
                    weights: &weights[2],
                    input: call.gdn.lease(GdnBuffer::Gated)?,
                    output: projected,
                    workspace: workspace.as_ref(),
                },
                cancelled,
            )?;
        }
        let postnorm_stages = if let Some(postnorm) = &call.postnorm {
            if mode == Mode::Fault(PostnormCubinFault::CancelBeforeFused) {
                let flag = AtomicBool::new(true);
                if !matches!(postnorm.enqueue(Some(&flag)), Err(Error::State(_))) {
                    return Err(Error::State(
                        "late postnorm cancellation did not refuse".into(),
                    ));
                }
                0
            } else {
                postnorm.enqueue(cancelled)?;
                10
            }
        } else {
            0
        };
        let mut mlp_calls = 0;
        let mut activation_launches = 0;
        let mut workspace_clears = 0;
        if let Some(mlp) = &call.mlp {
            let programs = mlp_programs
                .as_ref()
                .ok_or_else(|| Error::State("original MLP programs absent".into()))?;
            let activation = activation_programs
                .as_ref()
                .ok_or_else(|| Error::State("original activation programs absent".into()))?;
            let [gate_up, down] = MlpProjection::for_phase(call.phase)?;
            let gate = programs.enqueue(
                gate_up,
                MlpBindings {
                    weights: &weights[3],
                    input: call.postnorm()?.normalized(),
                    output: &mlp.gate_up,
                    workspace: workspace.as_ref(),
                },
                cancelled,
            )?;
            native_launches += gate.kernels;
            workspace_clears += gate.clears;
            activation_launches = activation.enqueue(
                Activation::for_phase(call.phase)?,
                &mlp.gate_up,
                &mlp.activation,
                cancelled,
            )?;
            native_launches += activation_launches;
            let bindings = MlpBindings {
                weights: &weights[4],
                input: &mlp.activation,
                output: &mlp.output,
                workspace: workspace.as_ref(),
            };
            let result = if mode == Mode::MlpFault(LayerMlpCubinFault::CancelBeforeDown) {
                let flag = AtomicBool::new(true);
                if !matches!(programs.enqueue(down, bindings, Some(&flag)), Err(Error::State(ref message))
                    if message == "closed original cubin qualification cancelled")
                {
                    return Err(Error::State(
                        "connected MLP late cancellation did not refuse Down".into(),
                    ));
                }
                Default::default()
            } else {
                programs.enqueue(down, bindings, cancelled)?
            };
            native_launches += result.kernels;
            workspace_clears += result.clears;
            mlp_calls = 1 + usize::from(result.kernels != 0);
        }
        if let Some(witness) = mlp_abandoned {
            // SAME armed producer/owner drops; no readback, sync or retirement.
            drop(flight);
            drop(slot);
            drop((
                call,
                calls,
                weights,
                parameters,
                cache_one,
                cache_four,
                workspace,
                programs,
                images,
                mlp_programs,
                activation_programs,
                sources,
                queue,
                owner,
                source,
                service,
                inputs.cublas,
                inputs.cublas_lt,
            ));
            return witness
                .inspect(mlp_faults::SubmittedStages {
                    rms: 10,
                    native: native_launches,
                    gdn: gdn_launches,
                    postnorm: postnorm_stages,
                    mlp_calls,
                    activation: activation_launches,
                    clears: workspace_clears,
                })
                .map(|result| Outcome::MlpFault(Box::new(result)));
        }
        if let Some(witness) = abandoned {
            // Original armed drops retain real submissions. No readback,
            // producer Ready, synchronization or native retirement follows.
            drop(flight);
            drop(slot);
            drop((
                call,
                calls,
                weights,
                parameters,
                cache_one,
                cache_four,
                workspace,
                programs,
                images,
                mlp_programs,
                activation_programs,
                sources,
                queue,
                owner,
                source,
                service,
                inputs.cublas,
                inputs.cublas_lt,
            ));
            return witness
                .inspect(faults::SubmittedStages {
                    rms: 10,
                    native: native_launches,
                    gdn: gdn_launches,
                    postnorm: postnorm_stages,
                })
                .map(Outcome::Fault);
        }
        let expected_start = 3 + 4 * usize::from(call.postnorm.is_some());
        let (ready, raw) = flight
            .complete_with_readback(&call.outputs, &call.reads, |raw| {
                if raw.len() != expected_start + owner.fixtures.as_slice()[index].expected.len() {
                    return Err(Error::State(
                        "complete original layer readback cohort".into(),
                    ));
                }
                call.rms.validate_readback(&raw[..3])?;
                if let Some(postnorm) = &call.postnorm {
                    postnorm.validate_readback(&raw[3..expected_start])?;
                }
                for (record, actual) in owner.fixtures.as_slice()[index]
                    .expected
                    .iter()
                    .zip(&raw[expected_start..])
                {
                    if let Some(first) =
                        record
                            .output
                            .first_difference(call.phase, actual, &record.bytes)?
                    {
                        return Err(Error::State(format!(
                            "connected native call{index} {:?} differs at word{first}",
                            record.output
                        )));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.producer_flights += 1;
        compared.projection_calls += scope.projection_count();
        compared.native_launches += native_launches;
        compared.rms_stages += 10;
        compared.postnorm_stages += postnorm_stages;
        compared.mlp_projection_calls += mlp_calls;
        compared.activation_launches += activation_launches;
        compared.workspace_clears += workspace_clears;
        compared.gdn_launches += gdn_launches;
        for record in &owner.fixtures.as_slice()[index].expected {
            if record.output.word_bytes() == 4 {
                compared.f32_words += record.bytes.len() / 4;
            } else {
                compared.bf16_words += record.bytes.len() / 2;
            }
        }
        drop((ready, raw));
        index += 1;
    }
    let immutable = weights
        .iter()
        .chain(&parameters)
        .cloned()
        .collect::<Vec<_>>();
    let accesses = immutable
        .iter()
        .map(|lease| (lease.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = immutable
        .iter()
        .map(|lease| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
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
    let mut flight = service.submit(999, &accesses)?;
    flight.retain_completion_owner(programs.clone())?;
    if scope.mlp() {
        flight.retain_completion_owner(images.clone())?;
    }
    if let Some(programs) = &mlp_programs {
        flight.retain_completion_owner(programs.clone())?;
    }
    if let Some(programs) = &activation_programs {
        flight.retain_completion_owner(programs.clone())?;
    }
    for call in &calls {
        flight.retain_completion_owner(call.clone())?;
    }
    not_cancelled(cancelled)?;
    // SAFETY: all original producer flights completed. Native retirement stays
    // inside this retained original read-only flight through trusted readback.
    unsafe { programs.retire_kernels_after_physical_completion() }.map_err(native_error)?;
    if let Some(programs) = &mlp_programs {
        unsafe { programs.retire_kernels_after_physical_completion() }.map_err(native_error)?;
    }
    if let Some(programs) = &activation_programs {
        unsafe { programs.retire_kernels_after_physical_completion() }.map_err(native_error)?;
    }
    unsafe { images.set().retire_after_physical_completion() }.map_err(native_error)?;
    let raw = flight
        .complete_read_only_readback(&reads, |raw| {
            if raw.len() != scope.weight_count() + owner.parameters.len()
                || raw
                    .iter()
                    .zip(
                        owner
                            .weights
                            .iter()
                            .take(scope.weight_count())
                            .chain(&owner.parameters),
                    )
                    .any(|(a, b)| &a[..] != b.as_slice())
            {
                return Err(Error::State(
                    "connected native original source parameters changed".into(),
                ));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((
        raw,
        reads,
        accesses,
        immutable,
        calls,
        weights,
        parameters,
        cache_one,
        cache_four,
        workspace,
        programs,
        images,
        mlp_programs,
        activation_programs,
        sources,
        queue,
        owner,
        source,
    ));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "connected native buffers/events/native admission not retired".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State(
            "connected native original envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(compared))
}
