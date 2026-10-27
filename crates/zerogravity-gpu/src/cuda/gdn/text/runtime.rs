//! Call-free preparation for ordinary parameterized GDN execution.
//! The caller still owns admitted buffers, ordering, completion and publication.
use super::*;
use cubecl_runtime::kernel::BufferIOAttr;
use std::{collections::BTreeMap, fmt, sync::Arc};
use zerogravity_core::accelerator::program::SourceConvGeometry;

const PREFILL_ROWS: usize = 64;
const GDN_HEADS: usize = 16;
const GDN_HEAD_WIDTH: usize = 128;

/// Closed mathematical geometry implemented by the retained parameterized source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeKernelGeometry {
    qkv_channels: usize,
    z_channels: usize,
    coefficient_channels: usize,
    convolution_kernel: usize,
    heads: usize,
    head_width: usize,
    chunk_rows: usize,
    cache_slots: usize,
}
impl GdnRuntimeKernelGeometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qkv_channels: usize,
        z_channels: usize,
        coefficient_channels: usize,
        convolution_kernel: usize,
        heads: usize,
        head_width: usize,
        chunk_rows: usize,
        cache_slots: usize,
    ) -> Result<Self> {
        let geometry = Self {
            qkv_channels,
            z_channels,
            coefficient_channels,
            convolution_kernel,
            heads,
            head_width,
            chunk_rows,
            cache_slots,
        };
        if geometry.qkv_channels != 6144
            || geometry.z_channels != 2048
            || geometry.coefficient_channels != 32
            || geometry.convolution_kernel != 4
            || geometry.heads != GDN_HEADS
            || geometry.head_width != GDN_HEAD_WIDTH
            || geometry.chunk_rows != PREFILL_ROWS
            || geometry.cache_slots != CACHE_SLOTS
        {
            return Err(Error::Unsupported(
                "parameterized GDN runtime kernel geometry".into(),
            ));
        }
        Ok(geometry)
    }

    pub fn qkv_channels(self) -> usize {
        self.qkv_channels
    }
    pub fn z_channels(self) -> usize {
        self.z_channels
    }
    pub fn coefficient_channels(self) -> usize {
        self.coefficient_channels
    }
    pub fn convolution_kernel(self) -> usize {
        self.convolution_kernel
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn head_width(self) -> usize {
        self.head_width
    }
    pub fn chunk_rows(self) -> usize {
        self.chunk_rows
    }
    pub fn cache_slots(self) -> usize {
        self.cache_slots
    }

    fn conv(self) -> SourceConvGeometry {
        SourceConvGeometry {
            qkv_channels: self.qkv_channels,
            z_channels: self.z_channels,
            coefficient_channels: self.coefficient_channels,
            kernel: self.convolution_kernel,
        }
    }
}

/// Explicit source-family configuration for ordinary calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeConfig {
    kernels: GdnKernelConfig,
    geometry: GdnRuntimeKernelGeometry,
}
impl GdnRuntimeConfig {
    pub fn new(kernels: GdnKernelConfig, geometry: GdnRuntimeKernelGeometry) -> Self {
        Self { kernels, geometry }
    }
    pub fn kernels(self) -> GdnKernelConfig {
        self.kernels
    }
    pub fn geometry(self) -> GdnRuntimeKernelGeometry {
        self.geometry
    }
}

/// One explicitly selected state slot in the retained five-slot arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeCacheSlot(u8);
impl GdnRuntimeCacheSlot {
    pub fn new(index: usize) -> Result<Self> {
        if index >= CACHE_SLOTS {
            return Err(Error::Unsupported(
                "parameterized GDN runtime cache slot".into(),
            ));
        }
        Ok(Self(index as u8))
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
    pub fn convolution_offset(self) -> usize {
        self.index() * CACHE_SLOT_BYTES
    }
    pub fn recurrent_offset(self) -> usize {
        self.convolution_offset() + CONV_STATE_BYTES
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeMode {
    Prefill,
    CachedDecode,
}

/// Call-free causal geometry. `frontier` is the number of prior tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeGeometry {
    mode: GdnRuntimeMode,
    frontier: usize,
    rows: usize,
    has_initial: bool,
}
impl GdnRuntimeGeometry {
    fn prefill(frontier: usize, rows: usize, has_initial: bool) -> Result<Self> {
        if !(1..=PREFILL_ROWS).contains(&rows)
            || has_initial != (frontier != 0)
            || frontier.checked_add(rows).is_none()
        {
            return Err(Error::Unsupported(
                "parameterized GDN runtime prefill geometry/state".into(),
            ));
        }
        Ok(Self {
            mode: GdnRuntimeMode::Prefill,
            frontier,
            rows,
            has_initial,
        })
    }

    fn cached(frontier: usize) -> Result<Self> {
        if frontier == 0 || frontier.checked_add(1).is_none() {
            return Err(Error::Unsupported(
                "parameterized GDN runtime cached frontier".into(),
            ));
        }
        Ok(Self {
            mode: GdnRuntimeMode::CachedDecode,
            frontier,
            rows: 1,
            has_initial: true,
        })
    }

    pub fn mode(self) -> GdnRuntimeMode {
        self.mode
    }
    pub fn frontier(self) -> usize {
        self.frontier
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn has_initial(self) -> bool {
        self.has_initial
    }
    pub fn resulting_frontier(self) -> usize {
        self.frontier + self.rows
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GdnRuntimeBuffer {
    PackedQkvz,
    PackedBa,
    AContiguous,
    BContiguous,
    ZCompact,
    ConvWeights,
    ALog,
    DtBias,
    GatedGamma,
    CacheArena,
    CacheIndices,
    HasInitial,
    QueryStarts,
    ConvBatch,
    ConvChunkOffsets,
    ChunkIndices,
    ChunkOffsets,
    ConvOutput,
    Q,
    K,
    V,
    GRaw,
    Beta,
    GCumulative,
    Kkt,
    Inverse,
    W,
    U,
    HChunks,
    InitialState,
    FinalState,
    VNew,
    Core,
    Gated,
    Rstd,
}
impl From<GdnBuffer> for GdnRuntimeBuffer {
    fn from(value: GdnBuffer) -> Self {
        use GdnBuffer::*;
        match value {
            PackedQkvz => Self::PackedQkvz,
            PackedBa => Self::PackedBa,
            AContiguous => Self::AContiguous,
            BContiguous => Self::BContiguous,
            ZCompact => Self::ZCompact,
            ConvWeights => Self::ConvWeights,
            ALog => Self::ALog,
            DtBias => Self::DtBias,
            GatedGamma => Self::GatedGamma,
            CacheArena => Self::CacheArena,
            CacheIndices => Self::CacheIndices,
            HasInitial => Self::HasInitial,
            QueryStarts => Self::QueryStarts,
            ConvBatch => Self::ConvBatch,
            ConvChunkOffsets => Self::ConvChunkOffsets,
            ChunkIndices => Self::ChunkIndices,
            ChunkOffsets => Self::ChunkOffsets,
            ConvOutput => Self::ConvOutput,
            Q => Self::Q,
            K => Self::K,
            V => Self::V,
            GRaw => Self::GRaw,
            Beta => Self::Beta,
            GCumulative => Self::GCumulative,
            Kkt => Self::Kkt,
            Inverse => Self::Inverse,
            W => Self::W,
            U => Self::U,
            HChunks => Self::HChunks,
            InitialState => Self::InitialState,
            FinalState => Self::FinalState,
            VNew => Self::VNew,
            Core => Self::Core,
            Gated => Self::Gated,
            Rstd => Self::Rstd,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeBufferOwnership {
    InputOutput,
    Parameters,
    CacheState,
    Control,
    Scratch,
}
fn ownership(role: GdnBuffer) -> GdnRuntimeBufferOwnership {
    use GdnBuffer::*;
    match role {
        PackedQkvz | PackedBa | ConvOutput | ZCompact | Gated | Rstd => {
            GdnRuntimeBufferOwnership::InputOutput
        }
        ConvWeights | ALog | DtBias | GatedGamma => GdnRuntimeBufferOwnership::Parameters,
        CacheArena => GdnRuntimeBufferOwnership::CacheState,
        CacheIndices | HasInitial | QueryStarts | ConvBatch | ConvChunkOffsets | ChunkIndices
        | ChunkOffsets => GdnRuntimeBufferOwnership::Control,
        _ => GdnRuntimeBufferOwnership::Scratch,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeBufferLayout {
    role: GdnRuntimeBuffer,
    bytes: usize,
    ownership: GdnRuntimeBufferOwnership,
}
impl GdnRuntimeBufferLayout {
    pub fn role(self) -> GdnRuntimeBuffer {
        self.role
    }
    pub fn bytes(self) -> usize {
        self.bytes
    }
    pub fn ownership(self) -> GdnRuntimeBufferOwnership {
        self.ownership
    }
}

/// One maximum-64-row layout shared by every prefill chunk and cached decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnRuntimeWorkspaceLayout {
    config: GdnRuntimeConfig,
    buffers: Vec<GdnRuntimeBufferLayout>,
}
impl GdnRuntimeWorkspaceLayout {
    pub fn new(config: GdnRuntimeConfig) -> Result<Self> {
        let phase = GdnPhase::TextPrefill(PREFILL_ROWS);
        let mut roles = BTreeMap::new();
        for &stage in phase.stages() {
            let spec = build_stage_spec(config, phase, stage)?;
            for binding in spec.bindings {
                roles
                    .entry(binding.role)
                    .and_modify(|bytes: &mut usize| {
                        *bytes = (*bytes).max(binding.role.bytes(phase))
                    })
                    .or_insert_with(|| binding.role.bytes(phase));
            }
        }
        for &stage in GdnPhase::CachedDecode.stages() {
            let spec = build_stage_spec(config, GdnPhase::CachedDecode, stage)?;
            for binding in spec.bindings {
                roles
                    .entry(binding.role)
                    .and_modify(|bytes: &mut usize| {
                        *bytes = (*bytes).max(binding.role.bytes(GdnPhase::CachedDecode));
                    })
                    .or_insert_with(|| binding.role.bytes(GdnPhase::CachedDecode));
            }
        }
        let buffers = roles
            .into_iter()
            .map(|(role, bytes)| GdnRuntimeBufferLayout {
                role: role.into(),
                bytes,
                ownership: ownership(role),
            })
            .collect::<Vec<_>>();
        let result = Self { config, buffers };
        if result.bytes(GdnRuntimeBuffer::CacheArena) != Some(CACHE_SLOTS * CACHE_SLOT_BYTES)
            || result.config.geometry.cache_slots() != CACHE_SLOTS
        {
            return Err(Error::State(
                "parameterized GDN runtime five-slot workspace".into(),
            ));
        }
        Ok(result)
    }

    pub fn config(&self) -> GdnRuntimeConfig {
        self.config
    }
    pub fn buffers(&self) -> &[GdnRuntimeBufferLayout] {
        &self.buffers
    }
    pub fn bytes(&self, role: GdnRuntimeBuffer) -> Option<usize> {
        self.buffers
            .iter()
            .find(|entry| entry.role == role)
            .map(|entry| entry.bytes)
    }
    pub fn cache_slots(&self) -> usize {
        CACHE_SLOTS
    }
    pub fn cache_slot_bytes(&self) -> usize {
        CACHE_SLOT_BYTES
    }

    fn require_spec(&self, spec: &LaunchSpec) -> Result<()> {
        for binding in &spec.bindings {
            let available = self
                .bytes(binding.role.into())
                .ok_or_else(|| Error::State("parameterized GDN runtime workspace role".into()))?;
            if binding
                .offset
                .checked_add(binding.physical_bytes)
                .is_none_or(|end| end > available)
            {
                return Err(Error::State(
                    "parameterized GDN runtime workspace binding extent".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeStageKind {
    Conv,
    Prep,
    Cumsum,
    Kkt,
    Solve,
    Recompute,
    State,
    Output,
    Gated,
    Decode,
}
impl From<GdnStage> for GdnRuntimeStageKind {
    fn from(value: GdnStage) -> Self {
        match value {
            GdnStage::Conv => Self::Conv,
            GdnStage::Prep => Self::Prep,
            GdnStage::Cumsum => Self::Cumsum,
            GdnStage::Kkt => Self::Kkt,
            GdnStage::Solve => Self::Solve,
            GdnStage::Recompute => Self::Recompute,
            GdnStage::H => Self::State,
            GdnStage::Output => Self::Output,
            GdnStage::Gated => Self::Gated,
            GdnStage::Decode => Self::Decode,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    Dead,
}
impl From<BufferIOAttr> for GdnRuntimeAccess {
    fn from(value: BufferIOAttr) -> Self {
        match value {
            BufferIOAttr::ReadOnly => Self::ReadOnly,
            BufferIOAttr::WriteOnly => Self::WriteOnly,
            BufferIOAttr::ReadWrite => Self::ReadWrite,
            BufferIOAttr::Dead => Self::Dead,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeBinding {
    role: GdnRuntimeBuffer,
    offset: usize,
    bytes: usize,
    access: GdnRuntimeAccess,
}
impl GdnRuntimeBinding {
    pub fn role(self) -> GdnRuntimeBuffer {
        self.role
    }
    pub fn offset(self) -> usize {
        self.offset
    }
    pub fn bytes(self) -> usize {
        self.bytes
    }
    pub fn access(self) -> GdnRuntimeAccess {
        self.access
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeScalarType {
    U32,
    U64,
    F32Bits,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeScalarField {
    ty: GdnRuntimeScalarType,
    offset: usize,
}
impl GdnRuntimeScalarField {
    pub fn ty(self) -> GdnRuntimeScalarType {
        self.ty
    }
    pub fn offset(self) -> usize {
        self.offset
    }
}

/// Exact selected source, ABI and launch geometry for one stage.
#[derive(Clone, PartialEq, Eq)]
pub struct GdnRuntimeStageIdentity {
    config: GdnRuntimeConfig,
    geometry: GdnRuntimeGeometry,
    kind: GdnRuntimeStageKind,
    entry: &'static str,
    source: &'static str,
    threads: u32,
    shared_bytes: usize,
    grid: [u32; 3],
    bindings: Vec<GdnRuntimeBinding>,
    scalar_fields: Vec<GdnRuntimeScalarField>,
    scalar_words: Vec<u64>,
}
impl fmt::Debug for GdnRuntimeStageIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GdnRuntimeStageIdentity")
            .field("config", &self.config)
            .field("geometry", &self.geometry)
            .field("kind", &self.kind)
            .field("entry", &self.entry)
            .field("source_bytes", &self.source.len())
            .field("threads", &self.threads)
            .field("shared_bytes", &self.shared_bytes)
            .field("grid", &self.grid)
            .field("bindings", &self.bindings)
            .field("scalar_fields", &self.scalar_fields)
            .field("scalar_words", &self.scalar_words)
            .finish()
    }
}
impl GdnRuntimeStageIdentity {
    pub fn config(&self) -> GdnRuntimeConfig {
        self.config
    }
    pub fn geometry(&self) -> GdnRuntimeGeometry {
        self.geometry
    }
    pub fn kind(&self) -> GdnRuntimeStageKind {
        self.kind
    }
    pub fn entry(&self) -> &'static str {
        self.entry
    }
    pub fn source(&self) -> &'static str {
        self.source
    }
    pub fn threads(&self) -> u32 {
        self.threads
    }
    pub fn shared_bytes(&self) -> usize {
        self.shared_bytes
    }
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }
    pub fn bindings(&self) -> &[GdnRuntimeBinding] {
        &self.bindings
    }
    pub fn scalar_fields(&self) -> &[GdnRuntimeScalarField] {
        &self.scalar_fields
    }
    pub fn scalar_words(&self) -> &[u64] {
        &self.scalar_words
    }
}

#[derive(Clone)]
pub struct GdnRuntimeStage {
    identity: GdnRuntimeStageIdentity,
    pub(super) spec: LaunchSpec,
}
impl GdnRuntimeStage {
    pub fn identity(&self) -> &GdnRuntimeStageIdentity {
        &self.identity
    }

    /// Validate exact pointer extents/service and device launch limits. This does
    /// not submit work or grant completion/publication authority.
    pub fn validate_launch(
        &self,
        client: &cubecl::client::Client,
        handles: &[cubecl::server::Handle],
    ) -> Result<()> {
        self.spec.kernel.validate(client, self.spec.grid, handles)
    }

    /// Bind exact views in ABI order. The enclosing physical owner must retain
    /// the leases and an admitted flight through completion or quarantine.
    pub fn prepare_launch(
        &self,
        client: &cubecl::client::Client,
        handles: Vec<cubecl::server::Handle>,
    ) -> Result<GdnRuntimeLaunch> {
        Ok(GdnRuntimeLaunch {
            identity: self.identity.clone(),
            launch: self.spec.kernel.prepare(client, self.spec.grid, handles)?,
        })
    }
}
impl fmt::Debug for GdnRuntimeStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.identity.fmt(f)
    }
}

/// Bound low-level launch; state publication remains with the caller's owner.
#[derive(Clone)]
pub struct GdnRuntimeLaunch {
    identity: GdnRuntimeStageIdentity,
    launch: crate::cuda::kernel::PreparedPinnedLaunch,
}
impl GdnRuntimeLaunch {
    pub fn identity(&self) -> &GdnRuntimeStageIdentity {
        &self.identity
    }

    /// Enqueue on the bound existing client. This is not a completion signal.
    pub fn launch(&self) -> Result<()> {
        self.launch.launch()
    }
}
impl fmt::Debug for GdnRuntimeLaunch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.identity.fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnRuntimeControl {
    role: GdnRuntimeBuffer,
    bytes: Vec<u8>,
}
impl GdnRuntimeControl {
    pub fn role(&self) -> GdnRuntimeBuffer {
        self.role
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnRuntimeCacheView {
    offset: usize,
    bytes: usize,
}
impl GdnRuntimeCacheView {
    pub fn offset(self) -> usize {
        self.offset
    }
    pub fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnRuntimeInitialState {
    Zero,
    Cache(GdnRuntimeCacheView),
    InPlace(GdnRuntimeCacheView),
}

/// Prepared source/ABI/geometry for one 1..=64-row chunk or cached token.
#[derive(Clone, Debug)]
pub struct GdnRuntimePlan {
    config: GdnRuntimeConfig,
    slot: GdnRuntimeCacheSlot,
    geometry: GdnRuntimeGeometry,
    workspace: Arc<GdnRuntimeWorkspaceLayout>,
    controls: Vec<GdnRuntimeControl>,
    stages: Vec<GdnRuntimeStage>,
}
impl GdnRuntimePlan {
    pub fn prefill_chunk(
        config: GdnRuntimeConfig,
        slot: GdnRuntimeCacheSlot,
        frontier: usize,
        rows: usize,
        has_initial: bool,
    ) -> Result<Self> {
        let workspace = Arc::new(GdnRuntimeWorkspaceLayout::new(config)?);
        Self::prefill_with_workspace(config, slot, frontier, rows, has_initial, workspace)
    }

    pub fn cached_decode(
        config: GdnRuntimeConfig,
        slot: GdnRuntimeCacheSlot,
        frontier: usize,
    ) -> Result<Self> {
        let workspace = Arc::new(GdnRuntimeWorkspaceLayout::new(config)?);
        Self::cached_with_workspace(config, slot, frontier, workspace)
    }

    fn prefill_with_workspace(
        config: GdnRuntimeConfig,
        slot: GdnRuntimeCacheSlot,
        frontier: usize,
        rows: usize,
        has_initial: bool,
        workspace: Arc<GdnRuntimeWorkspaceLayout>,
    ) -> Result<Self> {
        let geometry = GdnRuntimeGeometry::prefill(frontier, rows, has_initial)?;
        Self::prepare(config, slot, geometry, workspace)
    }

    fn cached_with_workspace(
        config: GdnRuntimeConfig,
        slot: GdnRuntimeCacheSlot,
        frontier: usize,
        workspace: Arc<GdnRuntimeWorkspaceLayout>,
    ) -> Result<Self> {
        let geometry = GdnRuntimeGeometry::cached(frontier)?;
        Self::prepare(config, slot, geometry, workspace)
    }

    fn prepare(
        config: GdnRuntimeConfig,
        slot: GdnRuntimeCacheSlot,
        geometry: GdnRuntimeGeometry,
        workspace: Arc<GdnRuntimeWorkspaceLayout>,
    ) -> Result<Self> {
        if workspace.config() != config {
            return Err(Error::State(
                "parameterized GDN runtime workspace configuration".into(),
            ));
        }
        let phase = match geometry.mode {
            GdnRuntimeMode::Prefill => GdnPhase::TextPrefill(geometry.rows),
            GdnRuntimeMode::CachedDecode => GdnPhase::CachedDecode,
        };
        let controls = runtime_controls(phase, slot, geometry.has_initial)?;
        for control in &controls {
            if workspace
                .bytes(control.role)
                .is_none_or(|available| control.bytes.len() > available)
            {
                return Err(Error::State(
                    "parameterized GDN runtime control workspace extent".into(),
                ));
            }
        }
        let mut stages = Vec::with_capacity(phase.stages().len());
        for &stage in phase.stages() {
            let (spec, asset, scalars) = build_stage(config, phase, stage)?;
            workspace.require_spec(&spec)?;
            let scalar_fields = scalars
                .schema()
                .fields_with_offsets()
                .map(|(ty, offset)| GdnRuntimeScalarField {
                    ty: match ty {
                        crate::cuda::kernel::StaticScalarType::U32 => GdnRuntimeScalarType::U32,
                        crate::cuda::kernel::StaticScalarType::U64 => GdnRuntimeScalarType::U64,
                        crate::cuda::kernel::StaticScalarType::F32Bits => {
                            GdnRuntimeScalarType::F32Bits
                        }
                    },
                    offset,
                })
                .collect();
            let bindings = spec
                .bindings
                .iter()
                .map(|binding| GdnRuntimeBinding {
                    role: binding.role.into(),
                    offset: binding.offset,
                    bytes: binding.physical_bytes,
                    access: binding.io.into(),
                })
                .collect();
            stages.push(GdnRuntimeStage {
                identity: GdnRuntimeStageIdentity {
                    config,
                    geometry,
                    kind: stage.into(),
                    entry: asset.entry,
                    source: asset.source,
                    threads: asset.threads,
                    shared_bytes: asset.shared as usize,
                    grid: asset.grid,
                    bindings,
                    scalar_fields,
                    scalar_words: scalars.words().to_vec(),
                },
                spec,
            });
        }
        Ok(Self {
            config,
            slot,
            geometry,
            workspace,
            controls,
            stages,
        })
    }

    pub fn config(&self) -> GdnRuntimeConfig {
        self.config
    }
    pub fn slot(&self) -> GdnRuntimeCacheSlot {
        self.slot
    }
    pub fn geometry(&self) -> GdnRuntimeGeometry {
        self.geometry
    }
    pub fn workspace(&self) -> &GdnRuntimeWorkspaceLayout {
        &self.workspace
    }
    pub fn controls(&self) -> &[GdnRuntimeControl] {
        &self.controls
    }
    pub fn stages(&self) -> &[GdnRuntimeStage] {
        &self.stages
    }
    pub fn stage(&self, kind: GdnRuntimeStageKind) -> Option<&GdnRuntimeStage> {
        self.stages.iter().find(|stage| stage.identity.kind == kind)
    }
    pub fn convolution_state(&self) -> GdnRuntimeCacheView {
        GdnRuntimeCacheView {
            offset: self.slot.convolution_offset(),
            bytes: CONV_STATE_BYTES,
        }
    }
    pub fn initial_recurrent_state(&self) -> GdnRuntimeInitialState {
        let view = GdnRuntimeCacheView {
            offset: self.slot.recurrent_offset(),
            bytes: RECURRENT_STATE_BYTES,
        };
        match self.geometry.mode {
            GdnRuntimeMode::CachedDecode => GdnRuntimeInitialState::InPlace(view),
            GdnRuntimeMode::Prefill if self.geometry.has_initial => {
                GdnRuntimeInitialState::Cache(view)
            }
            GdnRuntimeMode::Prefill => GdnRuntimeInitialState::Zero,
        }
    }
    pub fn final_recurrent_state(&self) -> GdnRuntimeCacheView {
        GdnRuntimeCacheView {
            offset: self.slot.recurrent_offset(),
            bytes: RECURRENT_STATE_BYTES,
        }
    }
}

/// Lazy chunk construction supports checked token counts without large host vectors.
#[derive(Clone, Debug)]
pub struct GdnRuntimePrefill {
    config: GdnRuntimeConfig,
    slot: GdnRuntimeCacheSlot,
    frontier: usize,
    tokens: usize,
    has_initial: bool,
    workspace: Arc<GdnRuntimeWorkspaceLayout>,
}
impl GdnRuntimePrefill {
    pub fn new(
        config: GdnRuntimeConfig,
        slot: GdnRuntimeCacheSlot,
        frontier: usize,
        tokens: usize,
        has_initial: bool,
    ) -> Result<Self> {
        if tokens == 0 || has_initial != (frontier != 0) || frontier.checked_add(tokens).is_none() {
            return Err(Error::Unsupported(
                "parameterized GDN runtime prefill span/state".into(),
            ));
        }
        Ok(Self {
            config,
            slot,
            frontier,
            tokens,
            has_initial,
            workspace: Arc::new(GdnRuntimeWorkspaceLayout::new(config)?),
        })
    }

    pub fn config(&self) -> GdnRuntimeConfig {
        self.config
    }
    pub fn slot(&self) -> GdnRuntimeCacheSlot {
        self.slot
    }
    pub fn frontier(&self) -> usize {
        self.frontier
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn resulting_frontier(&self) -> usize {
        self.frontier + self.tokens
    }
    pub fn chunk_count(&self) -> usize {
        self.tokens.div_ceil(PREFILL_ROWS)
    }
    pub fn workspace(&self) -> &GdnRuntimeWorkspaceLayout {
        &self.workspace
    }
    pub fn chunk(&self, index: usize) -> Result<GdnRuntimePlan> {
        if index >= self.chunk_count() {
            return Err(Error::Unsupported(
                "parameterized GDN runtime prefill chunk index".into(),
            ));
        }
        let consumed = index
            .checked_mul(PREFILL_ROWS)
            .ok_or_else(|| Error::Limit("parameterized GDN chunk offset overflow".into()))?;
        let frontier = self
            .frontier
            .checked_add(consumed)
            .ok_or_else(|| Error::Limit("parameterized GDN chunk frontier overflow".into()))?;
        let rows = (self.tokens - consumed).min(PREFILL_ROWS);
        GdnRuntimePlan::prefill_with_workspace(
            self.config,
            self.slot,
            frontier,
            rows,
            self.has_initial || index != 0,
            self.workspace.clone(),
        )
    }
    pub fn cached_decode(&self, frontier: usize) -> Result<GdnRuntimePlan> {
        GdnRuntimePlan::cached_with_workspace(
            self.config,
            self.slot,
            frontier,
            self.workspace.clone(),
        )
    }
}

fn runtime_controls(
    phase: GdnPhase,
    slot: GdnRuntimeCacheSlot,
    has_initial: bool,
) -> Result<Vec<GdnRuntimeControl>> {
    let values = metadata(phase, GdnCacheSlot::One)?;
    values
        .into_iter()
        .map(|(role, mut bytes)| {
            match role {
                GdnBuffer::CacheIndices => bytes = (slot.index() as i32).to_le_bytes().to_vec(),
                GdnBuffer::HasInitial => bytes = vec![u8::from(has_initial)],
                _ => {}
            }
            Ok(GdnRuntimeControl {
                role: role.into(),
                bytes,
            })
        })
        .collect()
}

fn build_stage_spec(
    config: GdnRuntimeConfig,
    phase: GdnPhase,
    stage: GdnStage,
) -> Result<LaunchSpec> {
    build_stage(config, phase, stage).map(|(spec, _, _)| spec)
}

fn build_stage(
    config: GdnRuntimeConfig,
    phase: GdnPhase,
    stage: GdnStage,
) -> Result<(
    LaunchSpec,
    assets::Asset,
    crate::cuda::kernel::StaticScalarBlock,
)> {
    let (asset, scalars, family_config) = match phase {
        GdnPhase::TextPrefill(_) => {
            let (asset, scalars) = match stage {
                GdnStage::Conv => {
                    let definition = parameterized_conv::prepare(
                        phase,
                        GdnCacheSlot::One,
                        config.geometry.conv(),
                    )?;
                    (definition.asset, definition.scalars)
                }
                GdnStage::Prep | GdnStage::Cumsum | GdnStage::Solve => {
                    let definition =
                        parameterized_prefill::prepare(phase, GdnCacheSlot::One, stage)?;
                    (definition.asset, definition.scalars)
                }
                GdnStage::Kkt | GdnStage::Recompute | GdnStage::H | GdnStage::Output => {
                    let definition = parameterized_prefill_variants::prepare(
                        config.kernels,
                        phase,
                        GdnCacheSlot::One,
                        stage,
                    )?;
                    (definition.asset, definition.scalars)
                }
                GdnStage::Gated => {
                    let definition = parameterized_gated::prepare(phase, GdnCacheSlot::One)?;
                    (definition.asset, definition.scalars)
                }
                GdnStage::Decode => {
                    return Err(Error::Unsupported(
                        "parameterized GDN runtime prefill stage".into(),
                    ));
                }
            };
            (asset, scalars, Some(config.kernels))
        }
        GdnPhase::CachedDecode => {
            let (asset, scalars) = match stage {
                GdnStage::Conv | GdnStage::Decode => {
                    let definition =
                        parameterized_cached::prepare(phase, GdnCacheSlot::One, stage)?;
                    (definition.asset, definition.scalars)
                }
                GdnStage::Gated => {
                    let definition = parameterized_gated::prepare(phase, GdnCacheSlot::One)?;
                    (definition.asset, definition.scalars)
                }
                _ => {
                    return Err(Error::Unsupported(
                        "parameterized GDN runtime cached stage".into(),
                    ));
                }
            };
            (asset, scalars, None)
        }
        _ => {
            return Err(Error::Unsupported("parameterized GDN runtime phase".into()));
        }
    };
    let spec = kernel_from_asset_with_scalars(
        phase,
        stage,
        true,
        asset,
        arithmetic::GdnArithmeticFamily::Text(family_config),
        scalars,
    )?;
    match stage {
        GdnStage::Gated => parameterized_gated::require_bindings(phase, scalars, &spec.bindings)?,
        GdnStage::Conv | GdnStage::Decode if phase == GdnPhase::CachedDecode => {
            parameterized_cached::require_bindings(phase, stage, scalars, &spec.bindings)?
        }
        _ => {}
    }
    Ok((spec, asset, scalars))
}

#[cfg(test)]
mod tests;
