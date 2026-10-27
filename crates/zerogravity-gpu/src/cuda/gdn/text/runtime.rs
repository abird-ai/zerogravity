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
        if !(1..CACHE_SLOTS).contains(&index) {
            return Err(Error::Unsupported(
                "parameterized GDN runtime non-null cache slot".into(),
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
    RecurrentState,
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
        InitialState | FinalState => GdnRuntimeBufferOwnership::RecurrentState,
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
        // Copy-only source/destination extents are part of the executable layout.
        for role in [
            GdnBuffer::PackedBa,
            GdnBuffer::AContiguous,
            GdnBuffer::BContiguous,
            GdnBuffer::ZCompact,
            GdnBuffer::InitialState,
            GdnBuffer::FinalState,
            GdnBuffer::Inverse,
        ] {
            roles
                .entry(role)
                .and_modify(|bytes: &mut usize| *bytes = (*bytes).max(role.bytes(phase)))
                .or_insert_with(|| role.bytes(phase));
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
}
impl fmt::Debug for GdnRuntimeStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.identity.fmt(f)
    }
}

/// One ordered operation in the complete executable schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeScheduleKind {
    PackA,
    PackB,
    PackZ,
    CopyQkvPrefix,
    Conv,
    Prep,
    Cumsum,
    Kkt,
    ZeroInverse,
    Solve,
    Recompute,
    ZeroInitialState,
    GatherInitialState,
    State,
    ScatterFinalState,
    Output,
    Gated,
    Decode,
}
impl From<GdnRuntimeStageKind> for GdnRuntimeScheduleKind {
    fn from(value: GdnRuntimeStageKind) -> Self {
        match value {
            GdnRuntimeStageKind::Conv => Self::Conv,
            GdnRuntimeStageKind::Prep => Self::Prep,
            GdnRuntimeStageKind::Cumsum => Self::Cumsum,
            GdnRuntimeStageKind::Kkt => Self::Kkt,
            GdnRuntimeStageKind::Solve => Self::Solve,
            GdnRuntimeStageKind::Recompute => Self::Recompute,
            GdnRuntimeStageKind::State => Self::State,
            GdnRuntimeStageKind::Output => Self::Output,
            GdnRuntimeStageKind::Gated => Self::Gated,
            GdnRuntimeStageKind::Decode => Self::Decode,
        }
    }
}

#[derive(Clone)]
enum GdnRuntimeScheduleSource {
    Static(&'static str),
    Owned(Arc<str>),
}
impl GdnRuntimeScheduleSource {
    fn as_str(&self) -> &str {
        match self {
            Self::Static(source) => source,
            Self::Owned(source) => source,
        }
    }
}

/// Public schedule identity; executable binding remains crate-private.
#[derive(Clone)]
pub struct GdnRuntimeScheduleStep {
    kind: GdnRuntimeScheduleKind,
    entry: &'static str,
    source: GdnRuntimeScheduleSource,
    threads: u32,
    shared_bytes: usize,
    grid: [u32; 3],
    bindings: Vec<GdnRuntimeBinding>,
    kernel: crate::cuda::kernel::PinnedKernel,
}
impl GdnRuntimeScheduleStep {
    pub fn kind(&self) -> GdnRuntimeScheduleKind {
        self.kind
    }
    pub fn entry(&self) -> &'static str {
        self.entry
    }
    pub fn source(&self) -> &str {
        self.source.as_str()
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

    /// Caller must retain admitted leases and its physical flight through
    /// completion or quarantine. This grants no completion/publication right.
    pub(crate) fn prepare_launch(
        &self,
        client: &cubecl::client::Client,
        handles: Vec<cubecl::server::Handle>,
    ) -> Result<GdnRuntimeLaunch> {
        Ok(GdnRuntimeLaunch {
            launch: self.kernel.prepare(client, self.grid, handles)?,
        })
    }
}
impl fmt::Debug for GdnRuntimeScheduleStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GdnRuntimeScheduleStep")
            .field("kind", &self.kind)
            .field("entry", &self.entry)
            .field("source_bytes", &self.source.as_str().len())
            .field("threads", &self.threads)
            .field("shared_bytes", &self.shared_bytes)
            .field("grid", &self.grid)
            .field("bindings", &self.bindings)
            .finish()
    }
}

/// Bound low-level launch. It cannot complete or publish caller state.
#[derive(Clone)]
pub(crate) struct GdnRuntimeLaunch {
    launch: crate::cuda::kernel::PreparedPinnedLaunch,
}
impl GdnRuntimeLaunch {
    #[allow(dead_code)]
    pub(crate) fn launch(&self) -> Result<()> {
        self.launch.launch()
    }
}

/// Ordered launches retained by, and executable only under, the caller's flight.
#[derive(Clone)]
pub(crate) struct GdnRuntimeLaunchSchedule {
    launches: Vec<GdnRuntimeLaunch>,
}
impl GdnRuntimeLaunchSchedule {
    #[allow(dead_code)]
    pub(crate) fn launch(&self) -> Result<usize> {
        for launch in &self.launches {
            launch.launch()?;
        }
        Ok(self.launches.len())
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRuntimeStateKind {
    Convolution,
    RecurrentInitial,
    RecurrentFinal,
    RecurrentCache,
}

/// Exact state role and suballocation. Different roles require different owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnRuntimeStateView {
    role: GdnRuntimeBuffer,
    offset: usize,
    bytes: usize,
    kind: GdnRuntimeStateKind,
}
impl GdnRuntimeStateView {
    pub fn role(self) -> GdnRuntimeBuffer {
        self.role
    }
    pub fn offset(self) -> usize {
        self.offset
    }
    pub fn bytes(self) -> usize {
        self.bytes
    }
    pub fn kind(self) -> GdnRuntimeStateKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnRuntimeInitialState {
    Zero {
        destination: GdnRuntimeStateView,
    },
    Copy {
        source: GdnRuntimeStateView,
        destination: GdnRuntimeStateView,
    },
    InPlace {
        state: GdnRuntimeStateView,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnRuntimeFinalState {
    Scatter {
        source: GdnRuntimeStateView,
        destination: GdnRuntimeStateView,
    },
    InPlace {
        state: GdnRuntimeStateView,
    },
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
    schedule: Vec<GdnRuntimeScheduleStep>,
    initial_state: GdnRuntimeInitialState,
    final_state: GdnRuntimeFinalState,
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
        let initial_state = runtime_initial_state(slot, geometry);
        let final_state = runtime_final_state(slot, geometry);
        let schedule = build_schedule(phase, geometry, slot, &stages)?;
        let result = Self {
            config,
            slot,
            geometry,
            workspace,
            controls,
            stages,
            schedule,
            initial_state,
            final_state,
        };
        result.require_schedule()?;
        Ok(result)
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
    pub fn schedule(&self) -> &[GdnRuntimeScheduleStep] {
        &self.schedule
    }
    pub fn convolution_state(&self) -> GdnRuntimeStateView {
        GdnRuntimeStateView {
            role: GdnRuntimeBuffer::CacheArena,
            offset: self.slot.convolution_offset(),
            bytes: CONV_STATE_BYTES,
            kind: GdnRuntimeStateKind::Convolution,
        }
    }
    pub fn initial_recurrent_state(&self) -> GdnRuntimeInitialState {
        self.initial_state
    }
    pub fn final_recurrent_state(&self) -> GdnRuntimeFinalState {
        self.final_state
    }

    fn require_schedule(&self) -> Result<()> {
        require_schedule(self)
    }

    /// Bind caller-owned full allocation handles to exact operation views.
    /// Initial/final/cache state must be physically distinct allocations.
    #[allow(dead_code)]
    pub(crate) fn bind_schedule_handles(
        &self,
        handles: &BTreeMap<GdnRuntimeBuffer, cubecl::server::Handle>,
    ) -> Result<Vec<Vec<cubecl::server::Handle>>> {
        self.require_schedule()?;
        if self.geometry.mode == GdnRuntimeMode::Prefill {
            let state_handle = |role| {
                handles
                    .get(&role)
                    .ok_or_else(|| Error::State("GDN runtime state owner absent".into()))
            };
            let cache = state_handle(GdnRuntimeBuffer::CacheArena)?;
            let initial = state_handle(GdnRuntimeBuffer::InitialState)?;
            let final_state = state_handle(GdnRuntimeBuffer::FinalState)?;
            let cache_id = cache.memory.descriptor().id;
            let initial_id = initial.memory.descriptor().id;
            let final_id = final_state.memory.descriptor().id;
            if cache_id == initial_id || cache_id == final_id || initial_id == final_id {
                return Err(Error::State(
                    "GDN runtime cache/initial/final state allocations alias".into(),
                ));
            }
        }
        self.schedule
            .iter()
            .map(|step| {
                step.bindings
                    .iter()
                    .map(|binding| {
                        let layout = self
                            .workspace
                            .buffers
                            .iter()
                            .find(|layout| layout.role == binding.role)
                            .ok_or_else(|| Error::State("GDN runtime bound role absent".into()))?;
                        let handle = handles.get(&binding.role).ok_or_else(|| {
                            Error::State("GDN runtime allocation handle absent".into())
                        })?;
                        let available = usize::try_from(handle.size_in_used()).map_err(|_| {
                            Error::Limit("GDN runtime allocation handle extent".into())
                        })?;
                        let end = binding
                            .offset
                            .checked_add(binding.bytes)
                            .filter(|end| *end <= layout.bytes && *end <= available)
                            .ok_or_else(|| Error::State("GDN runtime handle view extent".into()))?;
                        Ok(handle
                            .clone()
                            .offset_start(binding.offset as u64)
                            .offset_end((available - end) as u64))
                    })
                    .collect()
            })
            .collect()
    }

    /// Low-level preparation only. The caller retains the admitted flight and
    /// every source/state owner; this capsule has no completion/publication API.
    #[allow(dead_code)]
    pub(crate) fn prepare_launches(
        &self,
        client: &cubecl::client::Client,
        handles: &BTreeMap<GdnRuntimeBuffer, cubecl::server::Handle>,
    ) -> Result<GdnRuntimeLaunchSchedule> {
        let handles = self.bind_schedule_handles(handles)?;
        let launches = self
            .schedule
            .iter()
            .zip(handles)
            .map(|(step, handles)| step.prepare_launch(client, handles))
            .collect::<Result<Vec<_>>>()?;
        Ok(GdnRuntimeLaunchSchedule { launches })
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

fn cache_recurrent_view(slot: GdnRuntimeCacheSlot) -> GdnRuntimeStateView {
    GdnRuntimeStateView {
        role: GdnRuntimeBuffer::CacheArena,
        offset: slot.recurrent_offset(),
        bytes: RECURRENT_STATE_BYTES,
        kind: GdnRuntimeStateKind::RecurrentCache,
    }
}

fn local_state_view(role: GdnRuntimeBuffer, kind: GdnRuntimeStateKind) -> GdnRuntimeStateView {
    GdnRuntimeStateView {
        role,
        offset: 0,
        bytes: RECURRENT_STATE_BYTES,
        kind,
    }
}

fn runtime_initial_state(
    slot: GdnRuntimeCacheSlot,
    geometry: GdnRuntimeGeometry,
) -> GdnRuntimeInitialState {
    match geometry.mode {
        GdnRuntimeMode::CachedDecode => GdnRuntimeInitialState::InPlace {
            state: cache_recurrent_view(slot),
        },
        GdnRuntimeMode::Prefill if geometry.has_initial => GdnRuntimeInitialState::Copy {
            source: cache_recurrent_view(slot),
            destination: local_state_view(
                GdnRuntimeBuffer::InitialState,
                GdnRuntimeStateKind::RecurrentInitial,
            ),
        },
        GdnRuntimeMode::Prefill => GdnRuntimeInitialState::Zero {
            destination: local_state_view(
                GdnRuntimeBuffer::InitialState,
                GdnRuntimeStateKind::RecurrentInitial,
            ),
        },
    }
}

fn runtime_final_state(
    slot: GdnRuntimeCacheSlot,
    geometry: GdnRuntimeGeometry,
) -> GdnRuntimeFinalState {
    match geometry.mode {
        GdnRuntimeMode::CachedDecode => GdnRuntimeFinalState::InPlace {
            state: cache_recurrent_view(slot),
        },
        GdnRuntimeMode::Prefill => GdnRuntimeFinalState::Scatter {
            source: local_state_view(
                GdnRuntimeBuffer::FinalState,
                GdnRuntimeStateKind::RecurrentFinal,
            ),
            destination: cache_recurrent_view(slot),
        },
    }
}

fn stage_step(stage: &GdnRuntimeStage) -> GdnRuntimeScheduleStep {
    let identity = stage.identity();
    GdnRuntimeScheduleStep {
        kind: identity.kind().into(),
        entry: identity.entry(),
        source: GdnRuntimeScheduleSource::Static(identity.source()),
        threads: identity.threads(),
        shared_bytes: identity.shared_bytes(),
        grid: identity.grid(),
        bindings: identity.bindings().to_vec(),
        kernel: stage.spec.kernel.clone(),
    }
}

#[derive(Clone, Copy)]
struct RuntimeCopyGeometry {
    source: Option<GdnRuntimeBuffer>,
    destination: GdnRuntimeBuffer,
    rows: usize,
    words: usize,
    source_stride: usize,
    destination_stride: usize,
    source_offset: usize,
    destination_offset: usize,
}

fn runtime_copy_geometry(
    phase: GdnPhase,
    slot: GdnRuntimeCacheSlot,
    kind: GdnRuntimeScheduleKind,
) -> Result<RuntimeCopyGeometry> {
    use GdnRuntimeBuffer::*;
    let rows = phase.tokens();
    let value = match kind {
        GdnRuntimeScheduleKind::PackA | GdnRuntimeScheduleKind::PackB if rows > 1 => {
            RuntimeCopyGeometry {
                source: Some(PackedBa),
                destination: if kind == GdnRuntimeScheduleKind::PackA {
                    AContiguous
                } else {
                    BContiguous
                },
                rows,
                words: 8,
                source_stride: 16,
                destination_stride: 8,
                source_offset: if kind == GdnRuntimeScheduleKind::PackA {
                    8
                } else {
                    0
                },
                destination_offset: 0,
            }
        }
        GdnRuntimeScheduleKind::PackZ => RuntimeCopyGeometry {
            source: Some(PackedQkvz),
            destination: ZCompact,
            rows,
            words: 1024,
            source_stride: 4096,
            destination_stride: 1024,
            source_offset: 3072,
            destination_offset: 0,
        },
        GdnRuntimeScheduleKind::CopyQkvPrefix if phase == GdnPhase::CachedDecode => {
            RuntimeCopyGeometry {
                source: Some(PackedQkvz),
                destination: ConvOutput,
                rows: 1,
                words: 3072,
                source_stride: 4096,
                destination_stride: 3072,
                source_offset: 0,
                destination_offset: 0,
            }
        }
        GdnRuntimeScheduleKind::ZeroInverse if phase != GdnPhase::CachedDecode => {
            RuntimeCopyGeometry {
                source: None,
                destination: Inverse,
                rows: 1,
                words: GdnBuffer::Inverse.bytes(phase) / 4,
                source_stride: 0,
                destination_stride: GdnBuffer::Inverse.bytes(phase) / 4,
                source_offset: 0,
                destination_offset: 0,
            }
        }
        GdnRuntimeScheduleKind::ZeroInitialState if phase != GdnPhase::CachedDecode => {
            RuntimeCopyGeometry {
                source: None,
                destination: InitialState,
                rows: 1,
                words: RECURRENT_STATE_BYTES / 4,
                source_stride: 0,
                destination_stride: RECURRENT_STATE_BYTES / 4,
                source_offset: 0,
                destination_offset: 0,
            }
        }
        GdnRuntimeScheduleKind::GatherInitialState if phase != GdnPhase::CachedDecode => {
            RuntimeCopyGeometry {
                source: Some(CacheArena),
                destination: InitialState,
                rows: 1,
                words: RECURRENT_STATE_BYTES / 4,
                source_stride: RECURRENT_STATE_BYTES / 4,
                destination_stride: RECURRENT_STATE_BYTES / 4,
                source_offset: slot.recurrent_offset() / 4,
                destination_offset: 0,
            }
        }
        GdnRuntimeScheduleKind::ScatterFinalState if phase != GdnPhase::CachedDecode => {
            RuntimeCopyGeometry {
                source: Some(FinalState),
                destination: CacheArena,
                rows: 1,
                words: RECURRENT_STATE_BYTES / 4,
                source_stride: RECURRENT_STATE_BYTES / 4,
                destination_stride: RECURRENT_STATE_BYTES / 4,
                source_offset: 0,
                destination_offset: slot.recurrent_offset() / 4,
            }
        }
        _ => {
            return Err(Error::Unsupported(
                "parameterized GDN runtime copy geometry".into(),
            ));
        }
    };
    Ok(value)
}

fn copy_binding(
    role: GdnRuntimeBuffer,
    offset_words: usize,
    rows: usize,
    stride_words: usize,
    words: usize,
    access: GdnRuntimeAccess,
) -> Result<GdnRuntimeBinding> {
    let offset = offset_words
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("GDN runtime copy offset overflow".into()))?;
    let span_words = (rows - 1)
        .checked_mul(stride_words)
        .and_then(|n| n.checked_add(words))
        .ok_or_else(|| Error::Limit("GDN runtime copy span overflow".into()))?;
    let bytes = span_words
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("GDN runtime copy bytes overflow".into()))?;
    Ok(GdnRuntimeBinding {
        role,
        offset,
        bytes,
        access,
    })
}

fn build_copy_step(
    phase: GdnPhase,
    slot: GdnRuntimeCacheSlot,
    kind: GdnRuntimeScheduleKind,
) -> Result<GdnRuntimeScheduleStep> {
    let geometry = runtime_copy_geometry(phase, slot, kind)?;
    let mut bindings = Vec::with_capacity(2);
    if let Some(source) = geometry.source {
        bindings.push(copy_binding(
            source,
            geometry.source_offset,
            geometry.rows,
            geometry.source_stride,
            geometry.words,
            GdnRuntimeAccess::ReadOnly,
        )?);
    }
    bindings.push(copy_binding(
        geometry.destination,
        geometry.destination_offset,
        geometry.rows,
        geometry.destination_stride,
        geometry.words,
        GdnRuntimeAccess::WriteOnly,
    )?);
    let entry = match kind {
        GdnRuntimeScheduleKind::PackA => "zg_gdn_runtime_pack_a",
        GdnRuntimeScheduleKind::PackB => "zg_gdn_runtime_pack_b",
        GdnRuntimeScheduleKind::PackZ => "zg_gdn_runtime_pack_z",
        GdnRuntimeScheduleKind::CopyQkvPrefix => "zg_gdn_runtime_copy_qkv_prefix",
        GdnRuntimeScheduleKind::ZeroInverse => "zg_gdn_runtime_zero_inverse",
        GdnRuntimeScheduleKind::ZeroInitialState => "zg_gdn_runtime_zero_initial",
        GdnRuntimeScheduleKind::GatherInitialState => "zg_gdn_runtime_gather_initial",
        GdnRuntimeScheduleKind::ScatterFinalState => "zg_gdn_runtime_scatter_final",
        _ => {
            return Err(Error::Unsupported(
                "parameterized GDN runtime copy operation".into(),
            ));
        }
    };
    let total = geometry
        .rows
        .checked_mul(geometry.words)
        .ok_or_else(|| Error::Limit("GDN runtime copy launch overflow".into()))?;
    let total_u32 = u32::try_from(total)
        .map_err(|_| Error::Limit("GDN runtime copy launch exceeds u32".into()))?;
    let (parameters, statement) = if geometry.source.is_some() {
        (
            "const unsigned int* source, unsigned int* destination",
            format!(
                "unsigned int row = index / {words}u; unsigned int col = index % {words}u; destination[row * {destination_stride}u + col] = source[row * {source_stride}u + col];",
                words = geometry.words,
                destination_stride = geometry.destination_stride,
                source_stride = geometry.source_stride,
            ),
        )
    } else {
        (
            "unsigned int* destination",
            "destination[index] = 0u;".into(),
        )
    };
    let source: Arc<str> = format!(
        "// cubecl.cuda.dynamic_shared_bytes=0\n#if __CUDA_ARCH__ != 860\n#error Closed SM86 GDN runtime copy\n#endif\nextern \"C\" __global__ __launch_bounds__(128) void {entry}({parameters}) {{ unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; if (index < {total_u32}u) {{ {statement} }} }}\n"
    )
    .into();
    let kernel = crate::cuda::kernel::PinnedKernel::new(
        entry,
        source.clone(),
        128,
        bindings
            .iter()
            .map(|binding| {
                (
                    binding.bytes,
                    match binding.access {
                        GdnRuntimeAccess::ReadOnly => BufferIOAttr::ReadOnly,
                        GdnRuntimeAccess::WriteOnly => BufferIOAttr::WriteOnly,
                        GdnRuntimeAccess::ReadWrite => BufferIOAttr::ReadWrite,
                        GdnRuntimeAccess::Dead => BufferIOAttr::Dead,
                    },
                )
            })
            .collect::<Vec<_>>()
            .into(),
    )?;
    Ok(GdnRuntimeScheduleStep {
        kind,
        entry,
        source: GdnRuntimeScheduleSource::Owned(source),
        threads: 128,
        shared_bytes: 0,
        grid: [total_u32.div_ceil(128), 1, 1],
        bindings,
        kernel,
    })
}

fn build_schedule(
    phase: GdnPhase,
    geometry: GdnRuntimeGeometry,
    slot: GdnRuntimeCacheSlot,
    stages: &[GdnRuntimeStage],
) -> Result<Vec<GdnRuntimeScheduleStep>> {
    let mut result = Vec::new();
    if geometry.mode == GdnRuntimeMode::CachedDecode {
        result.push(build_copy_step(
            phase,
            slot,
            GdnRuntimeScheduleKind::CopyQkvPrefix,
        )?);
        result.push(build_copy_step(phase, slot, GdnRuntimeScheduleKind::PackZ)?);
    } else {
        if geometry.rows > 1 {
            result.push(build_copy_step(phase, slot, GdnRuntimeScheduleKind::PackA)?);
            result.push(build_copy_step(phase, slot, GdnRuntimeScheduleKind::PackB)?);
        }
        result.push(build_copy_step(phase, slot, GdnRuntimeScheduleKind::PackZ)?);
    }
    for stage in stages {
        match stage.identity.kind {
            GdnRuntimeStageKind::Solve => result.push(build_copy_step(
                phase,
                slot,
                GdnRuntimeScheduleKind::ZeroInverse,
            )?),
            GdnRuntimeStageKind::State => result.push(build_copy_step(
                phase,
                slot,
                if geometry.has_initial {
                    GdnRuntimeScheduleKind::GatherInitialState
                } else {
                    GdnRuntimeScheduleKind::ZeroInitialState
                },
            )?),
            _ => {}
        }
        result.push(stage_step(stage));
        if stage.identity.kind == GdnRuntimeStageKind::State {
            result.push(build_copy_step(
                phase,
                slot,
                GdnRuntimeScheduleKind::ScatterFinalState,
            )?);
        }
    }
    Ok(result)
}

fn expected_schedule_kinds(geometry: GdnRuntimeGeometry) -> Vec<GdnRuntimeScheduleKind> {
    use GdnRuntimeScheduleKind::*;
    if geometry.mode == GdnRuntimeMode::CachedDecode {
        return vec![CopyQkvPrefix, PackZ, Conv, Decode, Gated];
    }
    let mut kinds = Vec::new();
    if geometry.rows > 1 {
        kinds.extend([PackA, PackB]);
    }
    kinds.extend([
        PackZ,
        Conv,
        Prep,
        Cumsum,
        Kkt,
        ZeroInverse,
        Solve,
        Recompute,
        if geometry.has_initial {
            GatherInitialState
        } else {
            ZeroInitialState
        },
        State,
        ScatterFinalState,
        Output,
        Gated,
    ]);
    kinds
}

fn require_schedule(plan: &GdnRuntimePlan) -> Result<()> {
    let phase = match plan.geometry.mode {
        GdnRuntimeMode::Prefill => GdnPhase::TextPrefill(plan.geometry.rows),
        GdnRuntimeMode::CachedDecode => GdnPhase::CachedDecode,
    };
    let actual = plan
        .schedule
        .iter()
        .map(|step| step.kind)
        .collect::<Vec<_>>();
    if actual != expected_schedule_kinds(plan.geometry) {
        return Err(Error::State(
            "GDN runtime incomplete or reordered schedule".into(),
        ));
    }
    for step in &plan.schedule {
        for binding in &step.bindings {
            let available = plan.workspace.bytes(binding.role).ok_or_else(|| {
                Error::State("GDN runtime schedule role absent from workspace".into())
            })?;
            if binding
                .offset
                .checked_add(binding.bytes)
                .is_none_or(|end| end > available)
            {
                return Err(Error::State(format!(
                    "GDN runtime schedule {:?}/{:?} binding {}+{} exceeds {}",
                    step.kind, binding.role, binding.offset, binding.bytes, available
                )));
            }
        }
        if matches!(
            step.kind,
            GdnRuntimeScheduleKind::PackA
                | GdnRuntimeScheduleKind::PackB
                | GdnRuntimeScheduleKind::PackZ
                | GdnRuntimeScheduleKind::CopyQkvPrefix
                | GdnRuntimeScheduleKind::ZeroInverse
                | GdnRuntimeScheduleKind::ZeroInitialState
                | GdnRuntimeScheduleKind::GatherInitialState
                | GdnRuntimeScheduleKind::ScatterFinalState
        ) {
            let expected = build_copy_step(phase, plan.slot, step.kind)?;
            if step.bindings != expected.bindings {
                return Err(Error::State("GDN runtime copy access contract".into()));
            }
        } else {
            let stage_kind = plan
                .stages
                .iter()
                .find(|stage| GdnRuntimeScheduleKind::from(stage.identity.kind) == step.kind)
                .ok_or_else(|| Error::State("GDN runtime scheduled stage absent".into()))?;
            if step.bindings != stage_kind.identity.bindings {
                return Err(Error::State("GDN runtime stage access contract".into()));
            }
        }
    }
    let initial_layout = plan
        .workspace
        .buffers
        .iter()
        .find(|layout| layout.role == GdnRuntimeBuffer::InitialState);
    let final_layout = plan
        .workspace
        .buffers
        .iter()
        .find(|layout| layout.role == GdnRuntimeBuffer::FinalState);
    match (initial_layout, final_layout) {
        (
            Some(GdnRuntimeBufferLayout {
                ownership: GdnRuntimeBufferOwnership::RecurrentState,
                ..
            }),
            Some(GdnRuntimeBufferLayout {
                ownership: GdnRuntimeBufferOwnership::RecurrentState,
                ..
            }),
        ) => {}
        _ => return Err(Error::State("GDN runtime recurrent ownership".into())),
    }
    if plan.initial_state != runtime_initial_state(plan.slot, plan.geometry)
        || plan.final_state != runtime_final_state(plan.slot, plan.geometry)
    {
        return Err(Error::State("GDN runtime state view mutation".into()));
    }
    match plan.initial_state {
        GdnRuntimeInitialState::Zero { destination }
            if destination.role == GdnRuntimeBuffer::InitialState
                && destination.kind == GdnRuntimeStateKind::RecurrentInitial => {}
        GdnRuntimeInitialState::Copy {
            source,
            destination,
        } if source.role == GdnRuntimeBuffer::CacheArena
            && destination.role == GdnRuntimeBuffer::InitialState
            && source.role != destination.role
            && source.kind == GdnRuntimeStateKind::RecurrentCache
            && destination.kind == GdnRuntimeStateKind::RecurrentInitial => {}
        GdnRuntimeInitialState::InPlace { state }
            if plan.geometry.mode == GdnRuntimeMode::CachedDecode
                && state.role == GdnRuntimeBuffer::CacheArena
                && state.kind == GdnRuntimeStateKind::RecurrentCache => {}
        _ => {
            return Err(Error::State(
                "GDN runtime initial-state ownership/alias".into(),
            ));
        }
    }
    match plan.final_state {
        GdnRuntimeFinalState::Scatter {
            source,
            destination,
        } if plan.geometry.mode == GdnRuntimeMode::Prefill
            && source.role == GdnRuntimeBuffer::FinalState
            && destination.role == GdnRuntimeBuffer::CacheArena
            && source.role != destination.role
            && source.kind == GdnRuntimeStateKind::RecurrentFinal
            && destination.kind == GdnRuntimeStateKind::RecurrentCache => {}
        GdnRuntimeFinalState::InPlace { state }
            if plan.geometry.mode == GdnRuntimeMode::CachedDecode
                && state.role == GdnRuntimeBuffer::CacheArena
                && state.kind == GdnRuntimeStateKind::RecurrentCache => {}
        _ => {
            return Err(Error::State(
                "GDN runtime final-state ownership/alias".into(),
            ));
        }
    }
    if let (
        GdnRuntimeInitialState::Copy { destination, .. }
        | GdnRuntimeInitialState::Zero { destination },
        GdnRuntimeFinalState::Scatter { source, .. },
    ) = (plan.initial_state, plan.final_state)
        && destination.role == source.role
    {
        return Err(Error::State(
            "GDN runtime initial/final recurrent allocations alias".into(),
        ));
    }
    Ok(())
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
