//! Source-compiled SM86 BF16 primitives for an ordinary text backend.
//!
//! This layer defines arithmetic, exact pointer ABI, checked launch geometry,
//! and byte plans only. It allocates nothing, owns no queue, publishes no model
//! state, and does not confer numerical qualification.

use crate::cuda::kernel::{
    PinnedKernel, PreparedPinnedLaunch, StaticScalarBlock, StaticScalarSchema, StaticScalarType,
    StaticScalarValue,
};
use cubecl::{client::Client, server::Handle};
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
use zerogravity_core::{Error, Result};

const THREADS: u32 = 256;
pub const MAX_RUNTIME_ROWS: usize = 64;
pub const MAX_CONTEXT_CAPACITY: usize = 32_768;
/// Device status: runtime token or position exceeded its admitted extent.
pub const STATUS_OUT_OF_RANGE: u32 = 1;
/// Device status: an arithmetic input or output was non-finite.
pub const STATUS_NONFINITE: u32 = 2;
/// Device status: greedy logits contained a NaN, which is ignored for argmax.
pub const STATUS_NAN_LOGIT: u32 = 4;
const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;
const STATUS_BYTES: usize = 4;
const SOURCE_BODY: &str = include_str!("kernels.cu");

/// Qwen full-attention geometry. Dimensions are source data, not model IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub hidden_width: usize,
    pub intermediate_width: usize,
    pub vocabulary: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_width: usize,
    pub rotary_width: usize,
}

impl Geometry {
    pub fn validate(self) -> Result<Self> {
        if self.hidden_width == 0
            || self.hidden_width > 131_072
            || self.intermediate_width == 0
            || self.intermediate_width > 1_048_576
            || self.vocabulary == 0
            || self.vocabulary > 1_048_576
            || !(1..=64).contains(&self.query_heads)
            || !(1..=64).contains(&self.kv_heads)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.head_width != 256
            || self.rotary_width == 0
            || self.rotary_width > self.head_width
            || !self.rotary_width.is_multiple_of(2)
        {
            return Err(Error::Unsupported(
                "ordinary text Qwen BF16 geometry".into(),
            ));
        }
        for (name, value) in [
            ("query width", self.query_width()),
            ("KV width", self.kv_width()),
            (
                "embedding extent",
                checked_mul(self.vocabulary, self.hidden_width, "embedding")
                    .and_then(|n| checked_mul(n, BF16_BYTES, "embedding")),
            ),
        ] {
            value.map_err(|_| Error::Limit(format!("ordinary text {name} overflow")))?;
        }
        Ok(self)
    }

    pub fn query_width(self) -> Result<usize> {
        checked_mul(self.query_heads, self.head_width, "query width")
    }

    pub fn kv_width(self) -> Result<usize> {
        checked_mul(self.kv_heads, self.head_width, "KV width")
    }
}

/// Runtime token interval. The cache append and attention contract is
/// `[base, base + rows)`, with every row attending through its own new token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub rows: usize,
    pub base: usize,
    pub capacity: usize,
}

impl Step {
    pub fn new(rows: usize, base: usize, capacity: usize) -> Result<Self> {
        if !(1..=MAX_RUNTIME_ROWS).contains(&rows)
            || !(1..=MAX_CONTEXT_CAPACITY).contains(&capacity)
            || base.checked_add(rows).is_none_or(|end| end > capacity)
            || u32::try_from(base).is_err()
            || u32::try_from(capacity).is_err()
        {
            return Err(Error::Unsupported(
                "ordinary text rows/base/capacity interval".into(),
            ));
        }
        Ok(Self {
            rows,
            base,
            capacity,
        })
    }
}

/// Exact pointer order is the order returned by [`Definition::bindings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferRole {
    Positions,
    Query,
    Key,
    Value,
    InverseFrequency,
    TokenIds,
    EmbeddingTable,
    Input,
    Residual,
    Gamma,
    Gate,
    Up,
    KeyCache,
    ValueCache,
    Output,
    Logits,
    Maxima,
    Status,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    fn native(self) -> BufferIOAttr {
        match self {
            Self::Read => BufferIOAttr::ReadOnly,
            Self::Write => BufferIOAttr::WriteOnly,
            Self::ReadWrite => BufferIOAttr::ReadWrite,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferBinding {
    pub role: BufferRole,
    pub bytes: usize,
    pub access: Access,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primitive {
    Positions,
    Rope,
    Embedding,
    Rms,
    ResidualRms,
    SiluUp,
    SigmoidGate,
    Residual,
    KvAppend,
    Attention,
    Greedy,
}

impl Primitive {
    fn entry(self) -> &'static str {
        match self {
            Self::Positions => "zg_ot_positions_v1",
            Self::Rope => "zg_ot_rope_v1",
            Self::Embedding => "zg_ot_embedding_v1",
            Self::Rms => "zg_ot_rms_v1",
            Self::ResidualRms => "zg_ot_residual_rms_v1",
            Self::SiluUp => "zg_ot_silu_up_v1",
            Self::SigmoidGate => "zg_ot_sigmoid_gate_v1",
            Self::Residual => "zg_ot_residual_v1",
            Self::KvAppend => "zg_ot_kv_append_v1",
            Self::Attention => "zg_ot_attention_v1",
            Self::Greedy => "zg_ot_greedy_v1",
        }
    }
}

/// Complete source, ABI, static scalar block, and launch geometry.
#[derive(Clone)]
pub struct Definition {
    primitive: Primitive,
    source: Arc<str>,
    bindings: Arc<[BufferBinding]>,
    scalars: StaticScalarBlock,
    grid: [u32; 3],
    kernel: PinnedKernel,
}

impl Definition {
    fn new(
        primitive: Primitive,
        source: Arc<str>,
        bindings: Vec<BufferBinding>,
        scalars: StaticScalarBlock,
        grid: [u32; 3],
    ) -> Result<Self> {
        if bindings.is_empty() || bindings.iter().any(|binding| binding.bytes == 0) {
            return Err(Error::State("ordinary text empty ABI binding".into()));
        }
        let native = bindings
            .iter()
            .map(|binding| (binding.bytes, binding.access.native()))
            .collect::<Vec<_>>()
            .into();
        let kernel = PinnedKernel::new_with_static_scalars(
            primitive.entry(),
            source.clone(),
            THREADS,
            native,
            scalars,
        )?;
        kernel.require_typed_abi(
            primitive.entry(),
            bindings
                .iter()
                .map(|binding| (binding.bytes, binding.access.native())),
        )?;
        Ok(Self {
            primitive,
            source,
            bindings: bindings.into(),
            scalars,
            grid,
            kernel,
        })
    }

    pub fn primitive(&self) -> Primitive {
        self.primitive
    }

    pub fn entry(&self) -> &'static str {
        self.primitive.entry()
    }

    /// Exact source submitted to CubeCL, before its identity-only entry suffix.
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn threads(&self) -> u32 {
        THREADS
    }

    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn bindings(&self) -> &[BufferBinding] {
        &self.bindings
    }

    /// By-value words in CUDA formal order; u32/f32 values occupy low 32 bits.
    pub fn scalar_words(&self) -> &[u64] {
        self.scalars.words()
    }

    /// Validate exact service/extents and retain handles in an immutable launch.
    /// The existing physical owner remains responsible for access admission,
    /// ordering, completion, and quarantine.
    pub fn prepare(&self, client: &Client, handles: Vec<Handle>) -> Result<PreparedLaunch> {
        Ok(PreparedLaunch {
            inner: self.kernel.prepare(client, self.grid, handles)?,
            primitive: self.primitive,
        })
    }

    /// Compile/load on the already-owned queue; creates no client or allocator.
    #[cfg(target_os = "linux")]
    pub fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }
}

#[derive(Clone)]
pub struct PreparedLaunch {
    inner: PreparedPinnedLaunch,
    primitive: Primitive,
}

impl PreparedLaunch {
    pub fn primitive(&self) -> Primitive {
        self.primitive
    }

    pub fn handles(&self) -> &[Handle] {
        self.inner.handles()
    }

    pub fn launch(&self) -> Result<()> {
        self.inner.launch()
    }

    pub fn launch_on(&self, client: &Client) -> Result<()> {
        self.inner.launch_on(client)
    }
}

/// Byte requirements only. Fields are exact physical binding extents; this is
/// not an allocation request and intentionally does not choose aliasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryPlan {
    pub rows: usize,
    pub capacity: usize,
    pub positions_bytes: usize,
    pub token_ids_bytes: usize,
    pub hidden_batch_bytes: usize,
    pub intermediate_batch_bytes: usize,
    pub query_batch_bytes: usize,
    pub kv_batch_bytes: usize,
    pub embedding_table_bytes: usize,
    pub inverse_frequency_bytes: usize,
    pub key_cache_bytes: usize,
    pub value_cache_bytes: usize,
    pub logits_bytes: usize,
    pub greedy_tokens_bytes: usize,
    pub greedy_maxima_bytes: usize,
    pub status_bytes: usize,
    pub max_activation_bytes: usize,
}

impl MemoryPlan {
    pub fn kv_bytes_for_layers(self, layers: usize) -> Result<usize> {
        self.key_cache_bytes
            .checked_add(self.value_cache_bytes)
            .and_then(|bytes| bytes.checked_mul(layers))
            .ok_or_else(|| Error::Limit("ordinary text layered KV bytes overflow".into()))
    }
}

/// Immutable source family. Definitions are derived directly for each checked
/// rows/base/capacity interval.
#[derive(Clone)]
pub struct Program {
    geometry: Geometry,
    source: Arc<str>,
}

impl Program {
    pub fn new(geometry: Geometry) -> Result<Self> {
        let geometry = geometry.validate()?;
        let scale = 1.0f32 / (geometry.head_width as f32).sqrt();
        if !scale.is_finite() {
            return Err(Error::Limit("ordinary text attention scale".into()));
        }
        let source = format!(
            "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_HIDDEN_WIDTH {hidden}\n#define ZG_INTERMEDIATE_WIDTH {intermediate}\n#define ZG_VOCABULARY {vocabulary}\n#define ZG_QUERY_HEADS {query_heads}\n#define ZG_KV_HEADS {kv_heads}\n#define ZG_HEAD_DIM {head_width}\n#define ZG_ROTARY_WIDTH {rotary_width}\n#define ZG_ATTENTION_SCALE {scale:.9e}f\n{SOURCE_BODY}",
            hidden = geometry.hidden_width,
            intermediate = geometry.intermediate_width,
            vocabulary = geometry.vocabulary,
            query_heads = geometry.query_heads,
            kv_heads = geometry.kv_heads,
            head_width = geometry.head_width,
            rotary_width = geometry.rotary_width,
        );
        if source.len() > 2 * 1024 * 1024 || source.as_bytes().contains(&0) {
            return Err(Error::Limit("ordinary text CUDA source extent".into()));
        }
        Ok(Self {
            geometry,
            source: source.into(),
        })
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn memory_plan(&self, rows: usize, capacity: usize) -> Result<MemoryPlan> {
        Step::new(rows, 0, capacity)?;
        let g = self.geometry;
        let bf16 = |r, w, role| extent(r, w, BF16_BYTES, role);
        let u32s = |n, role| checked_mul(n, 4, role);
        let hidden = bf16(rows, g.hidden_width, "hidden batch")?;
        let intermediate = bf16(rows, g.intermediate_width, "intermediate batch")?;
        let query = bf16(rows, g.query_width()?, "query batch")?;
        let kv = bf16(rows, g.kv_width()?, "KV batch")?;
        let cache = bf16(capacity, g.kv_width()?, "KV cache")?;
        let embedding = bf16(g.vocabulary, g.hidden_width, "embedding table")?;
        let logits = extent(rows, g.vocabulary, F32_BYTES, "logits")?;
        Ok(MemoryPlan {
            rows,
            capacity,
            positions_bytes: u32s(rows, "positions")?,
            token_ids_bytes: u32s(rows, "token IDs")?,
            hidden_batch_bytes: hidden,
            intermediate_batch_bytes: intermediate,
            query_batch_bytes: query,
            kv_batch_bytes: kv,
            embedding_table_bytes: embedding,
            inverse_frequency_bytes: checked_mul(g.rotary_width / 2, 4, "inverse frequency")?,
            key_cache_bytes: cache,
            value_cache_bytes: cache,
            logits_bytes: logits,
            greedy_tokens_bytes: u32s(rows, "greedy tokens")?,
            greedy_maxima_bytes: u32s(rows, "greedy maxima")?,
            status_bytes: STATUS_BYTES,
            max_activation_bytes: *[hidden, intermediate, query, kv, logits]
                .iter()
                .max()
                .ok_or_else(|| Error::State("ordinary text activation plan empty".into()))?,
        })
    }

    pub fn positions(&self, step: Step) -> Result<Definition> {
        let step = Step::new(step.rows, step.base, step.capacity)?;
        self.definition(
            Primitive::Positions,
            vec![binding(BufferRole::Positions, step.rows * 4, Access::Write)],
            u32_words(&[step.rows, step.base, step.capacity])?,
            element_grid(step.rows)?,
        )
    }

    pub fn rope(&self, step: Step) -> Result<Definition> {
        let step = Step::new(step.rows, step.base, step.capacity)?;
        let plan = self.memory_plan(step.rows, step.capacity)?;
        let pairs = checked_mul(
            step.rows,
            checked_mul(
                self.geometry.query_heads.max(self.geometry.kv_heads),
                self.geometry.rotary_width / 2,
                "RoPE heads",
            )?,
            "RoPE pairs",
        )?;
        self.definition(
            Primitive::Rope,
            vec![
                binding(BufferRole::Query, plan.query_batch_bytes, Access::ReadWrite),
                binding(BufferRole::Key, plan.kv_batch_bytes, Access::ReadWrite),
                binding(BufferRole::Positions, plan.positions_bytes, Access::Read),
                binding(
                    BufferRole::InverseFrequency,
                    plan.inverse_frequency_bytes,
                    Access::Read,
                ),
                status(),
            ],
            u32_words(&[step.rows, step.capacity])?,
            element_grid(pairs)?,
        )
    }

    pub fn embedding(&self, rows: usize) -> Result<Definition> {
        let plan = self.memory_plan(rows, rows)?;
        self.definition(
            Primitive::Embedding,
            vec![
                binding(BufferRole::TokenIds, plan.token_ids_bytes, Access::Read),
                binding(
                    BufferRole::EmbeddingTable,
                    plan.embedding_table_bytes,
                    Access::Read,
                ),
                binding(BufferRole::Output, plan.hidden_batch_bytes, Access::Write),
                status(),
            ],
            u32_words(&[rows])?,
            element_grid(checked_mul(
                rows,
                self.geometry.hidden_width,
                "embedding grid",
            )?)?,
        )
    }

    pub fn rms(&self, rows: usize, epsilon: f32) -> Result<Definition> {
        self.rms_definition(Primitive::Rms, rows, epsilon)
    }

    pub fn residual_rms(&self, rows: usize, epsilon: f32) -> Result<Definition> {
        self.rms_definition(Primitive::ResidualRms, rows, epsilon)
    }

    fn rms_definition(
        &self,
        primitive: Primitive,
        rows: usize,
        epsilon: f32,
    ) -> Result<Definition> {
        require_rows(rows)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::Unsupported("ordinary text RMS epsilon".into()));
        }
        let bytes = extent(rows, self.geometry.hidden_width, BF16_BYTES, "RMS batch")?;
        let gamma = checked_mul(self.geometry.hidden_width, BF16_BYTES, "RMS gamma")?;
        let mut bindings = vec![binding(BufferRole::Input, bytes, Access::Read)];
        if primitive == Primitive::ResidualRms {
            bindings.push(binding(BufferRole::Residual, bytes, Access::ReadWrite));
        }
        bindings.extend([
            binding(BufferRole::Gamma, gamma, Access::Read),
            binding(BufferRole::Output, bytes, Access::Write),
            status(),
        ]);
        self.definition(
            primitive,
            bindings,
            words(&[
                (StaticScalarType::U32, StaticScalarValue::U32(rows as u64)),
                (
                    StaticScalarType::F32Bits,
                    StaticScalarValue::F32Bits(epsilon.to_bits()),
                ),
            ])?,
            [rows as u32, 1, 1],
        )
    }

    pub fn silu_up(&self, rows: usize) -> Result<Definition> {
        self.elementwise(
            Primitive::SiluUp,
            rows,
            self.geometry.intermediate_width,
            BufferRole::Gate,
            BufferRole::Up,
        )
    }

    pub fn sigmoid_gate(&self, rows: usize) -> Result<Definition> {
        self.elementwise(
            Primitive::SigmoidGate,
            rows,
            self.geometry.query_width()?,
            BufferRole::Input,
            BufferRole::Gate,
        )
    }

    pub fn residual(&self, rows: usize) -> Result<Definition> {
        self.elementwise(
            Primitive::Residual,
            rows,
            self.geometry.hidden_width,
            BufferRole::Input,
            BufferRole::Residual,
        )
    }

    fn elementwise(
        &self,
        primitive: Primitive,
        rows: usize,
        width: usize,
        first: BufferRole,
        second: BufferRole,
    ) -> Result<Definition> {
        require_rows(rows)?;
        let bytes = extent(rows, width, BF16_BYTES, "elementwise batch")?;
        self.definition(
            primitive,
            vec![
                binding(first, bytes, Access::Read),
                binding(second, bytes, Access::Read),
                binding(BufferRole::Output, bytes, Access::Write),
                status(),
            ],
            u32_words(&[rows])?,
            element_grid(checked_mul(rows, width, "elementwise grid")?)?,
        )
    }

    pub fn kv_append(&self, step: Step) -> Result<Definition> {
        let step = Step::new(step.rows, step.base, step.capacity)?;
        let plan = self.memory_plan(step.rows, step.capacity)?;
        self.definition(
            Primitive::KvAppend,
            vec![
                binding(BufferRole::Key, plan.kv_batch_bytes, Access::Read),
                binding(BufferRole::Value, plan.kv_batch_bytes, Access::Read),
                binding(
                    BufferRole::KeyCache,
                    plan.key_cache_bytes,
                    Access::ReadWrite,
                ),
                binding(
                    BufferRole::ValueCache,
                    plan.value_cache_bytes,
                    Access::ReadWrite,
                ),
                status(),
            ],
            u32_words(&[step.rows, step.base, step.capacity])?,
            element_grid(plan.kv_batch_bytes / BF16_BYTES)?,
        )
    }

    pub fn attention(&self, step: Step) -> Result<Definition> {
        let step = Step::new(step.rows, step.base, step.capacity)?;
        let plan = self.memory_plan(step.rows, step.capacity)?;
        self.definition(
            Primitive::Attention,
            vec![
                binding(BufferRole::Query, plan.query_batch_bytes, Access::Read),
                binding(BufferRole::KeyCache, plan.key_cache_bytes, Access::Read),
                binding(BufferRole::ValueCache, plan.value_cache_bytes, Access::Read),
                binding(BufferRole::Output, plan.query_batch_bytes, Access::Write),
                status(),
            ],
            u32_words(&[step.rows, step.base, step.capacity])?,
            [
                u32::try_from(checked_mul(
                    step.rows,
                    self.geometry.query_heads,
                    "attention grid",
                )?)
                .map_err(|_| Error::Limit("ordinary text attention grid u32".into()))?,
                1,
                1,
            ],
        )
    }

    pub fn greedy(&self, rows: usize) -> Result<Definition> {
        let plan = self.memory_plan(rows, rows)?;
        self.definition(
            Primitive::Greedy,
            vec![
                binding(BufferRole::Logits, plan.logits_bytes, Access::Read),
                binding(
                    BufferRole::TokenIds,
                    plan.greedy_tokens_bytes,
                    Access::Write,
                ),
                binding(BufferRole::Maxima, plan.greedy_maxima_bytes, Access::Write),
                status(),
            ],
            u32_words(&[rows])?,
            [rows as u32, 1, 1],
        )
    }

    pub fn definitions(&self, step: Step, epsilon: f32) -> Result<Vec<Definition>> {
        Ok(vec![
            self.positions(step)?,
            self.rope(step)?,
            self.embedding(step.rows)?,
            self.rms(step.rows, epsilon)?,
            self.residual_rms(step.rows, epsilon)?,
            self.silu_up(step.rows)?,
            self.sigmoid_gate(step.rows)?,
            self.residual(step.rows)?,
            self.kv_append(step)?,
            self.attention(step)?,
            self.greedy(step.rows)?,
        ])
    }

    fn definition(
        &self,
        primitive: Primitive,
        bindings: Vec<BufferBinding>,
        scalars: StaticScalarBlock,
        grid: [u32; 3],
    ) -> Result<Definition> {
        Definition::new(primitive, self.source.clone(), bindings, scalars, grid)
    }
}

fn binding(role: BufferRole, bytes: usize, access: Access) -> BufferBinding {
    BufferBinding {
        role,
        bytes,
        access,
    }
}

fn status() -> BufferBinding {
    binding(BufferRole::Status, STATUS_BYTES, Access::ReadWrite)
}

fn require_rows(rows: usize) -> Result<()> {
    if !(1..=MAX_RUNTIME_ROWS).contains(&rows) {
        return Err(Error::Unsupported("ordinary text runtime rows".into()));
    }
    Ok(())
}

fn checked_mul(a: usize, b: usize, role: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit(format!("ordinary text {role} overflow")))
}

fn extent(rows: usize, width: usize, element_bytes: usize, role: &str) -> Result<usize> {
    checked_mul(rows, width, role).and_then(|n| checked_mul(n, element_bytes, role))
}

fn element_grid(elements: usize) -> Result<[u32; 3]> {
    let blocks = elements
        .checked_add(THREADS as usize - 1)
        .map(|n| n / THREADS as usize)
        .ok_or_else(|| Error::Limit("ordinary text launch grid overflow".into()))?;
    Ok([
        u32::try_from(blocks)
            .map_err(|_| Error::Limit("ordinary text launch grid exceeds u32".into()))?,
        1,
        1,
    ])
}

fn u32_words(values: &[usize]) -> Result<StaticScalarBlock> {
    let mut typed = Vec::with_capacity(values.len());
    for &value in values {
        let value = u32::try_from(value)
            .map_err(|_| Error::Limit("ordinary text u32 scalar overflow".into()))?;
        typed.push((
            StaticScalarType::U32,
            StaticScalarValue::U32(u64::from(value)),
        ));
    }
    words(&typed)
}

fn words(values: &[(StaticScalarType, StaticScalarValue)]) -> Result<StaticScalarBlock> {
    let fields = values
        .iter()
        .enumerate()
        .map(|(index, (ty, _))| (*ty, index * 8))
        .collect::<Vec<_>>();
    let schema = StaticScalarSchema::new(&fields, values.len() * 8, 8)?;
    let payload = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    StaticScalarBlock::new(schema, &payload)
}

#[cfg(test)]
mod tests;
