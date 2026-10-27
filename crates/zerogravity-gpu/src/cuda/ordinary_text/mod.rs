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
use cubecl_common::device::ServiceId;
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
use zerogravity_core::{Error, Result};

const THREADS: u32 = 256;
pub const MAX_RUNTIME_ROWS: usize = 64;
pub const MAX_CONTEXT_CAPACITY: usize = 262_144;
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
    /// Declared Qwen3.5 0.8B ordinary-text dimensions. Geometry alone
    /// grants no model, source, backend, or numerical authority.
    pub const fn qwen35_08() -> Self {
        Self {
            hidden_width: 1024,
            intermediate_width: 3584,
            vocabulary: 248_320,
            query_heads: 8,
            kv_heads: 2,
            head_width: 256,
            rotary_width: 64,
        }
    }

    /// Declared Qwen3.5 2B ordinary-text dimensions. Geometry alone
    /// grants no model, source, backend, or numerical authority.
    pub const fn qwen35_2b() -> Self {
        Self {
            hidden_width: 2048,
            intermediate_width: 6144,
            vocabulary: 248_320,
            query_heads: 8,
            kv_heads: 2,
            head_width: 256,
            rotary_width: 64,
        }
    }

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

/// Runtime token interval. Cache append covers `[base, base + rows)`.
/// The unselected attention diagnostic observes each row through its new token.
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

/// Exact internal pointer role; order comes from each crate-private definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BufferRole {
    Positions,
    Query,
    Key,
    Value,
    InverseFrequency,
    TokenIds,
    EmbeddingTable,
    Input,
    Residual,
    RawGamma,
    QueryRawGamma,
    KeyRawGamma,
    QueryNormalized,
    KeyNormalized,
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
pub(crate) enum Access {
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
pub(crate) struct BufferBinding {
    pub role: BufferRole,
    pub bytes: usize,
    pub alignment: usize,
    pub access: Access,
}

/// A caller-supplied physical handle tagged with its intended ABI role and
/// access. Construction is crate-private: only the physical owner may bind it.
#[derive(Clone)]
pub(crate) struct LaunchBinding {
    role: BufferRole,
    access: Access,
    handle: Handle,
}

impl LaunchBinding {
    pub(crate) fn new(role: BufferRole, access: Access, handle: Handle) -> Self {
        Self {
            role,
            access,
            handle,
        }
    }
}

/// Caller-owned status protocol. Clearing must complete before return;
/// completion/read must wait for this submission and return the observed word.
pub(crate) trait CallerStatusContract {
    fn binding(&self) -> LaunchBinding;
    fn clear_before_submit(&mut self, client: &Client) -> Result<()>;
    fn complete_and_read(&mut self, client: &Client) -> Result<u32>;
}

/// Completed kernel status only. Zero flags do not publish or imply `Ready`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchCompletion {
    pub(crate) status: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Primitive {
    Positions,
    Rope,
    Embedding,
    Rms,
    ResidualRms,
    SiluUp,
    SigmoidGate,
    Residual,
    KvAppend,
    QueryHeadRms,
    KeyHeadRms,
    /// Diagnostic online-FP32 arithmetic; not Qwen numerical authority and
    /// never selected by [`Program::definitions`].
    UnqualifiedOnlineFp32Attention,
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
            Self::QueryHeadRms => "zg_ot_query_head_rms_v1",
            Self::KeyHeadRms => "zg_ot_key_head_rms_v1",
            Self::UnqualifiedOnlineFp32Attention => "zg_ot_unqualified_online_fp32_attention_v1",
            Self::Greedy => "zg_ot_greedy_v1",
        }
    }
}

/// Complete source, ABI, static scalar block, and launch geometry.
#[derive(Clone)]
pub(crate) struct Definition {
    primitive: Primitive,
    source: Arc<str>,
    bindings: Arc<[BufferBinding]>,
    permitted_aliases: Arc<[(BufferRole, BufferRole)]>,
    scalars: StaticScalarBlock,
    grid: [u32; 3],
    kernel: PinnedKernel,
}

impl Definition {
    fn new(
        primitive: Primitive,
        source: Arc<str>,
        bindings: Vec<BufferBinding>,
        permitted_aliases: &[(BufferRole, BufferRole)],
        scalars: StaticScalarBlock,
        grid: [u32; 3],
    ) -> Result<Self> {
        if bindings.is_empty()
            || bindings.iter().any(|binding| {
                binding.bytes == 0 || !binding.alignment.is_power_of_two() || binding.alignment > 16
            })
            || bindings.last().is_none_or(|binding| {
                binding.role != BufferRole::Status || binding.access != Access::ReadWrite
            })
            || permitted_aliases.iter().any(|&(a, b)| {
                a == b
                    || !bindings.iter().any(|binding| binding.role == a)
                    || !bindings.iter().any(|binding| binding.role == b)
            })
        {
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
            permitted_aliases: permitted_aliases.into(),
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

    /// Validate role, access, service, extent, alignment, and alias policy;
    /// retain immutable handles without granting direct launch authority.
    pub(crate) fn prepare<S: CallerStatusContract>(
        &self,
        client: &Client,
        mut bindings: Vec<LaunchBinding>,
        status: &S,
    ) -> Result<PreparedLaunch> {
        if bindings
            .iter()
            .any(|binding| binding.role == BufferRole::Status)
        {
            return Err(Error::State(
                "ordinary text status must come from its caller owner".into(),
            ));
        }
        bindings.push(status.binding());
        self.validate_bindings(client.service_id(), &bindings)?;
        let status_handle = bindings
            .last()
            .ok_or_else(|| Error::State("ordinary text status binding absent".into()))?
            .handle
            .clone();
        Ok(PreparedLaunch {
            inner: self.kernel.prepare(
                client,
                self.grid,
                bindings.into_iter().map(|binding| binding.handle).collect(),
            )?,
            primitive: self.primitive,
            status_handle,
        })
    }

    fn validate_bindings(&self, service: ServiceId, bindings: &[LaunchBinding]) -> Result<()> {
        if bindings.len() != self.bindings.len()
            || bindings
                .iter()
                .zip(self.bindings.iter())
                .any(|(actual, expected)| {
                    actual.role != expected.role
                        || actual.access != expected.access
                        || actual.handle.service != service
                        || used_bytes(&actual.handle).ok() != Some(expected.bytes as u64)
                        || !actual
                            .handle
                            .offset_start
                            .unwrap_or(0)
                            .is_multiple_of(expected.alignment as u64)
                })
        {
            return Err(Error::State(
                "ordinary text role/access/service/extent/alignment differs".into(),
            ));
        }
        for left in 0..bindings.len() {
            for right in left + 1..bindings.len() {
                let a = &bindings[left];
                let b = &bindings[right];
                if !overlap(&a.handle, &b.handle)? {
                    continue;
                }
                let permitted = self
                    .permitted_aliases
                    .iter()
                    .any(|&(x, y)| (x == a.role && y == b.role) || (x == b.role && y == a.role));
                if !permitted || !same_range(&a.handle, &b.handle)? {
                    return Err(Error::State(
                        "ordinary text binding alias is not explicitly permitted".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Compile/load and require the exact driver-reflected pointer/scalar ABI.
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    fn diagnose_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel
            .diagnose_parameter_layout(queue, self.bindings.len())
    }
}

#[derive(Clone)]
pub(crate) struct PreparedLaunch {
    inner: PreparedPinnedLaunch,
    primitive: Primitive,
    status_handle: Handle,
}

impl PreparedLaunch {
    pub(crate) fn primitive(&self) -> Primitive {
        self.primitive
    }

    /// Clear through the caller's owner, submit once, then wait/read through
    /// that same owner. The returned flags are not a publication state.
    pub(crate) fn submit<S: CallerStatusContract>(
        &self,
        client: &Client,
        status: &mut S,
    ) -> Result<LaunchCompletion> {
        require_same_status(&self.status_handle, &status.binding())?;
        status.clear_before_submit(client)?;
        require_same_status(&self.status_handle, &status.binding())?;
        self.inner.launch_on(client)?;
        let observed = status.complete_and_read(client)?;
        require_same_status(&self.status_handle, &status.binding())?;
        Ok(LaunchCompletion { status: observed })
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

    pub(crate) fn positions(&self, step: Step) -> Result<Definition> {
        let step = Step::new(step.rows, step.base, step.capacity)?;
        self.definition(
            Primitive::Positions,
            vec![
                binding(BufferRole::Positions, step.rows * 4, Access::Write),
                status(),
            ],
            u32_words(&[step.rows, step.base, step.capacity])?,
            element_grid(step.rows)?,
        )
    }

    pub(crate) fn rope(&self, step: Step) -> Result<Definition> {
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

    pub(crate) fn embedding(&self, rows: usize) -> Result<Definition> {
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

    pub(crate) fn rms(&self, rows: usize, epsilon: f32) -> Result<Definition> {
        self.rms_definition(Primitive::Rms, rows, epsilon)
    }

    pub(crate) fn residual_rms(&self, rows: usize, epsilon: f32) -> Result<Definition> {
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
            binding(BufferRole::RawGamma, gamma, Access::Read),
            binding(BufferRole::Output, bytes, Access::Write),
            status(),
        ]);
        self.definition_with_aliases(
            primitive,
            bindings,
            &[(BufferRole::Input, BufferRole::Output)],
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

    pub(crate) fn silu_up(&self, rows: usize) -> Result<Definition> {
        self.elementwise(
            Primitive::SiluUp,
            rows,
            self.geometry.intermediate_width,
            BufferRole::Gate,
            BufferRole::Up,
        )
    }

    pub(crate) fn sigmoid_gate(&self, rows: usize) -> Result<Definition> {
        self.elementwise(
            Primitive::SigmoidGate,
            rows,
            self.geometry.query_width()?,
            BufferRole::Input,
            BufferRole::Gate,
        )
    }

    pub(crate) fn residual(&self, rows: usize) -> Result<Definition> {
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
        self.definition_with_aliases(
            primitive,
            vec![
                binding(first, bytes, Access::Read),
                binding(second, bytes, Access::Read),
                binding(BufferRole::Output, bytes, Access::Write),
                status(),
            ],
            &[(first, BufferRole::Output)],
            u32_words(&[rows])?,
            element_grid(checked_mul(rows, width, "elementwise grid")?)?,
        )
    }

    pub(crate) fn kv_append(&self, step: Step) -> Result<Definition> {
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

    pub(crate) fn query_head_rms(&self, rows: usize, epsilon: f32) -> Result<Definition> {
        self.head_rms_definition(
            Primitive::QueryHeadRms,
            rows,
            self.geometry.query_heads,
            BufferRole::Query,
            BufferRole::QueryRawGamma,
            BufferRole::QueryNormalized,
            epsilon,
        )
    }

    pub(crate) fn key_head_rms(&self, rows: usize, epsilon: f32) -> Result<Definition> {
        self.head_rms_definition(
            Primitive::KeyHeadRms,
            rows,
            self.geometry.kv_heads,
            BufferRole::Key,
            BufferRole::KeyRawGamma,
            BufferRole::KeyNormalized,
            epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn head_rms_definition(
        &self,
        primitive: Primitive,
        rows: usize,
        heads: usize,
        input: BufferRole,
        raw_gamma: BufferRole,
        output: BufferRole,
        epsilon: f32,
    ) -> Result<Definition> {
        require_rows(rows)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::Unsupported("ordinary text head RMS epsilon".into()));
        }
        let bytes = extent(
            rows,
            heads * self.geometry.head_width,
            BF16_BYTES,
            "head RMS",
        )?;
        let gamma = self.geometry.head_width * BF16_BYTES;
        self.definition_with_aliases(
            primitive,
            vec![
                binding(input, bytes, Access::Read),
                binding(raw_gamma, gamma, Access::Read),
                binding(output, bytes, Access::Write),
                status(),
            ],
            &[(input, output)],
            words(&[
                (StaticScalarType::U32, StaticScalarValue::U32(rows as u64)),
                (
                    StaticScalarType::F32Bits,
                    StaticScalarValue::F32Bits(epsilon.to_bits()),
                ),
            ])?,
            [
                u32::try_from(rows * heads)
                    .map_err(|_| Error::Limit("ordinary text head RMS grid".into()))?,
                1,
                1,
            ],
        )
    }

    /// Unqualified online-FP32 diagnostic. This intentionally does not claim
    /// or implement the declared Qwen BF16 attention cast/reduction schedule.
    fn unqualified_online_fp32_attention(&self, step: Step) -> Result<Definition> {
        let step = Step::new(step.rows, step.base, step.capacity)?;
        let plan = self.memory_plan(step.rows, step.capacity)?;
        self.definition(
            Primitive::UnqualifiedOnlineFp32Attention,
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

    pub(crate) fn greedy(&self, rows: usize) -> Result<Definition> {
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

    /// Selectable ordinary primitives. The unqualified online-FP32 attention
    /// diagnostic is deliberately absent.
    pub(crate) fn definitions(&self, step: Step, epsilon: f32) -> Result<Vec<Definition>> {
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
            self.query_head_rms(step.rows, epsilon)?,
            self.key_head_rms(step.rows, epsilon)?,
            self.greedy(step.rows)?,
        ])
    }

    fn diagnostic_definitions(&self, step: Step, epsilon: f32) -> Result<Vec<Definition>> {
        let mut definitions = self.definitions(step, epsilon)?;
        definitions.push(self.unqualified_online_fp32_attention(step)?);
        Ok(definitions)
    }

    fn definition(
        &self,
        primitive: Primitive,
        bindings: Vec<BufferBinding>,
        scalars: StaticScalarBlock,
        grid: [u32; 3],
    ) -> Result<Definition> {
        self.definition_with_aliases(primitive, bindings, &[], scalars, grid)
    }

    fn definition_with_aliases(
        &self,
        primitive: Primitive,
        bindings: Vec<BufferBinding>,
        permitted_aliases: &[(BufferRole, BufferRole)],
        scalars: StaticScalarBlock,
        grid: [u32; 3],
    ) -> Result<Definition> {
        Definition::new(
            primitive,
            self.source.clone(),
            bindings,
            permitted_aliases,
            scalars,
            grid,
        )
    }
}

/// Host mirror of the diagnostic greedy kernel. Non-finite values are ignored
/// for deterministic output and set flags; callers must not publish that row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyHostOutcome {
    pub token: u32,
    pub maximum_bits: u32,
    pub status: u32,
}

pub fn greedy_host_semantics(logits: &[f32]) -> Result<GreedyHostOutcome> {
    if logits.is_empty() || logits.len() > 1_048_576 || u32::try_from(logits.len()).is_err() {
        return Err(Error::Unsupported(
            "ordinary text host greedy vocabulary extent".into(),
        ));
    }
    let mut best: Option<(usize, f32)> = None;
    let mut flags = 0;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            flags |= STATUS_NONFINITE;
            if value.is_nan() {
                flags |= STATUS_NAN_LOGIT;
            }
            continue;
        }
        if best.is_none_or(|(best_index, maximum)| {
            value > maximum || value == maximum && index < best_index
        }) {
            best = Some((index, value));
        }
    }
    let (token, maximum) = best.unwrap_or((0, f32::NEG_INFINITY));
    Ok(GreedyHostOutcome {
        token: token as u32,
        maximum_bits: maximum.to_bits(),
        status: flags,
    })
}

/// Physical compile/reflection diagnostic only; it performs no launch and
/// grants no model or numerical authority. Includes the explicitly unqualified
/// online-FP32 attention body so its ABI cannot silently drift.
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub fn compile_reflection_diagnostic(queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
    let mut count = 0;
    for geometry in [Geometry::qwen35_08(), Geometry::qwen35_2b()] {
        let program = Program::new(geometry)?;
        for definition in program.diagnostic_definitions(Step::new(64, 262_080, 262_144)?, 1e-6)? {
            definition.diagnose_code(queue)?;
            count += 1;
        }
    }
    Ok(count)
}

fn binding(role: BufferRole, bytes: usize, access: Access) -> BufferBinding {
    BufferBinding {
        role,
        bytes,
        alignment: 4,
        access,
    }
}

fn status() -> BufferBinding {
    binding(BufferRole::Status, STATUS_BYTES, Access::ReadWrite)
}

fn used_bytes(handle: &Handle) -> Result<u64> {
    handle
        .size()
        .checked_sub(handle.offset_start.unwrap_or(0))
        .and_then(|bytes| bytes.checked_sub(handle.offset_end.unwrap_or(0)))
        .ok_or_else(|| Error::State("ordinary text invalid binding offsets".into()))
}

fn handle_range(handle: &Handle) -> Result<(u64, u64)> {
    let start = handle.offset_start.unwrap_or(0);
    let end = start
        .checked_add(used_bytes(handle)?)
        .ok_or_else(|| Error::Limit("ordinary text binding range overflow".into()))?;
    Ok((start, end))
}

fn same_allocation(a: &Handle, b: &Handle) -> bool {
    a.service == b.service && a.memory.descriptor().id == b.memory.descriptor().id
}

fn overlap(a: &Handle, b: &Handle) -> Result<bool> {
    if !same_allocation(a, b) {
        return Ok(false);
    }
    let (a_start, a_end) = handle_range(a)?;
    let (b_start, b_end) = handle_range(b)?;
    Ok(a_start < b_end && b_start < a_end)
}

fn same_range(a: &Handle, b: &Handle) -> Result<bool> {
    Ok(same_allocation(a, b) && handle_range(a)? == handle_range(b)?)
}

fn require_same_status(expected: &Handle, actual: &LaunchBinding) -> Result<()> {
    if actual.role != BufferRole::Status
        || actual.access != Access::ReadWrite
        || !same_range(expected, &actual.handle)?
        || expected.service != actual.handle.service
        || used_bytes(&actual.handle)? != STATUS_BYTES as u64
        || !actual.handle.offset_start.unwrap_or(0).is_multiple_of(4)
    {
        return Err(Error::State(
            "ordinary text caller status owner/binding changed".into(),
        ));
    }
    Ok(())
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
