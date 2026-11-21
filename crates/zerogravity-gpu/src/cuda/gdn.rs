#![cfg_attr(
    not(feature = "cuda-native-qualification"),
    allow(dead_code, unused_imports)
)]
//! Private SM86 .8B layer0 qualification; original generated instructions.
//! Runtime storage comes from the original resident owner, never a trace.
use crate::cuda::kernel::{PinnedKernel, StaticScalarBlock};
use cubecl_runtime::kernel::BufferIOAttr;
#[cfg(test)]
use std::collections::BTreeMap;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::ParameterWindow;
use zerogravity_core::accelerator::program::{InvocationPhase, TextInvocation};
use zerogravity_core::encoding::StoredTensorEncoding;
use zerogravity_core::{Error, Result};

mod arithmetic;
pub(crate) use arithmetic::GdnFamilyIdentity;
mod assets;
pub(crate) mod head_tile;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod qwen2b_arithmetic;
#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", feature = "cuda-native-qualification")
))]
pub(crate) mod state_transport;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use qwen2b_arithmetic::{Qwen2BGdnArithmetic, Qwen2BOriginalGdnReference};
#[cfg(feature = "cuda-native-qualification")]
pub(crate) mod resident;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use resident::{
    ComponentConvEnqueued, ComponentGdnAdapter, GdnConvEnqueued, GdnStateAdapter, ModelGdnAdapter,
    ModelGdnConvEnqueued, ModelGdnTransportStorage,
};
#[cfg(feature = "cuda-native-qualification")]
pub(crate) use resident::{GdnArithmetic, GdnInputs, GdnPrepared, GdnTypedOutputs};

pub(crate) const CACHE_SLOT_BYTES: usize = 278528 * 4;
pub(crate) const CACHE_SLOTS: usize = 5;
pub(crate) const CONV_STATE_BYTES: usize = 3 * 6144 * 2;
pub(crate) const RECURRENT_STATE_BYTES: usize = 16 * 128 * 128 * 4;

pub(crate) fn loaded_gamma(window: &ParameterWindow) -> Result<Vec<u8>> {
    if !window.is_complete() || window.parameter().elements() != 128 {
        return Err(Error::Malformed("complete original GDN gamma128".into()));
    }
    let values = match window.parameter().tensor().kind {
        StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) => {
            window.as_slice().to_vec()
        }
        StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) => {
            let mut out = Vec::with_capacity(256);
            for raw in window.as_slice().chunks_exact(4) {
                let bits = u32::from_le_bytes(raw.try_into().unwrap());
                if !f32::from_bits(bits).is_finite() {
                    return Err(Error::Malformed("nonfinite original GDN gamma".into()));
                }
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
            }
            out
        }
        _ => return Err(Error::Unsupported("GDN gamma source dtype".into())),
    };
    if values.len() != 256
        || values.chunks_exact(2).any(|raw| {
            let bits = u16::from_le_bytes([raw[0], raw[1]]);
            bits & 0x7f80 == 0x7f80
        })
    {
        return Err(Error::Malformed("finite loaded BF16 GDN gamma".into()));
    }
    Ok(values)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnPhase {
    InitialOne,
    Initial64,
    ContinuedTwo,
    CachedDecode,
    FixedPrompt,
    /// One actual prompt with ceil(M/64) GDN chunks, never two model calls.
    TextPrefill(usize),
    /// Equal-row independent prefill sequences packed into one physical launch.
    /// Session/root authority remains external; this is physical GDN geometry.
    TextPrefillCohort {
        rows: usize,
        lanes: usize,
    },
}
impl GdnPhase {
    pub(crate) fn for_reference_call(call: &TextInvocation) -> Result<Self> {
        match (call.phase(), call.tokens().base(), call.tokens().len()) {
            (InvocationPhase::Prefill, 0, 1) => Ok(Self::InitialOne),
            (InvocationPhase::Prefill, 0, 64) => Ok(Self::Initial64),
            (InvocationPhase::Prefill, 64, 2) => Ok(Self::ContinuedTwo),
            (InvocationPhase::Decode, 1 | 66, 1) => Ok(Self::CachedDecode),
            _ => Err(Error::Unsupported(
                "original five GDN reference-call geometries".into(),
            )),
        }
    }

    pub(crate) fn tokens(self) -> usize {
        match self {
            Self::InitialOne | Self::CachedDecode => 1,
            Self::Initial64 => 64,
            Self::FixedPrompt => 47,
            Self::ContinuedTwo => 2,
            Self::TextPrefill(prompt_tokens) => prompt_tokens,
            Self::TextPrefillCohort { rows, lanes } => rows * lanes,
        }
    }
    pub(crate) fn chunks(self) -> usize {
        match self {
            Self::TextPrefillCohort { rows, lanes } => rows.div_ceil(64) * lanes,
            _ => self.tokens().div_ceil(64),
        }
    }
    pub(crate) fn sequences(self) -> usize {
        match self {
            Self::TextPrefillCohort { lanes, .. } => lanes,
            _ => 1,
        }
    }
    pub(crate) fn rows_per_sequence(self) -> usize {
        match self {
            Self::TextPrefillCohort { rows, .. } => rows,
            _ => self.tokens(),
        }
    }
    pub(crate) fn conv_chunks(self) -> usize {
        self.rows_per_sequence().div_ceil(8) * self.sequences()
    }
    pub(crate) fn is_text_prefill(self) -> bool {
        matches!(self, Self::TextPrefill(_) | Self::TextPrefillCohort { .. })
    }
    pub(crate) fn has_initial_state(self) -> bool {
        matches!(self, Self::ContinuedTwo | Self::CachedDecode)
    }
    pub(crate) fn stages(self) -> &'static [GdnStage] {
        use GdnStage::*;
        if self == Self::CachedDecode {
            &[Conv, Decode, Gated]
        } else {
            &[Conv, Prep, Cumsum, Kkt, Solve, Recompute, H, Output, Gated]
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnCacheSlot {
    One,
    Four,
}
impl GdnCacheSlot {
    #[cfg(test)]
    pub(crate) fn new(index: u32) -> Result<Self> {
        match index {
            1 => Ok(Self::One),
            4 => Ok(Self::Four),
            _ => Err(Error::Unsupported(
                "closed non-null vLLM cache slot1/4".into(),
            )),
        }
    }
    pub(crate) fn index(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
        }
    }
    pub(crate) fn convolution_offset(self) -> usize {
        self.index() * CACHE_SLOT_BYTES
    }
    pub(crate) fn recurrent_offset(self) -> usize {
        self.convolution_offset() + CONV_STATE_BYTES
    }
    fn validate(self, phase: GdnPhase) -> Result<()> {
        if matches!(
            (phase, self),
            (
                GdnPhase::InitialOne
                    | GdnPhase::FixedPrompt
                    | GdnPhase::TextPrefill(_)
                    | GdnPhase::TextPrefillCohort { .. },
                Self::Four
            ) | (GdnPhase::Initial64 | GdnPhase::ContinuedTwo, Self::One)
        ) {
            return Err(Error::Unsupported(
                "closed vLLM actual phase/cache issuer".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnStage {
    Conv,
    Prep,
    Cumsum,
    Kkt,
    Solve,
    Recompute,
    H,
    Output,
    Gated,
    Decode,
}

/// Logical allocation names; equal roles retain ONE actual allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum GdnBuffer {
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
impl GdnBuffer {
    pub(crate) fn bytes(self, phase: GdnPhase) -> usize {
        let tokens = phase.tokens();
        let sequences = phase.sequences();
        match self {
            Self::PackedQkvz => tokens * 8192 * 2,
            Self::PackedBa => tokens * 32 * 2,
            Self::AContiguous | Self::BContiguous => tokens * 16 * 2,
            Self::ZCompact
            | Self::Q
            | Self::K
            | Self::V
            | Self::W
            | Self::U
            | Self::VNew
            | Self::Core
            | Self::Gated => tokens * 2048 * 2,
            Self::ConvWeights => 6144 * 4 * 2,
            Self::ALog => 16 * 4,
            Self::DtBias => 16 * 2,
            Self::GatedGamma => 128 * 2,
            Self::CacheArena => match phase {
                GdnPhase::TextPrefillCohort { .. } => (sequences + 1) * CACHE_SLOT_BYTES,
                _ => CACHE_SLOTS * CACHE_SLOT_BYTES,
            },
            Self::CacheIndices => sequences * 4,
            Self::HasInitial => sequences,
            Self::QueryStarts => (sequences + 1) * 4,
            Self::ChunkIndices => phase.chunks() * 8,
            Self::ConvBatch | Self::ConvChunkOffsets => 2048 * 4,
            Self::ChunkOffsets => (sequences + 1) * 8,
            Self::ConvOutput => tokens * 6144 * 2,
            Self::GRaw | Self::Beta | Self::GCumulative | Self::Rstd => tokens * 16 * 4,
            Self::Kkt => tokens * 16 * 64 * 4,
            Self::Inverse => tokens * 16 * 64 * 2,
            Self::HChunks => phase.chunks() * 16 * 128 * 128 * 2,
            Self::InitialState | Self::FinalState => sequences * RECURRENT_STATE_BYTES,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BindingSpec {
    pub role: GdnBuffer,
    pub offset: usize,
    pub physical_bytes: usize,
    pub io: BufferIOAttr,
    // Repeated in each pointer descriptor so existing StageArithmetic equality
    // retains the exact immutable by-value launch identity.
    pub static_scalars: StaticScalarBlock,
}
#[derive(Clone)]
pub(crate) struct LaunchSpec {
    arithmetic: arithmetic::StageArithmetic,
    pub kernel: PinnedKernel,
    pub grid: [u32; 3],
    pub bindings: Vec<BindingSpec>,
}
fn binding(phase: GdnPhase, role: GdnBuffer, io: BufferIOAttr) -> BindingSpec {
    BindingSpec {
        role,
        offset: 0,
        physical_bytes: role.bytes(phase),
        io,
        static_scalars: StaticScalarBlock::empty(),
    }
}
fn view(role: GdnBuffer, offset: usize, bytes: usize, io: BufferIOAttr) -> BindingSpec {
    BindingSpec {
        role,
        offset,
        physical_bytes: bytes,
        io,
        static_scalars: StaticScalarBlock::empty(),
    }
}

/// Original generated-source payload/metadata admission for repeated full-model
/// preparations. Covers retained code plus bounded construction temporaries.
pub(crate) fn model_metadata_bound(phase: GdnPhase) -> Result<usize> {
    phase
        .stages()
        .iter()
        // At most six packing/state-copy keys, independent of stage count.
        .try_fold(
            256usize * 1024 + 6 * PinnedKernel::KEY_METADATA_BYTES,
            |sum, stage| {
                let asset = assets::select(phase, *stage)?;
                sum.checked_add(
                    asset
                        .source
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("full model GDN code extent".into()))?,
                )
                .and_then(|n| n.checked_add(4096 + PinnedKernel::KEY_METADATA_BYTES))
                .ok_or_else(|| {
                    Error::Limit("full model GDN original code/metadata admission".into())
                })
            },
        )
}

pub(crate) fn kernel(phase: GdnPhase, stage: GdnStage) -> Result<LaunchSpec> {
    kernel_impl(phase, stage, false)
}
/// Same pinned PTX/instructions, actual distinct typed producer leases.
pub(crate) fn kernel_with_typed_outputs(phase: GdnPhase, stage: GdnStage) -> Result<LaunchSpec> {
    kernel_impl(phase, stage, true)
}

#[cfg(feature = "cuda-cublas")]
pub(crate) struct CachedConvBatch {
    kernel: PinnedKernel,
    grid: [u32; 3],
}
#[cfg(feature = "cuda-cublas")]
impl CachedConvBatch {
    pub(crate) fn prepare(rows: usize) -> Result<Self> {
        use BufferIOAttr::{ReadOnly as R, ReadWrite as RW};
        let (asset, scalars) = text::cached_conv_batch(rows)?;
        if !asset
            .source
            .starts_with("// cubecl.cuda.dynamic_shared_bytes=0\n")
            || !asset.source.contains(asset.ptx_sha256)
        {
            return Err(Error::State("cached GDN conv batch source identity".into()));
        }
        // The retained kernel indexes ctaid.x with the original 8192-BF16
        // packed-QKVZ stride even though it reads/writes only the conv prefix.
        let row_bytes = GdnBuffer::PackedQkvz.bytes(GdnPhase::CachedDecode);
        let io_bytes = row_bytes
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("cached GDN conv batch row bytes".into()))?;
        let index_bytes = rows
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("cached GDN conv batch index bytes".into()))?;
        let bindings = Arc::from([
            (io_bytes, RW),
            (GdnBuffer::ConvWeights.bytes(GdnPhase::CachedDecode), R),
            (GdnBuffer::CacheArena.bytes(GdnPhase::CachedDecode), RW),
            (index_bytes, R),
            (io_bytes, RW),
        ]);
        Ok(Self {
            kernel: PinnedKernel::new_with_static_scalars(
                asset.entry,
                Arc::from(asset.source),
                asset.threads,
                bindings,
                scalars,
            )?,
            grid: asset.grid,
        })
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub(crate) fn launch(
        &self,
        client: &cubecl::client::Client,
        packed_qkvz: &cubecl::server::Handle,
        output_scratch: &cubecl::server::Handle,
        weights: &cubecl::server::Handle,
        cache: &cubecl::server::Handle,
        indices: &cubecl::server::Handle,
    ) -> Result<()> {
        self.kernel.launch(
            client,
            self.grid,
            &[
                packed_qkvz.clone(),
                weights.clone(),
                cache.clone(),
                indices.clone(),
                output_scratch.clone(),
            ],
        )
    }
}

#[cfg(feature = "cuda-cublas")]
pub(crate) struct CachedDecodeBatch {
    kernel: PinnedKernel,
    grid: [u32; 3],
}
#[cfg(feature = "cuda-cublas")]
impl CachedDecodeBatch {
    pub(crate) fn prepare(rows: usize) -> Result<Self> {
        use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
        let (asset, scalars) = text::cached_decode_batch(rows)?;
        if !asset
            .source
            .starts_with("// cubecl.cuda.dynamic_shared_bytes=64\n")
            || !asset.source.contains(asset.ptx_sha256)
        {
            return Err(Error::State(
                "cached GDN decode batch source identity".into(),
            ));
        }
        let conv_bytes = GdnBuffer::PackedQkvz
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("cached GDN decode batch conv bytes".into()))?;
        let packed_ba = GdnBuffer::PackedBa
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("cached GDN decode batch BA bytes".into()))?;
        let ba_view = packed_ba
            .checked_sub(32)
            .ok_or_else(|| Error::Limit("cached GDN decode batch BA view".into()))?;
        let core_bytes = GdnBuffer::Core
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("cached GDN decode batch core bytes".into()))?;
        let state_bytes = GdnBuffer::CacheArena
            .bytes(GdnPhase::CachedDecode)
            .checked_sub(CONV_STATE_BYTES)
            .ok_or_else(|| Error::Limit("cached GDN decode batch state bytes".into()))?;
        let index_bytes = rows
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("cached GDN decode batch index bytes".into()))?;
        Ok(Self {
            kernel: PinnedKernel::new_with_static_scalars(
                asset.entry,
                Arc::from(asset.source),
                asset.threads,
                Arc::from([
                    (conv_bytes, R),
                    (ba_view, R),
                    (ba_view, R),
                    (GdnBuffer::ALog.bytes(GdnPhase::CachedDecode), R),
                    (GdnBuffer::DtBias.bytes(GdnPhase::CachedDecode), R),
                    (core_bytes, W),
                    (state_bytes, RW),
                    (state_bytes, RW),
                    (index_bytes, R),
                ]),
                scalars,
            )?,
            grid: asset.grid,
        })
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(
        &self,
        client: &cubecl::client::Client,
        conv: &cubecl::server::Handle,
        a: &cubecl::server::Handle,
        beta: &cubecl::server::Handle,
        a_log: &cubecl::server::Handle,
        dt_bias: &cubecl::server::Handle,
        core: &cubecl::server::Handle,
        state: &cubecl::server::Handle,
        indices: &cubecl::server::Handle,
    ) -> Result<()> {
        self.kernel.launch(
            client,
            self.grid,
            &[
                conv.clone(),
                a.clone(),
                beta.clone(),
                a_log.clone(),
                dt_bias.clone(),
                core.clone(),
                state.clone(),
                state.clone(),
                indices.clone(),
            ],
        )
    }
}

/// Cached GDN decode over caller-owned private native recurrent state.
///
/// This is a physical CUDA primitive only: it owns no model/session/root policy
/// and assumes each state handle already uses the retained native H/V/K layout.
#[cfg(feature = "cuda-cublas")]
pub(crate) struct CachedDecodePrivateBatch {
    kernel: PinnedKernel,
    grid: [u32; 3],
    rows: usize,
}
#[cfg(feature = "cuda-cublas")]
impl CachedDecodePrivateBatch {
    pub(crate) fn prepare(rows: usize) -> Result<Self> {
        use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
        let definition = text::cached_decode_private_batch(rows)?;
        if !definition
            .source
            .starts_with("// cubecl.cuda.dynamic_shared_bytes=64\n")
            || !definition.source.contains(definition.ptx_sha256)
        {
            return Err(Error::State(
                "private cached GDN decode source identity".into(),
            ));
        }
        let conv_bytes = GdnBuffer::PackedQkvz
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("private cached GDN decode conv bytes".into()))?;
        let packed_ba = GdnBuffer::PackedBa
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("private cached GDN decode BA bytes".into()))?;
        let ba_view = packed_ba
            .checked_sub(32)
            .ok_or_else(|| Error::Limit("private cached GDN decode BA view".into()))?;
        let core_bytes = GdnBuffer::Core
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("private cached GDN decode core bytes".into()))?;
        let index_bytes = rows
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("private cached GDN decode index bytes".into()))?;
        let mut bindings = vec![
            (conv_bytes, R),
            (ba_view, R),
            (ba_view, R),
            (GdnBuffer::ALog.bytes(GdnPhase::CachedDecode), R),
            (GdnBuffer::DtBias.bytes(GdnPhase::CachedDecode), R),
            (core_bytes, W),
            (RECURRENT_STATE_BYTES, R),
            (RECURRENT_STATE_BYTES, W),
            (index_bytes, R),
        ];
        for _ in 1..rows {
            bindings.push((RECURRENT_STATE_BYTES, R));
            bindings.push((RECURRENT_STATE_BYTES, W));
        }
        Ok(Self {
            kernel: PinnedKernel::new_with_static_scalars(
                definition.entry,
                definition.source,
                definition.threads,
                bindings.into(),
                definition.scalars,
            )?,
            grid: definition.grid,
            rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(
        &self,
        client: &cubecl::client::Client,
        conv: &cubecl::server::Handle,
        a: &cubecl::server::Handle,
        beta: &cubecl::server::Handle,
        a_log: &cubecl::server::Handle,
        dt_bias: &cubecl::server::Handle,
        core: &cubecl::server::Handle,
        prior: &[cubecl::server::Handle],
        candidate: &[cubecl::server::Handle],
        indices: &cubecl::server::Handle,
    ) -> Result<()> {
        if prior.len() != self.rows || candidate.len() != self.rows {
            return Err(Error::State(
                "private cached GDN decode state coverage".into(),
            ));
        }
        let mut arguments = Vec::with_capacity(9 + 2 * self.rows.saturating_sub(1));
        arguments.extend([
            conv.clone(),
            a.clone(),
            beta.clone(),
            a_log.clone(),
            dt_bias.clone(),
            core.clone(),
            prior[0].clone(),
            candidate[0].clone(),
            indices.clone(),
        ]);
        for lane in 1..self.rows {
            arguments.push(prior[lane].clone());
            arguments.push(candidate[lane].clone());
        }
        self.kernel.launch(client, self.grid, &arguments)
    }
}

#[cfg(feature = "cuda-cublas")]
pub(crate) struct CachedGatedBatch {
    kernel: PinnedKernel,
    grid: [u32; 3],
}
#[cfg(feature = "cuda-cublas")]
impl CachedGatedBatch {
    pub(crate) fn prepare(rows: usize) -> Result<Self> {
        use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
        let (asset, scalars) = text::cached_gated_batch(rows)?;
        if !asset
            .source
            .starts_with("// cubecl.cuda.dynamic_shared_bytes=0\n")
            || !asset.source.contains(asset.ptx_sha256)
        {
            return Err(Error::State(
                "cached GDN gated batch source identity".into(),
            ));
        }
        let row_bytes = GdnBuffer::Core.bytes(GdnPhase::CachedDecode);
        let io_bytes = row_bytes
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("cached GDN gated batch row bytes".into()))?;
        let rstd_bytes = GdnBuffer::Rstd
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("cached GDN gated batch rstd bytes".into()))?;
        Ok(Self {
            kernel: PinnedKernel::new_with_static_scalars(
                asset.entry,
                Arc::from(asset.source),
                asset.threads,
                Arc::from([
                    (io_bytes, R),
                    (io_bytes, W),
                    (GdnBuffer::GatedGamma.bytes(GdnPhase::CachedDecode), R),
                    (io_bytes, R),
                    (rstd_bytes, W),
                ]),
                scalars,
            )?,
            grid: asset.grid,
        })
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub(crate) fn launch(
        &self,
        client: &cubecl::client::Client,
        core: &cubecl::server::Handle,
        gated: &cubecl::server::Handle,
        gamma: &cubecl::server::Handle,
        z: &cubecl::server::Handle,
        rstd: &cubecl::server::Handle,
    ) -> Result<()> {
        self.kernel.launch(
            client,
            self.grid,
            &[
                core.clone(),
                gated.clone(),
                gamma.clone(),
                z.clone(),
                rstd.clone(),
            ],
        )
    }
}

fn kernel_impl(phase: GdnPhase, stage: GdnStage, typed_outputs: bool) -> Result<LaunchSpec> {
    kernel_from_asset(
        phase,
        stage,
        typed_outputs,
        assets::select(phase, stage)?,
        arithmetic::GdnArithmeticFamily::Original800M,
    )
}
#[cfg(target_os = "linux")]
fn kernel_qwen2b_component(phase: GdnPhase, stage: GdnStage) -> Result<LaunchSpec> {
    kernel_from_asset(
        phase,
        stage,
        true,
        assets::select_qwen2b(phase, stage)?,
        arithmetic::GdnArithmeticFamily::Original2b,
    )
}
fn kernel_from_asset(
    phase: GdnPhase,
    stage: GdnStage,
    typed_outputs: bool,
    asset: assets::Asset,
    family: arithmetic::GdnArithmeticFamily,
) -> Result<LaunchSpec> {
    kernel_from_asset_with_scalars(
        phase,
        stage,
        typed_outputs,
        asset,
        family,
        StaticScalarBlock::empty(),
    )
}
fn kernel_from_asset_with_scalars(
    phase: GdnPhase,
    stage: GdnStage,
    typed_outputs: bool,
    asset: assets::Asset,
    family: arithmetic::GdnArithmeticFamily,
    static_scalars: StaticScalarBlock,
) -> Result<LaunchSpec> {
    use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as Write};
    use GdnBuffer::*;
    let b = |role, io| binding(phase, role, io);
    let a = if phase.tokens() == 1 {
        view(PackedBa, 32, 32, R)
    } else {
        b(AContiguous, R)
    };
    let beta_input = if phase.tokens() == 1 {
        view(PackedBa, 0, 32, R)
    } else {
        b(BContiguous, R)
    };
    let mut bindings = match stage {
        GdnStage::Conv if phase == GdnPhase::CachedDecode => vec![
            b(PackedQkvz, RW),
            b(ConvWeights, R),
            b(CacheArena, RW),
            b(CacheIndices, R),
            b(PackedQkvz, RW),
        ],
        GdnStage::Conv => vec![
            b(PackedQkvz, R),
            b(ConvWeights, R),
            b(CacheArena, RW),
            b(CacheIndices, R),
            b(HasInitial, R),
            b(QueryStarts, R),
            b(ConvBatch, R),
            b(ConvChunkOffsets, R),
            b(ConvOutput, Write),
        ],
        GdnStage::Prep => vec![
            b(ConvOutput, R),
            a,
            beta_input,
            b(ALog, R),
            b(DtBias, R),
            b(Q, Write),
            b(K, Write),
            b(V, Write),
            b(GRaw, Write),
            b(Beta, Write),
        ],
        GdnStage::Cumsum => vec![
            b(GRaw, R),
            b(GCumulative, Write),
            b(QueryStarts, R),
            b(ChunkIndices, R),
        ],
        GdnStage::Kkt => vec![
            b(K, R),
            b(Beta, R),
            b(GCumulative, R),
            b(Kkt, Write),
            b(QueryStarts, R),
            b(ChunkIndices, R),
        ],
        GdnStage::Solve => vec![
            b(Kkt, R),
            b(Inverse, Write),
            b(QueryStarts, R),
            b(ChunkIndices, R),
        ],
        GdnStage::Recompute => vec![
            b(K, R),
            b(V, R),
            b(Beta, R),
            b(W, Write),
            b(U, Write),
            b(Inverse, R),
            b(GCumulative, R),
            b(QueryStarts, R),
            b(ChunkIndices, R),
        ],
        // The state kernel's v is recomputed U; output's v is VNew.
        GdnStage::H => vec![
            b(K, R),
            b(U, R),
            b(W, R),
            b(VNew, Write),
            b(GCumulative, R),
            b(HChunks, Write),
            b(InitialState, R),
            b(FinalState, Write),
            b(QueryStarts, R),
            b(ChunkOffsets, R),
        ],
        GdnStage::Output => vec![
            b(Q, R),
            b(K, R),
            b(VNew, R),
            b(HChunks, R),
            b(GCumulative, R),
            b(Core, Write),
            b(QueryStarts, R),
            b(ChunkIndices, R),
        ],
        GdnStage::Gated => {
            let z = if phase.tokens() == 1 {
                view(PackedQkvz, 6144 * 2, 2048 * 2, R)
            } else {
                b(ZCompact, R)
            };
            vec![
                b(Core, R),
                b(Gated, Write),
                b(GatedGamma, R),
                z,
                b(Rstd, Write),
            ]
        }
        GdnStage::Decode => {
            let state = view(
                CacheArena,
                CONV_STATE_BYTES,
                CacheArena.bytes(phase) - CONV_STATE_BYTES,
                RW,
            );
            vec![
                b(PackedQkvz, R),
                a,
                beta_input,
                b(ALog, R),
                b(DtBias, R),
                b(Core, Write),
                state,
                state,
                b(CacheIndices, R),
            ]
        }
    };
    if typed_outputs {
        for value in &mut bindings {
            if phase == GdnPhase::CachedDecode
                && matches!(stage, GdnStage::Conv | GdnStage::Decode)
                && value.role == PackedQkvz
            {
                // Selected cached kernels address exactly the first6144BF16;
                // source extent audit is required with any asset/grid change.
                value.role = ConvOutput;
                value.offset = 0;
                value.physical_bytes = ConvOutput.bytes(phase);
            } else if phase.tokens() == 1 && stage == GdnStage::Gated && value.role == PackedQkvz {
                value.role = ZCompact;
                value.offset = 0;
            }
        }
    }
    for value in &mut bindings {
        value.static_scalars = static_scalars;
        if value.physical_bytes == 0
            || value
                .offset
                .checked_add(value.physical_bytes)
                .is_none_or(|end| end > value.role.bytes(phase))
        {
            return Err(Error::State(
                "closed vLLM binding exceeds original allocation".into(),
            ));
        }
    }
    if !asset.source.starts_with(&format!(
        "// cubecl.cuda.dynamic_shared_bytes={}\n",
        asset.shared
    )) || !asset.source.contains(asset.ptx_sha256)
    {
        return Err(Error::State("pinned vLLM source/shared header".into()));
    }
    let arithmetic =
        arithmetic::StageArithmetic::for_family(family, phase, stage, &asset, &bindings)?;
    let kernel = PinnedKernel::new_with_static_scalars(
        asset.entry,
        Arc::from(asset.source),
        asset.threads,
        bindings
            .iter()
            .map(|value| (value.physical_bytes, value.io))
            .collect::<Vec<_>>()
            .into(),
        static_scalars,
    )?;
    Ok(LaunchSpec {
        arithmetic,
        kernel,
        grid: asset.grid,
        bindings,
    })
}

/// Pure source/binding inspection only; no production2B preparation is admitted.
#[cfg(test)]
fn kernel_qwen2b(phase: GdnPhase, stage: GdnStage, typed_outputs: bool) -> Result<LaunchSpec> {
    let asset = assets::select_qwen2b(phase, stage)?;
    // The original bindings/layouts are inspected independently of code family.
    let mut spec = kernel_impl(phase, stage, typed_outputs)?;
    spec.kernel = PinnedKernel::new(
        asset.entry,
        Arc::from(asset.source),
        asset.threads,
        spec.bindings
            .iter()
            .map(|value| (value.physical_bytes, value.io))
            .collect::<Vec<_>>()
            .into(),
    )?;
    spec.grid = asset.grid;
    spec.arithmetic =
        arithmetic::StageArithmetic::original2b(phase, stage, &asset, &spec.bindings)?;
    Ok(spec)
}

/// Allocation requirements only; no contents or source/state authority.
#[cfg(test)]
pub(crate) fn required_buffers(phase: GdnPhase) -> Result<Vec<(GdnBuffer, usize)>> {
    let mut roles = BTreeMap::new();
    for &stage in phase.stages() {
        for binding in kernel(phase, stage)?.bindings {
            roles.insert(binding.role, binding.role.bytes(phase));
        }
    }
    Ok(roles.into_iter().collect())
}

/// Source metadata for ONE actual sequence; never observation payloads.
pub(crate) fn metadata(phase: GdnPhase, slot: GdnCacheSlot) -> Result<Vec<(GdnBuffer, Vec<u8>)>> {
    use GdnBuffer::*;
    slot.validate(phase)?;
    let sequences = phase.sequences();
    let rows = phase.rows_per_sequence();
    let cache_indices = (0..sequences)
        .map(|lane| i32::try_from(lane + 1).map_err(|_| Error::Limit("GDN cache index".into())))
        .collect::<Result<Vec<_>>>()?;
    let packed = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut values = vec![(CacheIndices, packed(&cache_indices))];
    if phase == GdnPhase::CachedDecode {
        return Ok(values);
    }
    let mut batch = vec![-1_i32; 2048];
    let mut offsets = vec![-1_i32; 2048];
    let conv_per_sequence = rows.div_ceil(8);
    let mut cursor = 0usize;
    for lane in 0..sequences {
        for chunk in 0..conv_per_sequence {
            batch[cursor] = i32::try_from(lane)
                .map_err(|_| Error::Limit("GDN convolution batch index".into()))?;
            offsets[cursor] = i32::try_from(chunk)
                .map_err(|_| Error::Limit("GDN convolution chunk offset".into()))?;
            cursor += 1;
        }
    }
    let query_starts = (0..=sequences)
        .map(|lane| {
            lane.checked_mul(rows)
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| Error::Limit("GDN query-start extent".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let chunks_per_sequence = rows.div_ceil(64);
    let chunk_indices = (0..sequences)
        .flat_map(|lane| {
            (0..chunks_per_sequence).flat_map(move |chunk| {
                [
                    i32::try_from(lane).expect("bounded GDN sequence index"),
                    i32::try_from(chunk).expect("bounded GDN chunk index"),
                ]
            })
        })
        .collect::<Vec<_>>();
    let chunk_offsets = (0..=sequences)
        .map(|lane| {
            lane.checked_mul(chunks_per_sequence)
                .and_then(|value| i64::try_from(value).ok())
                .ok_or_else(|| Error::Limit("GDN chunk-offset extent".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    values.extend([
        (
            HasInitial,
            vec![u8::from(phase.has_initial_state()); sequences],
        ),
        (QueryStarts, packed(&query_starts)),
        (ConvBatch, packed(&batch)),
        (ConvChunkOffsets, packed(&offsets)),
        (ChunkIndices, packed(&chunk_indices)),
        (
            ChunkOffsets,
            chunk_offsets
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        ),
    ]);
    Ok(values)
}

#[cfg(all(
    test,
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
mod batch_qualification_tests {
    use super::*;

    #[test]
    #[ignore = "physical SM86 cached GDN gated B8-vs-two-B4 qualification"]
    fn physical_cached_gated_b8_matches_two_b4_launches_exactly() -> Result<()> {
        futures_lite::future::block_on(async {
            let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("cached-gdn-gated-b8")?;
            let service = diagnostic.service.clone();
            let client = service.client();
            let rows = 8usize;
            let half_rows = 4usize;
            let io_row_bytes = GdnBuffer::Core.bytes(GdnPhase::CachedDecode);
            let rstd_row_bytes = GdnBuffer::Rstd.bytes(GdnPhase::CachedDecode);
            let io_bytes = rows * io_row_bytes;
            let half_io_bytes = half_rows * io_row_bytes;
            let rstd_bytes = rows * rstd_row_bytes;
            let half_rstd_bytes = half_rows * rstd_row_bytes;

            let bf16 = |i: usize, seed: usize| -> u16 {
                match (i + seed) % 8 {
                    0 => 0x3f80,
                    1 => 0xbf00,
                    2 => 0x3e80,
                    3 => 0x4000,
                    4 => 0xbe00,
                    5 => 0x3f00,
                    6 => 0xc000,
                    _ => 0x0000,
                }
            };
            let bytes_from_words = |words: &[u16]| {
                words
                    .iter()
                    .flat_map(|word| word.to_ne_bytes())
                    .collect::<Vec<_>>()
            };
            let view =
                |handle: &cubecl::server::Handle, total: usize, offset: usize, bytes: usize| {
                    handle
                        .clone()
                        .offset_start(offset as u64)
                        .offset_end((total - offset - bytes) as u64)
                };

            let core_words = (0..io_bytes / 2).map(|i| bf16(i, 3)).collect::<Vec<_>>();
            let z_words = (0..io_bytes / 2).map(|i| bf16(i, 11)).collect::<Vec<_>>();
            let gamma_words = (0..128).map(|i| bf16(i, 19)).collect::<Vec<_>>();
            let core = client.create_from_slice(&bytes_from_words(&core_words));
            let z = client.create_from_slice(&bytes_from_words(&z_words));
            let gamma = client.create_from_slice(&bytes_from_words(&gamma_words));
            let seed_output = vec![0x5au8; io_bytes];
            let seed_rstd = vec![0xa5u8; rstd_bytes];
            let sequential_output = client.create_from_slice(&seed_output);
            let wide_output = client.create_from_slice(&seed_output);
            let sequential_rstd = client.create_from_slice(&seed_rstd);
            let wide_rstd = client.create_from_slice(&seed_rstd);

            let b4 = CachedGatedBatch::prepare(half_rows)?;
            for half in 0..2 {
                let io_offset = half * half_io_bytes;
                let rstd_offset = half * half_rstd_bytes;
                b4.launch(
                    client,
                    &view(&core, io_bytes, io_offset, half_io_bytes),
                    &view(&sequential_output, io_bytes, io_offset, half_io_bytes),
                    &gamma,
                    &view(&z, io_bytes, io_offset, half_io_bytes),
                    &view(&sequential_rstd, rstd_bytes, rstd_offset, half_rstd_bytes),
                )?;
            }
            CachedGatedBatch::prepare(rows)?.launch(
                client,
                &core,
                &wide_output,
                &gamma,
                &z,
                &wide_rstd,
            )?;

            client
                .sync()
                .await
                .map_err(|error| Error::State(format!("cached GDN gated B8 sync: {error}")))?;
            let sequential_output = client.read_one(sequential_output).map_err(|error| {
                Error::State(format!("cached GDN gated B4 output read: {error}"))
            })?;
            let wide_output = client.read_one(wide_output).map_err(|error| {
                Error::State(format!("cached GDN gated B8 output read: {error}"))
            })?;
            let sequential_rstd = client
                .read_one(sequential_rstd)
                .map_err(|error| Error::State(format!("cached GDN gated B4 rstd read: {error}")))?;
            let wide_rstd = client
                .read_one(wide_rstd)
                .map_err(|error| Error::State(format!("cached GDN gated B8 rstd read: {error}")))?;
            if sequential_output != wide_output || sequential_rstd != wide_rstd {
                return Err(Error::State(
                    "cached GDN gated B8 differs from two B4 launches".into(),
                ));
            }
            println!("CACHED_GDN_GATED_B8_PASS exact_output=true exact_rstd=true");

            client
                .memory_cleanup()
                .map_err(|error| Error::State(format!("cached GDN gated B8 cleanup: {error}")))?;
            service.close_model_namespace()?;
            Ok(())
        })
    }
}

#[cfg(all(
    test,
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
mod private_decode_tests;

#[cfg(all(test, feature = "cuda-native-qualification"))]
mod tests;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod fixed_prompt_component;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use fixed_prompt_component::{FixedPromptGdnArithmetic, FixedPromptGdnIntermediate};

#[cfg(target_os = "linux")]
mod text;
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
pub(crate) use text::GdnRuntimePartition;
#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
pub(crate) use text::qualify_zero_runtime_on_nvidia;
pub use text::{
    GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel, GdnRuntimeAccess,
    GdnRuntimeBinding, GdnRuntimeBuffer, GdnRuntimeBufferLayout, GdnRuntimeBufferOwnership,
    GdnRuntimeCacheSlot, GdnRuntimeConfig, GdnRuntimeControl, GdnRuntimeFinalState,
    GdnRuntimeGeometry, GdnRuntimeInitialState, GdnRuntimeKernelGeometry, GdnRuntimeMode,
    GdnRuntimePlan, GdnRuntimePrefill, GdnRuntimeScalarField, GdnRuntimeScalarType,
    GdnRuntimeScheduleKind, GdnRuntimeScheduleStep, GdnRuntimeStage, GdnRuntimeStageIdentity,
    GdnRuntimeStageKind, GdnRuntimeStateKind, GdnRuntimeStateView, GdnRuntimeWorkspaceLayout,
    GdnStateKernel, TextGdnImplementation,
};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use text::{TextGdnCode, TextGdnGeometry, TextGdnWorkspace};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use resident::prepare_state_code;
