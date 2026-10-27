#![cfg_attr(
    not(feature = "cuda-native-qualification"),
    allow(dead_code, unused_imports)
)]
//! Private SM86 .8B layer0 qualification; original generated instructions.
//! Runtime storage comes from the original resident owner, never a trace.
// Former names: VllmGdnPhase, short_hf, canonical47. Native ABI/source pins unchanged.
use crate::cuda::kernel::{PinnedKernel, StaticScalarBlock};
use cubecl_runtime::kernel::BufferIOAttr;
#[cfg(test)]
use std::collections::BTreeMap;
use std::sync::Arc;
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode};
use zerogravity_core::{Error, Result};

mod arithmetic;
pub(crate) use arithmetic::GdnFamilyIdentity;
mod assets;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod qwen2b_arithmetic;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use qwen2b_arithmetic::{Qwen2bGdnArithmetic, Qwen2bOriginalGdnReference};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnPhase {
    InitialOne,
    Initial64,
    ContinuedTwo,
    CachedDecode,
    FixedPrompt,
    /// One actual prompt with ceil(M/64) GDN chunks, never two model calls.
    TextPrefill(usize),
}
impl GdnPhase {
    pub(crate) fn for_reference_call(call: &SourceCall) -> Result<Self> {
        match (call.mode(), call.rows().base(), call.rows().len()) {
            (SourceCallMode::Prefill, 0, 1) => Ok(Self::InitialOne),
            (SourceCallMode::Prefill, 0, 64) => Ok(Self::Initial64),
            (SourceCallMode::Prefill, 64, 2) => Ok(Self::ContinuedTwo),
            (SourceCallMode::Decode, 1 | 66, 1) => Ok(Self::CachedDecode),
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
        }
    }
    pub(crate) fn chunks(self) -> usize {
        self.tokens().div_ceil(64)
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
                GdnPhase::InitialOne | GdnPhase::FixedPrompt | GdnPhase::TextPrefill(_),
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
            Self::CacheArena => CACHE_SLOTS * CACHE_SLOT_BYTES,
            Self::CacheIndices => 4,
            Self::HasInitial => 1,
            Self::QueryStarts => 8,
            Self::ChunkIndices => phase.chunks() * 8,
            Self::ConvBatch | Self::ConvChunkOffsets => 2048 * 4,
            Self::ChunkOffsets => 16,
            Self::ConvOutput => tokens * 6144 * 2,
            Self::GRaw | Self::Beta | Self::GCumulative | Self::Rstd => tokens * 16 * 4,
            Self::Kkt => tokens * 16 * 64 * 4,
            Self::Inverse => tokens * 16 * 64 * 2,
            Self::HChunks => phase.chunks() * 16 * 128 * 128 * 2,
            Self::InitialState | Self::FinalState => RECURRENT_STATE_BYTES,
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
                        .ok_or_else(|| Error::Limit("full24 GDN code extent".into()))?,
                )
                .and_then(|n| n.checked_add(4096 + PinnedKernel::KEY_METADATA_BYTES))
                .ok_or_else(|| Error::Limit("full24 GDN original code/metadata admission".into()))
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
fn kernel_impl(phase: GdnPhase, stage: GdnStage, typed_outputs: bool) -> Result<LaunchSpec> {
    kernel_from_asset(
        phase,
        stage,
        typed_outputs,
        assets::select(phase, stage)?,
        arithmetic::GdnArithmeticFamily::Original08,
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
    let mut values = vec![(CacheIndices, (slot.index() as i32).to_le_bytes().to_vec())];
    if phase == GdnPhase::CachedDecode {
        return Ok(values);
    }
    let mut batch = vec![-1_i32; 2048];
    let mut offsets = vec![-1_i32; 2048];
    for index in 0..phase.tokens().div_ceil(8) {
        batch[index] = 0;
        offsets[index] = index as i32;
    }
    let packed = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    values.extend([
        (HasInitial, vec![u8::from(phase.has_initial_state())]),
        (QueryStarts, packed(&[0, phase.tokens() as i32])),
        (ConvBatch, packed(&batch)),
        (ConvChunkOffsets, packed(&offsets)),
        (
            ChunkIndices,
            packed(
                &(0..phase.chunks())
                    .flat_map(|chunk| [0, chunk as i32])
                    .collect::<Vec<_>>(),
            ),
        ),
        (
            ChunkOffsets,
            [0_i64, phase.chunks() as i64]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        ),
    ]);
    Ok(values)
}

#[cfg(all(test, feature = "cuda-native-qualification"))]
mod tests;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod fixed_prompt_component;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use fixed_prompt_component::{FixedPromptGdnArithmetic, FixedPromptGdnIntermediate};

#[cfg(target_os = "linux")]
mod text;
#[cfg(target_os = "linux")]
pub use text::{
    GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel, GdnRuntimeAccess,
    GdnRuntimeBinding, GdnRuntimeBuffer, GdnRuntimeBufferLayout, GdnRuntimeBufferOwnership,
    GdnRuntimeCacheSlot, GdnRuntimeCacheView, GdnRuntimeConfig, GdnRuntimeControl,
    GdnRuntimeGeometry, GdnRuntimeInitialState, GdnRuntimeKernelGeometry, GdnRuntimeLaunch,
    GdnRuntimeMode, GdnRuntimePlan, GdnRuntimePrefill, GdnRuntimeScalarField, GdnRuntimeScalarType,
    GdnRuntimeStage, GdnRuntimeStageIdentity, GdnRuntimeStageKind, GdnRuntimeWorkspaceLayout,
    GdnStateKernel, TextGdnImplementation,
};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use text::{TextGdnCode, TextGdnGeometry, TextGdnWorkspace};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use resident::prepare_state_code;
