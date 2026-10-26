//! Closed .8B HF layer0 input RMS diagnostic; no route/profile capability.
//! Caller admits metadata and every stage buffer before constructing specs,
//! and retains original source/queue/flight leases through completion.
// Former names: Vllm08, short_hf, canonical47. Width/row guards unchanged.
use crate::cuda::kernel::PinnedKernel;
use cubecl_runtime::kernel::BufferIOAttr;
use std::{borrow::Cow, sync::Arc};
use zerogravity_core::{Error, Result};

const COLUMNS: usize = 1024;

#[path = "rms/columns.rs"]
mod columns;
#[cfg(target_os = "linux")]
#[path = "rms/leaf_source.rs"]
mod leaf_source;
use columns::Columns;
#[cfg(target_os = "linux")]
pub use leaf_source::RmsLeafSource;

#[cfg(target_os = "linux")]
#[path = "rms/qwen2b.rs"]
mod qwen2b;
#[cfg(target_os = "linux")]
pub use qwen2b::{
    Qwen2bDualRmsLeaf, Qwen2bRmsFault, Qwen2bRmsFaultOutcome, Qwen2bRmsReadback, Qwen2bRmsReceipt,
    qualify_qwen2b_rms, qualify_qwen2b_rms_fault,
};
const SOURCE: &str = include_str!("rms/kernels.cu");
#[cfg(all(feature = "qualification", feature = "cuda"))]
mod definitions;
#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
#[path = "rms/fixed_prompt.rs"]
mod fixed_prompt;
#[path = "rms/tail_mask.rs"]
mod tail_mask;
#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
pub use fixed_prompt::{
    FixedPromptDualRmsLeaf, FixedPromptRmsFault, FixedPromptRmsFaultOutcome,
    FixedPromptRmsReadback, FixedPromptRmsReceipt, qualify_fixed_prompt_rms,
    qualify_fixed_prompt_rms_fault,
};

#[cfg(all(feature = "qualification", feature = "cuda"))]
#[path = "rms/resident.rs"]
mod resident;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use resident::RmsPrepared;

#[cfg(all(feature = "qualification", feature = "cuda"))]
#[path = "rms/fused.rs"]
mod fused;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use fused::{FusedRmsPrepared, FusedRmsRetentionState, FusedRmsRetentionWitness};

#[cfg(all(feature = "qualification", feature = "cuda"))]
#[path = "rms/qualification.rs"]
mod qualification;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub use qualification::{InputRmsFixture, qualify_input_rms};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rows {
    One,
    Two,
    FixedPrompt,
    SixtyFour,
    /// Private finite source span; legacy numeric row admission is unchanged.
    Text {
        prompt_tokens: usize,
    },
    /// Physical axis of four separately bound fixed_prompt requests.
    CohortB4 {
        initial: bool,
    },
}
impl Rows {
    pub(crate) fn new(value: usize) -> Result<Self> {
        match value {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            64 => Ok(Self::SixtyFour),
            _ => Err(Error::Unsupported("observed vLLM input RMS rows".into())),
        }
    }
    pub(crate) const fn count(self) -> usize {
        match self {
            Self::CohortB4 { initial } => {
                if initial {
                    188
                } else {
                    4
                }
            }
            Self::One => 1,
            Self::Two => 2,
            Self::FixedPrompt => 47,
            Self::SixtyFour => 64,
            Self::Text { prompt_tokens } => prompt_tokens,
        }
    }
    // Reduce.cuh setReduceConfig<float,float,4,4>: input vectorization,
    // MAX_NUM_THREADS512; outputs split across logical block.y.
    const fn mean_block(self) -> (u32, u32) {
        match self {
            Self::CohortB4 { initial: true } => (32, 16),
            Self::CohortB4 { initial: false } => (128, 4),
            Self::One => (256, 1),
            Self::Two => (256, 2),
            Self::FixedPrompt | Self::SixtyFour | Self::Text { .. } => (32, 16),
        }
    }
}

/// One original ATen op/store boundary per stage; no fused RMS expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    GammaWiden,
    GammaShift,
    InputWiden,
    Square,
    Mean,
    Epsilon,
    Inverse,
    Normalize,
    Weight,
    Bf16Store,
}
impl Stage {
    pub(crate) const ORDER: [Self; 10] = [
        Self::GammaWiden,
        Self::GammaShift,
        Self::InputWiden,
        Self::Square,
        Self::Mean,
        Self::Epsilon,
        Self::Inverse,
        Self::Normalize,
        Self::Weight,
        Self::Bf16Store,
    ];
    const fn entry(self) -> &'static str {
        match self {
            Self::GammaWiden => "zg_vllm_input_gamma_widen",
            Self::GammaShift => "zg_vllm_input_gamma_shift",
            Self::InputWiden => "zg_vllm_input_widen",
            Self::Square => "zg_vllm_input_square",
            Self::Mean => "zg_vllm_input_mean",
            Self::Epsilon => "zg_vllm_input_epsilon",
            Self::Inverse => "zg_vllm_input_inverse",
            Self::Normalize => "zg_vllm_input_normalize",
            Self::Weight => "zg_vllm_input_weight",
            Self::Bf16Store => "zg_vllm_input_bf16_store",
        }
    }
    fn buffers(self, rows: Rows) -> Arc<[(usize, BufferIOAttr)]> {
        self.buffers_for(rows, Columns::Qwen08)
    }
    fn buffers_for(self, rows: Rows, columns: Columns) -> Arc<[(usize, BufferIOAttr)]> {
        let columns = columns.count();
        use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
        let dense_f32 = rows.count() * columns * 4;
        let dense_bf16 = rows.count() * columns * 2;
        let scalar_f32 = rows.count() * 4;
        match self {
            Self::GammaWiden => Arc::from([(columns * 2, R), (columns * 4, W)]),
            Self::GammaShift => Arc::from([(columns * 4, R), (columns * 4, W)]),
            Self::InputWiden => Arc::from([(dense_bf16, R), (dense_f32, W)]),
            Self::Square => Arc::from([(dense_f32, R), (dense_f32, W)]),
            Self::Mean => Arc::from([(dense_f32, R), (scalar_f32, W)]),
            Self::Epsilon | Self::Inverse => Arc::from([(scalar_f32, R), (scalar_f32, W)]),
            Self::Normalize => Arc::from([(dense_f32, R), (scalar_f32, R), (dense_f32, W)]),
            Self::Weight => Arc::from([(dense_f32, R), (columns * 4, R), (dense_f32, W)]),
            Self::Bf16Store => Arc::from([(dense_f32, R), (dense_bf16, W)]),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Spec {
    pub(crate) kernel: PinnedKernel,
    pub(crate) grid: [u32; 3],
}

/// No device allocation, stream, completion, source substitution or traversal API.
/// The source-derived kernel remains unqualified until genuine CUDA tests pass.
pub(crate) fn specification(stage: Stage, rows: Rows) -> Result<Spec> {
    specification_for(stage, rows, Columns::Qwen08)
}

fn rendered_source(rows: Rows, columns: Columns) -> Result<Arc<str>> {
    if matches!(rows, Rows::FixedPrompt | Rows::CohortB4 { initial: true })
        && columns != Columns::Qwen08
    {
        return Err(Error::Unsupported(
            "fixed_prompt RMS is original1024 only".into(),
        ));
    }
    let (mean_width, mean_height) = columns.mean_block(rows);
    let count = columns.count();
    let factor = columns.mean_factor();
    let body = if matches!(
        rows,
        Rows::FixedPrompt | Rows::Text { .. } | Rows::CohortB4 { initial: true }
    ) {
        Cow::Owned(tail_mask::fixed_prompt_source(SOURCE)?)
    } else {
        Cow::Borrowed(SOURCE)
    };
    Ok(format!(
        "#define ZG_ROWS {}\n#define ZG_COLUMNS {count}\n#define ZG_MEAN_FACTOR {factor}\n#define ZG_MEAN_WIDTH {mean_width}\n#define ZG_MEAN_HEIGHT {mean_height}\n{body}",
        rows.count(),
    ).into())
}

fn specification_for(stage: Stage, rows: Rows, columns: Columns) -> Result<Spec> {
    let source = rendered_source(rows, columns)?;
    let (mean_width, mean_height) = columns.mean_block(rows);
    let count = columns.count();
    let (threads, grid) = if stage == Stage::Mean {
        (
            mean_width * mean_height,
            [rows.count().div_ceil(mean_height as usize) as u32, 1, 1],
        )
    } else {
        let words = match stage {
            Stage::GammaWiden | Stage::GammaShift => count,
            Stage::Epsilon | Stage::Inverse => rows.count(),
            _ => rows.count() * count,
        };
        (256, [words.div_ceil(256) as u32, 1, 1])
    };
    Ok(Spec {
        kernel: PinnedKernel::new(
            stage.entry(),
            source,
            threads,
            if columns == Columns::Qwen08 {
                stage.buffers(rows)
            } else {
                stage.buffers_for(rows, columns)
            },
        )?,
        grid,
    })
}

#[cfg(test)]
#[path = "rms/tests.rs"]
mod tests;

#[cfg(target_os = "linux")]
mod layer_binding;
#[cfg(target_os = "linux")]
pub(crate) use layer_binding::BoundRmsSource;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod final_binding;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub use final_binding::BoundFinalRmsSource;

mod model_binding;
pub(crate) use model_binding::BoundModelRmsSource;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "rms/pooled.rs"]
mod pooled;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub(crate) use pooled::{
    FullModelRmsScratch, RmsScratchRun, RmsWorkspaceOutput, RmsWorkspaceViews,
};

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "rms/cohort.rs"]
pub(crate) mod cohort;
