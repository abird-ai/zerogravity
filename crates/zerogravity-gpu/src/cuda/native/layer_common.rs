//! Shared authentic layer records/source loading; no execution provider.
use super::*;
use crate::cuda::kernel::PinnedKernel;
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;
use zerogravity_artifacts::source_weights::TensorWindow;
use zerogravity_core::accelerator::program::MatrixAxes;

#[cfg(test)]
#[path = "layer/tests.rs"]
mod tests;

/// Native named outputs, used only after the connected producer flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerOutput {
    Normalized,
    Qkvz,
    Ba,
    Convolution,
    Core,
    Gated,
    Projected,
    ConvolutionState,
    RecurrentState,
    Q,
    K,
    V,
    LogGate,
    Beta,
    GCumulative,
    W,
    U,
    HChunks,
    VNew,
    FinalState,
    PostNormSumF32,
    PostNormalized,
    PostResidual,
    GateUp,
    Activation,
    MlpOutput,
}
impl LayerOutput {
    const COMMON: [Self; 9] = [
        Self::Normalized,
        Self::Qkvz,
        Self::Ba,
        Self::Convolution,
        Self::Core,
        Self::Gated,
        Self::Projected,
        Self::ConvolutionState,
        Self::RecurrentState,
    ];
    const PREFILL: [Self; 11] = [
        Self::Q,
        Self::K,
        Self::V,
        Self::LogGate,
        Self::Beta,
        Self::GCumulative,
        Self::W,
        Self::U,
        Self::HChunks,
        Self::VNew,
        Self::FinalState,
    ];
    pub(super) fn required(phase: GdnPhase) -> Vec<Self> {
        let mut values = Self::COMMON.to_vec();
        if phase != GdnPhase::CachedDecode {
            values.extend(Self::PREFILL);
        }
        values
    }
    pub(super) fn word_bytes(self) -> usize {
        match self {
            Self::RecurrentState
            | Self::LogGate
            | Self::Beta
            | Self::GCumulative
            | Self::FinalState
            | Self::PostNormSumF32 => 4,
            _ => 2,
        }
    }
    pub(super) fn bytes(self, phase: GdnPhase) -> usize {
        let tokens = phase.tokens();
        let words = match self {
            Self::Normalized
            | Self::Projected
            | Self::PostNormSumF32
            | Self::PostNormalized
            | Self::PostResidual
            | Self::MlpOutput => tokens * 1024,
            Self::GateUp => tokens * 7168,
            Self::Activation => tokens * 3584,
            Self::Qkvz => tokens * 8192,
            Self::Ba => tokens * 32,
            Self::Convolution => tokens * 6144,
            Self::ConvolutionState => 3 * 6144,
            Self::RecurrentState | Self::FinalState | Self::HChunks => 16 * 128 * 128,
            Self::LogGate | Self::Beta | Self::GCumulative => tokens * 16,
            _ => tokens * 2048,
        };
        words * self.word_bytes()
    }
    pub(super) fn first_difference(
        self,
        phase: GdnPhase,
        actual: &[u8],
        expected: &[u8],
    ) -> Result<Option<usize>> {
        let bytes = self.bytes(phase);
        if actual.len() != bytes || expected.len() != bytes {
            return Err(Error::Malformed("complete native output comparison".into()));
        }
        let word = self.word_bytes();
        // Captured convolution is logical [6144,T], strides [1,6144].
        // Its original kernel storage is [T,6144]; compare logical elements.
        if self == Self::Convolution && phase.tokens() > 1 {
            let tokens = phase.tokens();
            return Ok((0..6144 * tokens).find(|&logical| {
                let physical = (logical % tokens) * 6144 + logical / tokens;
                actual[physical * word..(physical + 1) * word]
                    != expected[logical * word..(logical + 1) * word]
            }));
        }
        Ok(actual
            .chunks_exact(word)
            .zip(expected.chunks_exact(word))
            .position(|(a, b)| a != b))
    }
    pub(super) fn role(self) -> Option<GdnBuffer> {
        Some(match self {
            Self::Ba => GdnBuffer::PackedBa,
            Self::Core => GdnBuffer::Core,
            Self::Gated => GdnBuffer::Gated,
            Self::Q => GdnBuffer::Q,
            Self::K => GdnBuffer::K,
            Self::V => GdnBuffer::V,
            Self::LogGate => GdnBuffer::GRaw,
            Self::Beta => GdnBuffer::Beta,
            Self::GCumulative => GdnBuffer::GCumulative,
            Self::W => GdnBuffer::W,
            Self::U => GdnBuffer::U,
            Self::HChunks => GdnBuffer::HChunks,
            Self::VNew => GdnBuffer::VNew,
            Self::FinalState => GdnBuffer::FinalState,
            _ => return None,
        })
    }
}
pub struct LayerExpectedRecord {
    pub output: LayerOutput,
    pub bytes: Vec<u8>,
}
/// Actual teacher IDs are the only non-source mathematical inputs.
/// Expectations cannot be bound as device inputs by this interface.
pub struct LayerFixture {
    pub teacher_ids: Vec<u32>,
    pub expected: Vec<LayerExpectedRecord>,
}
pub(super) struct Snapshot {
    pub(super) lease: ResidentLease,
    kernel: PinnedKernel,
    handles: [Handle; 2],
}
impl Snapshot {
    pub(super) fn prepare(
        service: &Arc<ResidentService>,
        source: &ResidentLease,
        lease: ResidentLease,
    ) -> Result<Self> {
        if source.bytes() != 16384 || lease.bytes() != 16384 || source.id() == lease.id() {
            return Err(Error::State(
                "closed computed QKVZ snapshot geometry".into(),
            ));
        }
        let text = "// cubecl.cuda.dynamic_shared_bytes=0\n#if __CUDA_ARCH__ != 860\n#error Closed SM86 snapshot\n#endif\nextern \"C\" __global__ __launch_bounds__(128) void zg_vllm_qkvz_snapshot(const unsigned int* input, unsigned int* output) { unsigned int index=blockIdx.x*blockDim.x+threadIdx.x; if(index<4096u) output[index]=input[index]; }\n";
        let kernel = PinnedKernel::new(
            "zg_vllm_qkvz_snapshot",
            Arc::from(text),
            128,
            Arc::from([
                (16384, BufferIOAttr::ReadOnly),
                (16384, BufferIOAttr::WriteOnly),
            ]),
        )?;
        let handles = [source.handle(), lease.handle()];
        kernel.validate(service.client(), [32, 1, 1], &handles)?;
        Ok(Self {
            lease,
            kernel,
            handles,
        })
    }
    pub(super) fn launch(&self, service: &Arc<ResidentService>) -> Result<()> {
        self.kernel
            .launch(service.client(), [32, 1, 1], &self.handles)
    }
}
pub(super) fn finite_f32(raw: &[u8]) -> bool {
    raw.len().is_multiple_of(4)
        && raw
            .chunks_exact(4)
            .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).is_finite())
}
pub(super) type OriginalGdnParameters = (Vec<Arc<ParameterWindow>>, [Vec<u8>; 4]);

pub(super) fn load_parameters(source: &VerifiedTensorSource) -> Result<OriginalGdnParameters> {
    let mut windows = Vec::new();
    let mut values: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
    for (index, suffix, elements, kind) in [
        (0, "conv1d.weight", 6144 * 4, QuantKind::Bf16),
        (1, "A_log", 16, QuantKind::F32),
        (2, "dt_bias", 16, QuantKind::Bf16),
        (3, "norm.weight", 128, QuantKind::F32),
    ] {
        let name = format!("model.language_model.layers.0.linear_attn.{suffix}");
        let parameter = if index == 0 {
            source.conv4_parameter(&name)?
        } else {
            source.parameter(&name)?
        };
        let shape: &[u64] = match index {
            0 => &[6144, 1, 4],
            3 => &[128],
            _ => &[16],
        };
        if parameter.elements() != elements
            || parameter.tensor().kind != kind
            || parameter.tensor().shape != shape
        {
            return Err(Error::Malformed(
                "original HF layer0 GDN parameter geometry/dtype".into(),
            ));
        }
        let window = Arc::new(source.read_parameter(&parameter, 0..elements)?);
        if !window.is_complete() {
            return Err(Error::State(
                "original complete GDN parameter window".into(),
            ));
        }
        values[index] = if index == 3 {
            loaded_gamma(&window)?
        } else {
            window.as_slice().to_vec()
        };
        let finite = if index == 1 {
            finite_f32(&values[index])
        } else {
            finite_bf16(&values[index])
        };
        if !finite {
            return Err(Error::Malformed(
                "finite original layer0 GDN parameters".into(),
            ));
        }
        windows.push(window);
    }
    Ok((windows, values))
}

/// Original TP1 flat Q/K/V/Z packing, with no LoRA or quantization.
pub(super) fn qkvz_weights(
    source: &VerifiedTensorSource,
) -> Result<(Vec<Arc<TensorWindow>>, Vec<u8>)> {
    let mut windows = Vec::with_capacity(2);
    let mut weights = Vec::with_capacity(8192 * 1024 * 2);
    for (suffix, rows) in [("in_proj_qkv", 6144), ("in_proj_z", 2048)] {
        let name = format!("model.language_model.layers.0.linear_attn.{suffix}.weight");
        let matrix = source.matrix(&name, MatrixAxes::OutputInput)?;
        if matrix.rows() != rows
            || matrix.columns() != 1024
            || matrix.tensor().kind != QuantKind::Bf16
        {
            return Err(Error::Malformed(
                "original pinned BF16 QKVZ geometry".into(),
            ));
        }
        let window = Arc::new(source.read_rows(&matrix, 0..rows)?);
        if window.as_slice().len() != rows * 1024 * 2 || !finite_bf16(window.as_slice()) {
            return Err(Error::Malformed(
                "complete finite original QKVZ rows".into(),
            ));
        }
        weights.extend_from_slice(window.as_slice());
        windows.push(window);
    }
    Ok((windows, weights))
}

/// Original TP1 flat Q/K/V/Z, B/A and output projection source packing.
pub(super) fn projection_weight(
    source: &VerifiedTensorSource,
    index: usize,
) -> Result<(Vec<Arc<TensorWindow>>, Vec<u8>)> {
    if index == 0 {
        return qkvz_weights(source);
    }
    let tensors: &[(&str, usize, usize)] = match index {
        1 => &[("in_proj_b", 16, 1024), ("in_proj_a", 16, 1024)],
        2 => &[("out_proj", 1024, 2048)],
        _ => {
            return Err(Error::Unsupported(
                "closed original projection source".into(),
            ));
        }
    };
    let bytes = tensors
        .iter()
        .map(|(_, rows, columns)| rows * columns * 2)
        .sum();
    let mut weights = Vec::with_capacity(bytes);
    let mut windows = Vec::with_capacity(tensors.len());
    for &(suffix, rows, columns) in tensors {
        let name = format!("model.language_model.layers.0.linear_attn.{suffix}.weight");
        let matrix = source.matrix(&name, MatrixAxes::OutputInput)?;
        if matrix.rows() != rows
            || matrix.columns() != columns
            || matrix.tensor().kind != QuantKind::Bf16
        {
            return Err(Error::Malformed(
                "original pinned BF16 GDN projection geometry".into(),
            ));
        }
        let window = Arc::new(source.read_rows(&matrix, 0..rows)?);
        if window.as_slice().len() != rows * columns * 2 || !finite_bf16(window.as_slice()) {
            return Err(Error::Malformed(
                "complete finite original GDN projection rows".into(),
            ));
        }
        weights.extend_from_slice(window.as_slice());
        windows.push(window);
    }
    Ok((windows, weights))
}
