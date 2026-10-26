//! CPU/optional-region adapter for the single generic forward schedule.
//! The existing QwenState remains private and unchanged in representation.

use super::{
    LayerObservation, LayerState, QwenDenseCpu, QwenState, observe, source_gdn_recurrent_step,
};
use crate::accelerator::program::{
    ActivationPrecision, AttentionGeometry, CoefficientValues, ConvDomain, EmbeddingContract,
    KvDomain, MathematicalBackend, OrdinaryRmsContract, ProgramNumericalProfile,
    ProjectionContract, QkValues, RecurrentDomain, RopeGeometry, RopeValues, SourceParameter,
    SourceTensor, SourceVector, StateSlot, TensorView,
};
use crate::accelerator::{
    AttentionQkRegion, AttentionSoftmaxRegion, AttentionValueRegion, BinaryOperation, BinaryRegion,
    GdnCoefficientRegion, GdnCoefficients, GdnRecurrentExecution, GdnRecurrentRegion, RopeKvRegion,
};
use crate::operator_slice::OperatorSliceSource;
use crate::{Error, Result};
use std::sync::Arc;
#[cfg(all(feature = "source-binding", target_os = "linux"))]
use zerogravity_artifacts::source_weights::WindowWorkspace;

struct HostStorage {
    values: Vec<f32>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    bulk: Option<bulk_f32::BulkBacking>,
    // Payload drops before its actual same-source host reservation.
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    workspace: Option<Arc<WindowWorkspace>>,
}
impl std::ops::Deref for HostStorage {
    type Target = Vec<f32>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

enum HostSource<'a> {
    Loaded(&'a super::LoadedQwenModel),
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    Windows(&'a crate::source::binding::OriginalWindowOperators),
}
impl HostSource<'_> {
    fn metadata(&self) -> &dyn OperatorSliceSource {
        match self {
            Self::Loaded(source) => *source,
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            Self::Windows(source) => *source,
        }
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.metadata().source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        self.metadata().program_numerical_profile()
    }
    fn rms_epsilon(&self) -> f32 {
        self.metadata().rms_epsilon()
    }
    fn matrix_descriptor(
        &self,
        role: crate::source::binding::QwenMatrixRole,
    ) -> Result<SourceTensor> {
        self.metadata().matrix_descriptor(role)
    }
    fn validate_tensor(&self, tensor: &SourceTensor) -> Result<()> {
        match self {
            Self::Loaded(source) => tensor.validate_source(*source),
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            Self::Windows(source) => source.validate_matrix(tensor),
        }
    }
}

type HostKvView<'a> = (&'a [Arc<[f32]>], &'a [Arc<[f32]>]);

/// Host views share a Vec allocation without cloning numerical payload. A
/// strided view is packed explicitly by contiguous(), never by hidden math.
#[derive(Clone)]
pub(super) struct HostValue {
    data: Arc<HostStorage>,
    view: TensorView,
    elements: usize,
    precision: ActivationPrecision,
    source: Option<crate::accelerator::SourceIdentity>,
    profile: Option<ProgramNumericalProfile>,
}
impl HostValue {
    fn new(values: Vec<f32>) -> Self {
        let elements = values.len();
        Self {
            data: Arc::new(HostStorage {
                values,
                #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                bulk: None,
                #[cfg(all(feature = "source-binding", target_os = "linux"))]
                workspace: None,
            }),
            view: TensorView::vector(0, elements),
            elements,
            precision: ActivationPrecision::F32,
            source: None,
            profile: None,
        }
    }
    fn typed(
        values: Vec<f32>,
        precision: ActivationPrecision,
        source: crate::accelerator::SourceIdentity,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        if !source.is_bound() {
            return Err(Error::State("unbound typed host activation".into()));
        }
        precision.validate_values(&values)?;
        let mut value = Self::new(values);
        value.precision = precision;
        value.source = Some(source);
        value.profile = Some(profile);
        Ok(value)
    }
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    fn retain_workspace(mut self, workspace: Option<Arc<WindowWorkspace>>) -> Result<Self> {
        Arc::get_mut(&mut self.data)
            .ok_or_else(|| Error::State("new host backing unexpectedly shared".into()))?
            .workspace = workspace;
        Ok(self)
    }
    pub(super) fn as_slice(&self) -> Result<&[f32]> {
        if !self.view.is_contiguous() {
            return Err(Error::State(
                "host operator requires explicit contiguous input".into(),
            ));
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let values = match &self.data.bulk {
            Some(backing) => backing.values()?,
            None => &self.data.values,
        };
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        let values = &self.data.values;
        values
            .get(
                self.view.offset
                    ..self
                        .view
                        .offset
                        .checked_add(self.elements)
                        .ok_or_else(|| Error::Limit("host view end overflow".into()))?,
            )
            .ok_or_else(|| Error::State("invalid private host tensor view".into()))
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn same_backing(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            && self.view == other.view
            && self.elements == other.elements
    }
    pub fn into_vec(self) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if self.data.bulk.is_some() {
            return Err(Error::Unsupported(
                "leased bulk result has no Vec escape".into(),
            ));
        }
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        if self.data.workspace.is_some() {
            return Err(Error::Unsupported(
                "leased window values require borrowed output access".into(),
            ));
        }
        if self.view.offset == 0 && self.view.is_contiguous() && self.elements == self.data.len() {
            return Ok(match Arc::try_unwrap(self.data) {
                Ok(data) => data.values,
                Err(data) => data.values.clone(),
            });
        }
        Ok(self.as_slice()?.to_vec())
    }
}

pub(super) struct HostMath<'engine, 'state, 'trace> {
    source: HostSource<'engine>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    bulk_call: Option<bulk_f32::BulkCall<'engine>>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    zen_ssm_out_call: Option<zen_ssm_out::ZenSsmOutCall<'engine>>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    gguf_prefix_call: Option<gguf_prefix::GgufPrefixCall<'engine>>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    gguf_gdn_upstream_call: Option<gdn_upstream::UpstreamCall<'engine>>,
    engine: Option<&'engine QwenDenseCpu>,
    state: Option<&'state mut QwenState>,
    region_kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
    trace: Option<&'trace mut Vec<LayerObservation>>,
    position: usize,
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    pub(crate) initial_core_tape:
        Option<Arc<std::sync::Mutex<crate::source::binding::InitialCoreTape>>>,
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    cancelled: Option<&'engine std::sync::atomic::AtomicBool>,
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub fn new(
        engine: &'engine QwenDenseCpu,
        state: &'state mut QwenState,
        trace: Option<&'trace mut Vec<LayerObservation>>,
    ) -> Self {
        let position = state.position;
        Self {
            source: HostSource::Loaded(engine.model.as_ref()),
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            bulk_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            zen_ssm_out_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            gguf_prefix_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            gguf_gdn_upstream_call: None,
            engine: Some(engine),
            state: Some(state),
            region_kernel: engine.kernel.as_deref(),
            trace,
            position,
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            cancelled: None,
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            initial_core_tape: None,
        }
    }
    pub fn for_operator_slice(
        source: &'engine super::LoadedQwenModel,
        kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
    ) -> Self {
        Self {
            source: HostSource::Loaded(source),
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            bulk_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            zen_ssm_out_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            gguf_prefix_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            gguf_gdn_upstream_call: None,
            engine: None,
            state: None,
            region_kernel: kernel,
            trace: None,
            position: 0,
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            cancelled: None,
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            initial_core_tape: None,
        }
    }
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    pub fn for_window_operator_slice(
        source: &'engine crate::source::binding::OriginalWindowOperators,
        cancelled: Option<&'engine std::sync::atomic::AtomicBool>,
        kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
    ) -> Self {
        Self {
            source: HostSource::Windows(source),
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            bulk_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            zen_ssm_out_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            gguf_prefix_call: None,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            gguf_gdn_upstream_call: None,
            engine: None,
            state: None,
            region_kernel: kernel,
            trace: None,
            position: 0,
            cancelled,
            initial_core_tape: None,
        }
    }
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    fn prepare_output(
        &self,
        input: Option<&HostValue>,
        output_elements: usize,
        scratch_elements: usize,
    ) -> Result<Option<Arc<WindowWorkspace>>> {
        let HostSource::Windows(source) = &self.source else {
            return Ok(None);
        };
        if self
            .cancelled
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State(
                "window host operation cancelled before admission".into(),
            ));
        }
        if let Some(input) = input {
            let prior = input.data.workspace.as_ref().ok_or_else(|| {
                Error::State("window activation has no retained physical backing".into())
            })?;
            if !source.source().same_issuer(prior.source()) {
                return Err(Error::State(
                    "window activation belongs to another source issuer".into(),
                ));
            }
        }
        Ok(Some(Arc::new(
            source.workspace(output_elements, scratch_elements)?,
        )))
    }
    /// Explicit operator input transport, never a model producer or root.
    pub(super) fn import_attention_input(
        &self,
        input: crate::operator_slice::attention::AttentionSliceInput<'_>,
        hidden: usize,
    ) -> Result<HostValue> {
        input.validate()?;
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "typed attention CPU input provider".into(),
            ));
        }
        if input.source_identity() != self.source.source_identity()
            || input.numerical_profile() != self.source.program_numerical_profile()
            || self.source.metadata().geometry().hidden_size != hidden as u64
            || input.bytes().len()
                != hidden
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("attention private input width".into()))?
        {
            return Err(Error::State(
                "attention input source/profile/full hidden width".into(),
            ));
        }
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let workspace = self.prepare_output(None, hidden, 1)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(hidden)
            .map_err(|_| Error::Limit("private attention input transport".into()))?;
        values.extend(
            input.bytes().chunks_exact(2).map(|word| {
                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
            }),
        );
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        if self
            .cancelled
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State(
                "attention input cancelled after transport".into(),
            ));
        }
        let output = HostValue::typed(
            values,
            input.precision(),
            input.source_identity(),
            input.numerical_profile(),
        )?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn engine(&self) -> Result<&'engine QwenDenseCpu> {
        self.engine
            .ok_or_else(|| Error::Unsupported("operator slice has no session engine".into()))
    }
    fn state(&self) -> Result<&QwenState> {
        self.state
            .as_deref()
            .ok_or_else(|| Error::State("operator slice has no session state".into()))
    }
    fn state_mut(&mut self) -> Result<&mut QwenState> {
        self.state
            .as_deref_mut()
            .ok_or_else(|| Error::State("operator slice has no session state".into()))
    }
    fn legacy_precision(&self, values: &[&HostValue]) -> Result<()> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if values.iter().any(|v| v.data.bulk.is_some()) {
            return Err(Error::Unsupported(
                "bulk F32 values require explicit matrix contract".into(),
            ));
        }
        if self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
            || values
                .iter()
                .any(|value| value.precision != ActivationPrecision::F32)
        {
            return Err(Error::Unsupported(
                "legacy host operation lacks explicit numerical contract".into(),
            ));
        }
        Ok(())
    }
    fn tensor(&self, tensor: &SourceTensor) -> Result<()> {
        self.source.validate_tensor(tensor)
    }
    fn vector(&self, vector: SourceVector<'_>) -> Result<()> {
        if vector.source != self.source.source_identity() {
            return Err(Error::State(
                "host forward static vector changed source".into(),
            ));
        }
        Ok(())
    }
    fn kv(&self, domain: KvDomain, tokens: usize) -> Result<HostKvView<'_>> {
        match self.state()?.layers.get(domain.0) {
            Some(LayerState::Full { keys, values })
                if keys.len() == tokens && values.len() == tokens =>
            {
                Ok((keys, values))
            }
            _ => Err(Error::State(
                "attention KV prefix does not match source position".into(),
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)] // Mirrors already qualified closed-op inputs.
impl MathematicalBackend for HostMath<'_, '_, '_> {
    type Value = HostValue;
    type Index = usize;
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    fn torch_cpu_full_convolution_cache(
        &mut self,
        prior: &HostValue,
        minimal: &HostValue,
        input: &HostValue,
    ) -> Result<HostValue> {
        self.window_full_convolution_cache(prior, minimal, input)
    }
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    fn torch_cpu_bf16_gdn(
        &mut self,
        c: &crate::accelerator::program::TorchCpuGdnContract,
        input: crate::accelerator::program::TorchCpuGdnInputs<'_, HostValue>,
    ) -> Result<crate::accelerator::program::TorchCpuGdnValues<HostValue>> {
        self.execute_cpu_gdn(c, input)
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn source_operator(
        &mut self,
        actual: &crate::accelerator::program::SourceOperator,
        inputs: &[HostValue],
    ) -> Result<Vec<HostValue>> {
        self.execute_source_f32(actual, inputs)
            .map(|value| vec![value])
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn bulk_f32_input(
        &mut self,
        operand: &crate::accelerator::program::BulkF32Operand,
    ) -> Result<HostValue> {
        self.import_bulk_f32(operand)
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn bulk_f32_matmul(
        &mut self,
        matrix: &crate::accelerator::program::BulkF32Matrix,
        left: &HostValue,
        right: &HostValue,
    ) -> Result<HostValue> {
        self.execute_bulk_f32(matrix, left, right)
    }
    fn elements(&self, value: &HostValue) -> usize {
        value.elements
    }
    fn precision(&self, value: &HostValue) -> ActivationPrecision {
        value.precision
    }
    fn embed_profiled(
        &mut self,
        tensor: SourceTensor,
        token: usize,
        contract: EmbeddingContract,
    ) -> Result<HostValue> {
        if contract == EmbeddingContract::Source {
            return self.embed(tensor, token);
        }
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider does not declare typed CPU profile".into(),
            ));
        }
        self.tensor(&tensor)?;
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || tensor.kind != zerogravity_core::quant::QuantKind::Bf16
            || tensor.axes != crate::accelerator::program::MatrixAxes::OutputInput
            || tensor
                != self
                    .source
                    .matrix_descriptor(crate::source::binding::QwenMatrixRole::Embedding)?
        {
            return Err(Error::Unsupported(
                "host typed embedding source/contract".into(),
            ));
        }
        let token = u32::try_from(token)
            .map_err(|_| Error::Malformed("typed embedding token domain".into()))?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let workspace = self.prepare_output(None, tensor.columns, 1)?;
        let values = match &self.source {
            HostSource::Loaded(source) => zerogravity_cpu::source::source_reference_embedding(
                source.embedding_source(token)?,
            )?,
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            HostSource::Windows(source) => source.embed(&tensor, token as usize, self.cancelled)?,
        };
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn project_profiled(
        &mut self,
        tensor: SourceTensor,
        input: &HostValue,
        contract: ProjectionContract,
    ) -> Result<HostValue> {
        if contract == ProjectionContract::Source {
            return self.project(tensor, input);
        }
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider does not declare typed CPU profile".into(),
            ));
        }
        self.tensor(&tensor)?;
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || input.precision != ActivationPrecision::Bf16WidenedF32
            || input.source != Some(self.source.source_identity())
            || input.profile != Some(self.source.program_numerical_profile())
            || tensor.kind != zerogravity_core::quant::QuantKind::Bf16
            || tensor.axes != crate::accelerator::program::MatrixAxes::OutputInput
        {
            return Err(Error::Unsupported(
                "host typed projection source/contract".into(),
            ));
        }
        ActivationPrecision::Bf16WidenedF32.validate_values(input.as_slice()?)?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        if let HostSource::Windows(source) = &self.source {
            source.validate_projection_matrix(&tensor)?;
        }
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let workspace = self.prepare_output(
            Some(input),
            tensor.rows,
            tensor
                .rows
                .checked_add(128)
                .ok_or_else(|| Error::Limit("projection scratch elements".into()))?,
        )?;
        let values = match &self.source {
            HostSource::Loaded(source) => {
                zerogravity_cpu::source::projection::output_first_projection_with_profile(
                    self.source.source_identity(),
                    &tensor.name,
                    zerogravity_core::source::TensorSource::tensor(*source, &tensor.name)?,
                    input.as_slice()?,
                    self.region_kernel,
                    zerogravity_cpu::source::SourceProjectionProfile::TorchCpuBf16Avx2,
                )?
            }
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            HostSource::Windows(source) => {
                source.project(&tensor, input.as_slice()?, self.cancelled)?
            }
        };
        if values.len() != tensor.rows {
            return Err(Error::State(
                "complete typed projection output length".into(),
            ));
        }
        let mut rounded = Vec::new();
        rounded
            .try_reserve_exact(tensor.rows)
            .map_err(|_| Error::Limit("complete typed BF16 output allocation".into()))?;
        if rounded.capacity() != tensor.rows {
            return Err(Error::Limit(
                "typed BF16 output exceeds admitted capacity".into(),
            ));
        }
        for value in values {
            rounded.push(round_bf16(value)?);
        }
        let values = rounded;
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn ordinary_rms_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &HostValue,
        contract: OrdinaryRmsContract,
    ) -> Result<HostValue> {
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider does not declare typed ordinary RMS contract".into(),
            ));
        }
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || self.source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
            || input.precision != ActivationPrecision::Bf16WidenedF32
            || input.source != Some(self.source.source_identity())
            || input.profile != Some(self.source.program_numerical_profile())
        {
            return Err(Error::Unsupported(
                "host ordinary RMS numerical profile/epsilon/source/precision".into(),
            ));
        }
        let width = contract.validate(&parameter, input.elements)?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let scratch = input
            .elements
            .checked_mul(5)
            .and_then(|n| width.checked_mul(2).and_then(|w| n.checked_add(w)))
            .and_then(|n| {
                (input.elements / width)
                    .checked_mul(4)
                    .and_then(|r| n.checked_add(r))
            })
            .ok_or_else(|| Error::Limit("ordinary RMS scratch elements".into()))?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let workspace = self.prepare_output(Some(input), input.elements, scratch)?;
        let values = match &self.source {
            HostSource::Loaded(source) => {
                parameter.validate_source(*source)?;
                let raw = zerogravity_core::source::TensorSource::tensor(*source, &parameter.name)?;
                let output = zerogravity_cpu::numerics::torch_cpu::norm::ordinary_rms(
                    self.source.source_identity(),
                    raw,
                    input.as_slice()?,
                    width,
                )?;
                output.output_bf16().to_vec()
            }
            #[cfg(all(feature = "source-binding", target_os = "linux"))]
            HostSource::Windows(source) => {
                source.normalize(&parameter, input.as_slice()?, self.cancelled)?
            }
        };
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn causal_conv_silu_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &HostValue,
        history: &HostValue,
        contract: crate::accelerator::program::CausalConvSiluContract,
    ) -> Result<crate::accelerator::program::CausalConvValues<HostValue>> {
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        {
            self.execute_causal_conv_silu(parameter, input, history, contract)
        }
        #[cfg(not(all(feature = "source-binding", target_os = "linux")))]
        {
            let _ = (parameter, input, history, contract);
            Err(Error::Unsupported(
                "causal convolution requires leased original windows".into(),
            ))
        }
    }
    fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
        self.legacy_precision(&[])?;
        self.tensor(&tensor)?;
        if tensor.name != "token_embd.weight" {
            return Err(Error::State("unexpected Qwen embedding role".into()));
        }
        let token = u32::try_from(token)
            .map_err(|_| Error::Malformed("embedding ID exceeds source token domain".into()))?;
        Ok(HostValue::new(self.engine()?.embedding_region(token)?))
    }
    fn project(&mut self, tensor: SourceTensor, input: &HostValue) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.tensor(&tensor)?;
        Ok(HostValue::new(
            self.engine()?.project(&tensor.name, input.as_slice()?)?,
        ))
    }
    fn view(&mut self, input: &HostValue, view: TensorView) -> Result<HostValue> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if input.data.bulk.is_some() {
            return Err(Error::Unsupported(
                "bulk geometry cannot be reinterpreted by legacy view".into(),
            ));
        }
        // A view of an already strided view must first use explicit packing.
        if !input.view.is_contiguous() {
            return Err(Error::State(
                "nested strided host view requires packing".into(),
            ));
        }
        view.validate(input.elements)?;
        let elements = view.elements()?;
        let offset = input
            .view
            .offset
            .checked_add(view.offset)
            .ok_or_else(|| Error::Limit("host tensor offset overflow".into()))?;
        Ok(HostValue {
            data: input.data.clone(),
            view: TensorView { offset, ..view },
            elements,
            precision: input.precision,
            source: input.source,
            profile: input.profile,
        })
    }
    fn contiguous(&mut self, input: &HostValue) -> Result<HostValue> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if input.data.bulk.is_some() && input.view != TensorView::vector(0, input.elements) {
            return Err(Error::Unsupported(
                "bulk geometry cannot be implicitly packed".into(),
            ));
        }
        if input.view.is_contiguous() {
            return Ok(input.clone());
        }
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let workspace = self.prepare_output(Some(input), input.elements, 1)?;
        let mut values = Vec::with_capacity(input.elements);
        for row in 0..input.view.rows {
            let offset = input.view.offset + row * input.view.row_stride;
            values.extend_from_slice(&input.data[offset..offset + input.view.width]);
        }
        let output = match input.source {
            Some(source) => HostValue::typed(
                values,
                input.precision,
                source,
                input
                    .profile
                    .ok_or_else(|| Error::State("typed host profile missing".into()))?,
            ),
            None => Ok(HostValue::new(values)),
        }?;
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn rms(
        &mut self,
        name: &str,
        input: &HostValue,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.vector(gamma)?;
        if eps.to_bits() != self.engine()?.eps.to_bits() {
            return Err(Error::State(
                "host residual RMS changed source epsilon".into(),
            ));
        }
        Ok(HostValue::new(self.engine()?.normalize_region(
            name,
            input.as_slice()?,
            gamma.values,
        )?))
    }
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &HostValue,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.vector(gamma)?;
        Ok(HostValue::new(self.engine()?.normalize_grouped_region(
            name,
            input.as_slice()?,
            gamma.values,
            groups,
            eps,
        )?))
    }
    fn silu(
        &mut self,
        name: &str,
        input: &HostValue,
        multiplier: Option<&HostValue>,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        if let Some(value) = multiplier {
            self.legacy_precision(&[value])?;
        }
        let multiplier = multiplier.map(HostValue::as_slice).transpose()?;
        Ok(HostValue::new(self.engine()?.silu_region(
            name,
            input.as_slice()?,
            multiplier,
        )?))
    }
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    fn torch_cpu_bf16_silu_mul(
        &mut self,
        _name: &str,
        gate: &HostValue,
        up: &HostValue,
    ) -> Result<HostValue> {
        self.execute_torch_pointwise(gate, up, true)
    }
    #[cfg(all(feature = "source-binding", target_os = "linux"))]
    fn torch_cpu_bf16_add(
        &mut self,
        _name: &str,
        left: &HostValue,
        right: &HostValue,
    ) -> Result<HostValue> {
        self.execute_torch_pointwise(left, right, false)
    }
    fn torch_cpu_bf16_attention(
        &mut self,
        _name: &str,
        input: crate::accelerator::program::TorchCpuAttentionInputs<'_, HostValue>,
    ) -> Result<crate::accelerator::program::TorchCpuAttentionValues<HostValue>> {
        #[cfg(all(feature = "source-binding", target_os = "linux"))]
        {
            self.execute_cpu_attention(input)
        }
        #[cfg(not(all(feature = "source-binding", target_os = "linux")))]
        {
            let _ = input;
            Err(Error::Unsupported(
                "eager CPU attention source-window feature".into(),
            ))
        }
    }
    fn add(&mut self, name: &str, left: &HostValue, right: &HostValue) -> Result<HostValue> {
        self.legacy_precision(&[left, right])?;
        Ok(HostValue::new(self.engine()?.binary_region(
            BinaryRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                kind: BinaryOperation::Add,
                left: left.as_slice()?,
                right: right.as_slice()?,
            },
        )?))
    }
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &HostValue,
        gate: &HostValue,
    ) -> Result<HostValue> {
        self.legacy_precision(&[value, gate])?;
        Ok(HostValue::new(self.engine()?.binary_region(
            BinaryRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                kind: BinaryOperation::SigmoidGate,
                left: value.as_slice()?,
                right: gate.as_slice()?,
            },
        )?))
    }
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &HostValue,
        weights: SourceVector<'_>,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.vector(weights)?;
        let engine = self.engine()?;
        let Some(LayerState::Delta { conv, .. }) = self.state_mut()?.layers.get_mut(domain.0)
        else {
            return Err(Error::State("invalid private convolution domain".into()));
        };
        let input = input.as_slice()?;
        let mut output = vec![0.0; input.len()];
        engine.gdn_conv_region(
            name,
            input,
            weights.values,
            Arc::make_mut(conv).as_mut_slice(),
            &mut output,
        )?;
        Ok(HostValue::new(output))
    }
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &HostValue,
        beta: &HostValue,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<HostValue>> {
        self.legacy_precision(&[alpha, beta])?;
        self.vector(dt)?;
        self.vector(a)?;
        let ready = self
            .engine()?
            .gdn_coefficient_region(GdnCoefficientRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                alpha: alpha.as_slice()?,
                beta: beta.as_slice()?,
                dt: dt.values,
                a: a.values,
                scale,
            })?;
        let packed = ready
            .coefficients
            .iter()
            .flat_map(|c| [c.gain, c.beta, c.scale])
            .collect();
        Ok(CoefficientValues {
            packed: HostValue::new(packed),
            softplus: HostValue::new(ready.softplus),
            gate_raw: HostValue::new(ready.gate_raw),
        })
    }
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &HostValue,
        k: &HostValue,
        v: &HostValue,
        coefficients: &HostValue,
        width: usize,
    ) -> Result<HostValue> {
        self.legacy_precision(&[q, k, v, coefficients])?;
        let q = q.as_slice()?;
        let k = k.as_slice()?;
        let v = v.as_slice()?;
        let packed = coefficients.as_slice()?;
        if width == 0
            || !q.len().is_multiple_of(width)
            || (q.len() / width).checked_mul(3) != Some(packed.len())
        {
            return Err(Error::Malformed("host recurrent coefficient layout".into()));
        }
        let coefficients: Vec<_> = packed
            .chunks_exact(3)
            .map(|c| GdnCoefficients {
                gain: c[0],
                beta: c[1],
                scale: c[2],
            })
            .collect();
        let engine = self.engine()?;
        let source_identity = self.source.source_identity();
        let Some(LayerState::Delta { state, .. }) = self.state_mut()?.layers.get_mut(domain.0)
        else {
            return Err(Error::State("invalid private recurrent domain".into()));
        };
        let recurrent = Arc::make_mut(state).as_mut_slice();
        let result = engine.gdn_recurrent_region(GdnRecurrentRegion {
            source_identity,
            operation: name,
            width,
            q,
            k,
            v,
            coefficients: &coefficients,
            state: recurrent,
        })?;
        let mut output = vec![0.0; q.len()];
        match result {
            GdnRecurrentExecution::Completed(ready) => {
                output.copy_from_slice(&ready.output);
                recurrent.copy_from_slice(&ready.state);
            }
            GdnRecurrentExecution::Unsupported => {
                for (h, &coeff) in coefficients.iter().enumerate() {
                    source_gdn_recurrent_step(
                        &mut recurrent[h * width * width..(h + 1) * width * width],
                        &q[h * width..(h + 1) * width],
                        &k[h * width..(h + 1) * width],
                        &v[h * width..(h + 1) * width],
                        coeff,
                        &mut output[h * width..(h + 1) * width],
                    )?;
                }
            }
        }
        Ok(HostValue::new(output))
    }
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: usize) -> Result<()> {
        self.legacy_precision(&[])?;
        self.kv(domain, position)?;
        Ok(())
    }
    fn rope(
        &mut self,
        name: &str,
        q: &HostValue,
        k: &HostValue,
        v: &HostValue,
        position: usize,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<HostValue>> {
        self.legacy_precision(&[q, k, v])?;
        let ready = self.engine()?.rope_kv_region(RopeKvRegion {
            source_identity: self.source.source_identity(),
            operation: name,
            query: q.as_slice()?,
            key: k.as_slice()?,
            value: v.as_slice()?,
            head_width: geometry.attention.width,
            query_heads: geometry.attention.query_heads,
            kv_heads: geometry.attention.kv_heads,
            rotate: geometry.rotate,
            position,
            theta_scale: geometry.theta_scale,
        })?;
        Ok(RopeValues {
            query: HostValue::new(ready.query),
            key: HostValue::new(ready.key),
            value: HostValue::new(ready.value),
        })
    }
    fn append_kv(&mut self, domain: KvDomain, k: &HostValue, v: &HostValue) -> Result<()> {
        self.legacy_precision(&[k, v])?;
        let Some(LayerState::Full { keys, values }) = self.state_mut()?.layers.get_mut(domain.0)
        else {
            return Err(Error::State("invalid private KV domain".into()));
        };
        keys.push(Arc::from(k.as_slice()?));
        values.push(Arc::from(v.as_slice()?));
        Ok(())
    }
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &HostValue,
        valid_tokens: usize,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<HostValue>> {
        self.legacy_precision(&[query])?;
        let (keys, _) = self.kv(domain, valid_tokens)?;
        let ready = self.engine()?.attention_qk_region(AttentionQkRegion {
            source_identity: self.source.source_identity(),
            operation: name,
            query: query.as_slice()?,
            keys,
            head_width: geometry.width,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            scale,
        })?;
        Ok(QkValues {
            dots: HostValue::new(ready.dots),
            scores: HostValue::new(ready.scores),
        })
    }
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &HostValue,
        heads: usize,
        valid_tokens: usize,
    ) -> Result<HostValue> {
        self.legacy_precision(&[scores])?;
        let ready = self
            .engine()?
            .attention_softmax_region(AttentionSoftmaxRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                scores: scores.as_slice()?,
                heads,
                width: valid_tokens,
            })?;
        Ok(HostValue::new(ready.probabilities))
    }
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &HostValue,
        valid_tokens: usize,
        geometry: AttentionGeometry,
    ) -> Result<HostValue> {
        self.legacy_precision(&[probabilities])?;
        let (_, values) = self.kv(domain, valid_tokens)?;
        Ok(HostValue::new(self.engine()?.attention_value_region(
            AttentionValueRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                probabilities: probabilities.as_slice()?,
                values,
                head_width: geometry.width,
                query_heads: geometry.query_heads,
                kv_heads: geometry.kv_heads,
                padded_tokens: valid_tokens.div_ceil(256) * 256,
            },
        )?))
    }
    fn assert_finite(&mut self, layer: usize, value: &HostValue) -> Result<()> {
        if !value.as_slice()?.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed(format!(
                "nonfinite layer {layer} computation"
            )));
        }
        Ok(())
    }
    fn observe(&mut self, layer: usize, stage: &'static str, value: &HostValue) -> Result<()> {
        if self.trace.is_some() {
            observe(
                &mut self.trace,
                self.position,
                layer,
                stage,
                value.as_slice()?,
            );
        }
        Ok(())
    }
    fn observe_state(
        &mut self,
        layer: usize,
        stage: &'static str,
        domain: StateSlot,
    ) -> Result<()> {
        if self.trace.is_some() {
            let state = self
                .state
                .as_deref()
                .ok_or_else(|| Error::State("operator slice has no observed state".into()))?;
            let values = match domain {
                StateSlot::Conv(ConvDomain(i)) => match state.layers.get(i) {
                    Some(LayerState::Delta { conv, .. }) => conv.as_slice(),
                    _ => return Err(Error::State("invalid observed convolution domain".into())),
                },
                StateSlot::Recurrent(RecurrentDomain(i)) => match state.layers.get(i) {
                    Some(LayerState::Delta { state, .. }) => state.as_slice(),
                    _ => return Err(Error::State("invalid observed recurrent domain".into())),
                },
                StateSlot::Kv(_) => {
                    return Err(Error::Unsupported(
                        "KV state observation needs explicit slab export".into(),
                    ));
                }
            };
            observe(&mut self.trace, self.position, layer, stage, values);
        }
        Ok(())
    }
    fn advance_frontier(&mut self) -> Result<()> {
        if self.state()?.position != self.position {
            return Err(Error::State(
                "duplicate host proposed frontier advance".into(),
            ));
        }
        let next = self
            .state()?
            .position
            .checked_add(1)
            .ok_or_else(|| Error::Limit("host proposed position overflow".into()))?;
        self.state_mut()?.position = next;
        Ok(())
    }
}

fn round_bf16(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(Error::Malformed("nonfinite typed projection output".into()));
    }
    let bits = value.to_bits();
    let rounded = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err(Error::Limit("typed BF16 projection output overflow".into()));
    }
    Ok(rounded)
}

#[cfg(test)]
mod typed_value_tests {
    use super::*;
    #[test]
    fn typed_values_require_explicit_source_precision_and_keep_profile() {
        let source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
        let profile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
        assert!(
            HostValue::typed(
                vec![1.0],
                ActivationPrecision::Bf16WidenedF32,
                crate::accelerator::SourceIdentity::unbound(),
                profile
            )
            .is_err()
        );
        assert!(
            HostValue::typed(
                vec![1.0001],
                ActivationPrecision::Bf16WidenedF32,
                source,
                profile
            )
            .is_err()
        );
        let typed = HostValue::typed(
            vec![-0.0, 1.0],
            ActivationPrecision::Bf16WidenedF32,
            source,
            profile,
        )
        .unwrap();
        assert_eq!(typed.source, Some(source));
        assert_eq!(typed.profile, Some(profile));
        assert_eq!(typed.as_slice().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        let legacy = HostValue::new(vec![1.0]);
        assert_eq!(legacy.precision, ActivationPrecision::F32);
        assert_eq!(legacy.source, None);
        assert_eq!(legacy.profile, None);
    }
    #[test]
    fn projection_boundary_rounds_bf16_rne_and_refuses_finite_overflow() {
        for (input, expected) in [
            (0x3f80_8000, 0x3f80_0000),
            (0x3f81_8000, 0x3f82_0000),
            (0x8000_0000, 0x8000_0000),
        ] {
            assert_eq!(
                round_bf16(f32::from_bits(input)).unwrap().to_bits(),
                expected
            );
        }
        for value in [f32::MAX, f32::NAN, f32::INFINITY] {
            assert!(round_bf16(value).is_err());
        }
    }
}

#[cfg(all(feature = "source-binding", target_os = "linux"))]
mod full_window;
#[cfg(all(feature = "source-binding", target_os = "linux"))]
mod gdn_conv;

#[cfg(all(feature = "source-binding", target_os = "linux"))]
mod attention_tail;
#[cfg(all(feature = "source-binding", target_os = "linux"))]
mod output_tail;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod bulk_f32;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod zen_ssm_out;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod gguf_prefix;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod gdn_upstream;
