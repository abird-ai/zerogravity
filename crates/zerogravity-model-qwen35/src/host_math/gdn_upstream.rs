//! Source arithmetic under HostMath; all intermediate owners remain private.
use super::bulk_f32::BulkBacking;
use super::*;
use crate::accelerator::{MatVecRegion, SiluRegion};
use crate::operator_slice::bulk_f32::source_bindings::{GdnProjection, GdnUpstreamBinding};
use crate::operator_slice::bulk_f32::{GgufGdnUpstreamOwner, GgufGdnUpstreamSlice, not_cancelled};
use std::sync::atomic::AtomicBool;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};
use zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment;

const CHANNELS: usize = 6144;
const TEACHERS: usize = 64;
const WORDS: usize = CHANNELS * TEACHERS;
pub(super) struct UpstreamProjection {
    values: Vec<f32>,
    _output: GlobalResourceLease,
    _input: HostValue,
    owner: Arc<GgufGdnUpstreamOwner>,
    projection: GdnProjection,
}
pub(super) struct UpstreamConvolution {
    raw: Vec<f32>,
    activated: Vec<f32>,
    history: Vec<f32>,
    _output: GlobalResourceLease,
    _qkv: HostValue,
    owner: Arc<GgufGdnUpstreamOwner>,
}
pub(super) struct UpstreamQk {
    values: Vec<f32>,
    _output: GlobalResourceLease,
    _convolution: HostValue,
    _owner: Arc<GgufGdnUpstreamOwner>,
}
impl UpstreamProjection {
    pub(super) fn values(&self) -> &[f32] {
        &self.values
    }
}
impl UpstreamConvolution {
    pub(super) fn values(&self) -> &[f32] {
        &self.activated
    }
}
impl UpstreamQk {
    pub(super) fn values(&self) -> &[f32] {
        &self.values
    }
}
pub(super) struct UpstreamCall<'a> {
    slice: &'a GgufGdnUpstreamSlice,
    cancelled: Option<&'a AtomicBool>,
}
fn reserve(count: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("GGUF upstream allocation".into()))?;
    if values.capacity() != count {
        return Err(Error::Limit("GGUF upstream actual capacity".into()));
    }
    Ok(values)
}
fn lease(
    owner: &GgufGdnUpstreamOwner,
    kind: AllocationClass,
    bytes: usize,
) -> Result<GlobalResourceLease> {
    let admission = owner.prefix.admission();
    admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind,
            bytes: bytes as u64,
        }],
    )
}
fn full(value: &HostValue, owner: &GgufGdnUpstreamOwner, words: usize) -> bool {
    value.source == Some(owner.descriptor.source())
        && value.profile == Some(ProgramNumericalProfile::GgufSourceV1)
        && value.precision == ActivationPrecision::F32
        && value.elements == words
        && value.view == TensorView::vector(0, words)
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub(crate) fn for_gguf_gdn_upstream_slice(
        slice: &'engine GgufGdnUpstreamSlice,
        cancelled: Option<&'engine AtomicBool>,
    ) -> Self {
        let mut math = Self::for_operator_slice(slice.owner.prefix.model().as_ref(), None);
        math.gguf_gdn_upstream_call = Some(UpstreamCall { slice, cancelled });
        math
    }
    fn upstream_call(&self, descriptor: &GdnUpstreamBinding) -> Result<&UpstreamCall<'engine>> {
        let call = self
            .gguf_gdn_upstream_call
            .as_ref()
            .ok_or_else(|| Error::Unsupported("GGUF upstream has no retained prefix".into()))?;
        if self.engine.is_some()
            || self.state.is_some()
            || self.region_kernel.is_some()
            || self.bulk_call.is_some()
            || self.zen_ssm_out_call.is_some()
            || self.gguf_prefix_call.is_some()
            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
            || descriptor != &call.slice.owner.descriptor
            || descriptor.source() != self.source.source_identity()
        {
            return Err(Error::Unsupported(
                "GGUF upstream private same-engine scope".into(),
            ));
        }
        not_cancelled(call.cancelled)?;
        verify_environment()?;
        Ok(call)
    }
    pub(super) fn import_gguf_gdn_input(
        &self,
        descriptor: &GdnUpstreamBinding,
    ) -> Result<HostValue> {
        let call = self.upstream_call(descriptor)?;
        HostValue::from_bulk(
            BulkBacking::GgufGdnInput {
                owner: call.slice.owner.clone(),
            },
            131072,
            descriptor.source(),
        )
    }
    pub(super) fn execute_gguf_gdn_projection(
        &self,
        descriptor: &GdnUpstreamBinding,
        projection: GdnProjection,
        input: &HostValue,
    ) -> Result<HostValue> {
        let call = self.upstream_call(descriptor)?;
        let owner = &call.slice.owner;
        if !full(input, owner, 131072)
            || !matches!(input.data.bulk.as_ref(), Some(BulkBacking::GgufGdnInput { owner: actual }) if Arc::ptr_eq(actual, owner))
        {
            return Err(Error::State(
                "GGUF upstream exact normalized backing".into(),
            ));
        }
        let tensor = descriptor.tensor(projection);
        let model = owner.prefix.model();
        tensor.validate_source(model.as_ref())?;
        let weights = model.tensor_bytes(&tensor.name)?;
        let output = lease(owner, AllocationClass::Output, tensor.rows * TEACHERS * 4)?;
        let scratch = lease(owner, AllocationClass::Scratch, tensor.rows * 4 + 4096)?;
        let mut values = reserve(tensor.rows * TEACHERS)?;
        for activation in input.as_slice()?.chunks_exact(2048) {
            not_cancelled(call.cancelled)?;
            verify_environment()?;
            let row = zerogravity_cpu::source::source_reference_projection(MatVecRegion {
                source_identity: descriptor.source(),
                tensor_name: &tensor.name,
                kind: tensor.kind,
                columns: tensor.columns,
                rows: tensor.rows,
                packed_weights: weights,
                activation,
            })?;
            if row.capacity() != tensor.rows {
                return Err(Error::Limit(
                    "GGUF upstream projection scratch capacity".into(),
                ));
            }
            values.extend_from_slice(&row);
        }
        verify_environment()?;
        not_cancelled(call.cancelled)?;
        drop(scratch);
        HostValue::from_bulk(
            BulkBacking::GgufGdnProjection {
                completed: Box::new(UpstreamProjection {
                    values,
                    _output: output,
                    _input: input.clone(),
                    owner: owner.clone(),
                    projection,
                }),
            },
            tensor.rows * TEACHERS,
            descriptor.source(),
        )
    }
    pub(super) fn execute_gguf_gdn_convolution(
        &self,
        descriptor: &GdnUpstreamBinding,
        input: &HostValue,
    ) -> Result<HostValue> {
        let call = self.upstream_call(descriptor)?;
        let owner = &call.slice.owner;
        if !full(input, owner, WORDS)
            || !matches!(input.data.bulk.as_ref(), Some(BulkBacking::GgufGdnProjection { completed }) if Arc::ptr_eq(&completed.owner, owner) && completed.projection == GdnProjection::Qkv)
        {
            return Err(Error::State("GGUF upstream exact QKV backing".into()));
        }
        let model = owner.prefix.model();
        descriptor.convolution().validate_source(model.as_ref())?;
        let weights = model.layers[0]
            .conv_kernel
            .as_deref()
            .ok_or_else(|| Error::State("GGUF upstream convolution weights".into()))?;
        let raw_weights = model.tensor_bytes(&descriptor.convolution().name)?;
        if raw_weights
            .chunks_exact(4)
            .zip(weights)
            .any(|(bytes, value)| {
                u32::from_le_bytes(bytes.try_into().expect("four-byte source chunk"))
                    != value.to_bits()
            })
        {
            return Err(Error::State(
                "GGUF upstream convolution differs from original source".into(),
            ));
        }
        let output = lease(
            owner,
            AllocationClass::Output,
            (2 * WORDS + 3 * CHANNELS) * 4,
        )?;
        let mut raw = reserve(WORDS)?;
        raw.resize(WORDS, 0.0);
        let mut history = reserve(3 * CHANNELS)?;
        history.resize(3 * CHANNELS, 0.0);
        for (teacher, activation) in input.as_slice()?.chunks_exact(CHANNELS).enumerate() {
            not_cancelled(call.cancelled)?;
            verify_environment()?;
            crate::source_gdn_conv_step(
                activation,
                weights,
                &mut history,
                &mut raw[teacher * CHANNELS..(teacher + 1) * CHANNELS],
            )?;
        }
        not_cancelled(call.cancelled)?;
        let activated = crate::source_silu(SiluRegion {
            source_identity: descriptor.source(),
            operation: "blk.0.gdn_conv_silu",
            input: &raw,
            multiplier: None,
        })?;
        if activated.capacity() != WORDS || activated.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed(
                "GGUF upstream convolution activation/capacity".into(),
            ));
        }
        verify_environment()?;
        not_cancelled(call.cancelled)?;
        HostValue::from_bulk(
            BulkBacking::GgufGdnConvolution {
                completed: Box::new(UpstreamConvolution {
                    raw,
                    activated,
                    history,
                    _output: output,
                    _qkv: input.clone(),
                    owner: owner.clone(),
                }),
            },
            WORDS,
            descriptor.source(),
        )
    }
    pub(super) fn execute_gguf_gdn_qk(
        &self,
        descriptor: &GdnUpstreamBinding,
        input: &HostValue,
    ) -> Result<HostValue> {
        let call = self.upstream_call(descriptor)?;
        let owner = &call.slice.owner;
        if !full(input, owner, WORDS)
            || !matches!(input.data.bulk.as_ref(), Some(BulkBacking::GgufGdnConvolution { completed }) if Arc::ptr_eq(&completed.owner, owner))
        {
            return Err(Error::State(
                "GGUF upstream exact convolution backing".into(),
            ));
        }
        let output = lease(owner, AllocationClass::Output, WORDS * 4)?;
        // One Q/K teacher result, one returned RMS head and stack gamma.
        let scratch = lease(owner, AllocationClass::Scratch, 4096 * 4 + 128 * 8 + 1024)?;
        let mut values = reserve(WORDS)?;
        values.resize(WORDS, 0.0);
        let scale = [1.0 / 128.0_f32.sqrt(); 128];
        for (teacher, activated) in input.as_slice()?.chunks_exact(CHANNELS).enumerate() {
            not_cancelled(call.cancelled)?;
            verify_environment()?;
            let qk = crate::source_grouped_rms_normalize(
                &activated[..4096],
                &scale,
                32,
                descriptor.prefix().epsilon() / 128.0,
            )?;
            if qk.capacity() != 4096 {
                return Err(Error::Limit("GGUF upstream QK scratch capacity".into()));
            }
            values[teacher * 2048..(teacher + 1) * 2048].copy_from_slice(&qk[..2048]);
            values[131072 + teacher * 2048..131072 + (teacher + 1) * 2048]
                .copy_from_slice(&qk[2048..]);
            values[262144 + teacher * 2048..262144 + (teacher + 1) * 2048]
                .copy_from_slice(&activated[4096..]);
        }
        verify_environment()?;
        not_cancelled(call.cancelled)?;
        drop(scratch);
        HostValue::from_bulk(
            BulkBacking::GgufGdnQk {
                completed: Box::new(UpstreamQk {
                    values,
                    _output: output,
                    _convolution: input.clone(),
                    _owner: owner.clone(),
                }),
            },
            WORDS,
            descriptor.source(),
        )
    }
}
impl HostValue {
    pub(crate) fn gguf_gdn_conv_raw(&self, owner: &Arc<GgufGdnUpstreamOwner>) -> Result<&[f32]> {
        match self.data.bulk.as_ref() {
            Some(BulkBacking::GgufGdnConvolution { completed })
                if Arc::ptr_eq(&completed.owner, owner) =>
            {
                Ok(&completed.raw)
            }
            _ => Err(Error::State("GGUF convolution raw producer".into())),
        }
    }
    pub(crate) fn gguf_gdn_conv_history(
        &self,
        owner: &Arc<GgufGdnUpstreamOwner>,
    ) -> Result<&[f32]> {
        match self.data.bulk.as_ref() {
            Some(BulkBacking::GgufGdnConvolution { completed })
                if Arc::ptr_eq(&completed.owner, owner) =>
            {
                Ok(&completed.history)
            }
            _ => Err(Error::State("GGUF convolution history producer".into())),
        }
    }
}

impl HostMath<'_, '_, '_> {
    pub(super) fn execute_source_f32(
        &self,
        actual: &crate::accelerator::program::SourceOperator,
        inputs: &[HostValue],
    ) -> Result<HostValue> {
        actual.validate()?;
        if inputs.len() != actual.input_elements()?.len() {
            return Err(Error::Malformed("source F32 host exact arity".into()));
        }
        if let Some(call) = &self.gguf_prefix_call {
            let d = &call.slice.owner.descriptor;
            if *actual == d.embedding_operator()? {
                return self.execute_gguf_prefix_embedding(d);
            }
            if *actual == d.rms_operator()? {
                return self.execute_gguf_prefix_rms(d, &inputs[0]);
            }
        }
        if let Some(call) = &self.gguf_gdn_upstream_call {
            let d = &call.slice.owner.descriptor;
            if *actual == d.input_operator()? {
                return self.import_gguf_gdn_input(d);
            }
            for projection in [GdnProjection::Qkv, GdnProjection::Z] {
                if *actual == d.projection_operator(projection)? {
                    return self.execute_gguf_gdn_projection(d, projection, &inputs[0]);
                }
            }
            if *actual == d.convolution_operator()? {
                return self.execute_gguf_gdn_convolution(d, &inputs[0]);
            }
            if *actual == d.qk_operator()? {
                return self.execute_gguf_gdn_qk(d, &inputs[0]);
            }
        }
        if let Some(call) = &self.zen_ssm_out_call {
            let d = &call.slice.owner.descriptor;
            if *actual == d.input_operator()? {
                return self.import_zen_ssm_out(d);
            }
            if *actual == d.projection_operator()? {
                return self.execute_zen_ssm_out(d, &inputs[0]);
            }
        }
        Err(Error::Unsupported(
            "source F32 lacks original retained family binding".into(),
        ))
    }
}
