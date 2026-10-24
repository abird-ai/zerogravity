//! Closed GGUF operators under the same HostMath; no session construction.
use super::bulk_f32::BulkBacking;
use super::*;
use crate::accelerator::program::GgufPrefixDescriptor;
use crate::operator_slice::bulk_f32::{GgufPrefixOwner, GgufPrefixSlice, not_cancelled};
use std::sync::atomic::AtomicBool;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};
use zerogravity_cpu::math::haswell_bulk_f32::verify_environment;

const WORDS: usize = 2048 * 64;
pub(super) struct PrefixEmbedding {
    values: Vec<f32>,
    _output: GlobalResourceLease,
    owner: Arc<GgufPrefixOwner>,
}
pub(super) struct PrefixRms {
    unweighted: Vec<f32>,
    normalized: Vec<f32>,
    _output: GlobalResourceLease,
    _embedding: HostValue,
    owner: Arc<GgufPrefixOwner>,
}
impl PrefixEmbedding {
    pub(super) fn values(&self) -> &[f32] {
        &self.values
    }
}
impl PrefixRms {
    pub(super) fn values(&self) -> &[f32] {
        &self.normalized
    }
}
pub(super) struct GgufPrefixCall<'a> {
    slice: &'a GgufPrefixSlice,
    cancelled: Option<&'a AtomicBool>,
}
fn reserve(words: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(words)
        .map_err(|_| Error::Limit("GGUF prefix output allocation".into()))?;
    if values.capacity() != words {
        return Err(Error::Limit("GGUF prefix exact output capacity".into()));
    }
    Ok(values)
}
fn row(values: Vec<f32>) -> Result<Vec<f32>> {
    if values.len() != 2048 || values.capacity() != 2048 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "GGUF prefix source row geometry/finite/capacity".into(),
        ));
    }
    Ok(values)
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub(crate) fn for_gguf_prefix_slice(
        slice: &'engine GgufPrefixSlice,
        cancelled: Option<&'engine AtomicBool>,
    ) -> Self {
        let mut math = Self::for_operator_slice(slice.owner.model.as_ref(), None);
        math.gguf_prefix_call = Some(GgufPrefixCall { slice, cancelled });
        math
    }
    fn prefix_call(&self, descriptor: &GgufPrefixDescriptor) -> Result<&GgufPrefixCall<'engine>> {
        let call = self
            .gguf_prefix_call
            .as_ref()
            .ok_or_else(|| Error::Unsupported("GGUF prefix has no original owner".into()))?;
        if self.engine.is_some()
            || self.state.is_some()
            || self.region_kernel.is_some()
            || self.bulk_call.is_some()
            || self.gguf_gdn_upstream_call.is_some()
            || self.zen_ssm_out_call.is_some()
            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
            || descriptor != &call.slice.owner.descriptor
            || descriptor.source() != self.source.source_identity()
        {
            return Err(Error::Unsupported(
                "GGUF prefix private same-engine source/issuer scope".into(),
            ));
        }
        not_cancelled(call.cancelled)?;
        verify_environment()?;
        Ok(call)
    }
    pub(super) fn execute_gguf_prefix_embedding(
        &self,
        descriptor: &GgufPrefixDescriptor,
    ) -> Result<HostValue> {
        let call = self.prefix_call(descriptor)?;
        let owner = &call.slice.owner;
        descriptor
            .embedding()
            .validate_source(owner.model.as_ref())?;
        let output = owner.admission.arbiter.admit(
            &owner.admission.resource,
            owner.admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: (WORDS * 4) as u64,
            }],
        )?;
        // One selected decoded row plus codec stack/headers; no whole embedding cache.
        let scratch = owner.admission.arbiter.admit(
            &owner.admission.resource,
            owner.admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 8192 + 1024,
            }],
        )?;
        let mut values = reserve(WORDS)?;
        for &id in descriptor.span().ids() {
            not_cancelled(call.cancelled)?;
            verify_environment()?;
            let decoded = row(zerogravity_cpu::weights::source_reference_embedding(
                owner.model.embedding_source(id)?,
            )?)?;
            values.extend_from_slice(&decoded);
        }
        verify_environment()?;
        not_cancelled(call.cancelled)?;
        drop(scratch);
        HostValue::from_bulk(
            BulkBacking::GgufEmbedding {
                completed: Box::new(PrefixEmbedding {
                    values,
                    _output: output,
                    owner: owner.clone(),
                }),
            },
            WORDS,
            descriptor.source(),
        )
    }
    pub(super) fn execute_gguf_prefix_rms(
        &self,
        descriptor: &GgufPrefixDescriptor,
        input: &HostValue,
    ) -> Result<HostValue> {
        let call = self.prefix_call(descriptor)?;
        let owner = &call.slice.owner;
        if input.source != Some(descriptor.source())
            || input.profile != Some(ProgramNumericalProfile::GgufSourceV1)
            || input.precision != ActivationPrecision::F32
            || input.elements != WORDS
            || input.view != TensorView::vector(0, WORDS)
            || !matches!(input.data.bulk.as_ref(), Some(BulkBacking::GgufEmbedding { completed }) if Arc::ptr_eq(&completed.owner, owner))
        {
            return Err(Error::State(
                "GGUF prefix foreign/observed/partial embedding".into(),
            ));
        }
        descriptor.norm().validate_source(owner.model.as_ref())?;
        let output = owner.admission.arbiter.admit(
            &owner.admission.resource,
            owner.admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: (2 * WORDS * 4) as u64,
            }],
        )?;
        // Two helper row results + stack identity gamma; all before allocation.
        let scratch = owner.admission.arbiter.admit(
            &owner.admission.resource,
            owner.admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 3 * 8192 + 1024,
            }],
        )?;
        let mut unweighted = reserve(WORDS)?;
        let mut normalized = reserve(WORDS)?;
        let identity = [1.0_f32; 2048];
        let gamma = &owner.model.layers[0].attn_norm;
        let raw_gamma = owner.model.tensor_bytes(&descriptor.norm().name)?;
        if raw_gamma.chunks_exact(4).zip(gamma).any(|(bytes, value)| {
            u32::from_le_bytes(bytes.try_into().expect("four-byte source chunk")) != value.to_bits()
        }) {
            return Err(Error::State(
                "GGUF prefix cached gamma differs from original bytes".into(),
            ));
        }
        for values in input.as_slice()?.chunks_exact(2048) {
            not_cancelled(call.cancelled)?;
            verify_environment()?;
            let plain = row(crate::source_rms_normalize(
                values,
                &identity,
                descriptor.epsilon(),
            )?)?;
            // GGUF converter already stored effective (1+HF gamma); no +1 here.
            let weighted = row(crate::source_rms_normalize(
                values,
                gamma,
                descriptor.epsilon(),
            )?)?;
            unweighted.extend_from_slice(&plain);
            normalized.extend_from_slice(&weighted);
        }
        verify_environment()?;
        not_cancelled(call.cancelled)?;
        drop(scratch);
        HostValue::from_bulk(
            BulkBacking::GgufRms {
                completed: Box::new(PrefixRms {
                    unweighted,
                    normalized,
                    _output: output,
                    _embedding: input.clone(),
                    owner: owner.clone(),
                }),
            },
            WORDS,
            descriptor.source(),
        )
    }
}
impl HostValue {
    pub(crate) fn gguf_prefix_unweighted(&self, owner: &Arc<GgufPrefixOwner>) -> Result<&[f32]> {
        match self.data.bulk.as_ref() {
            Some(BulkBacking::GgufRms { completed }) if Arc::ptr_eq(&completed.owner, owner) => {
                Ok(&completed.unweighted)
            }
            _ => Err(Error::State(
                "GGUF prefix retained RMS output backing".into(),
            )),
        }
    }
}
