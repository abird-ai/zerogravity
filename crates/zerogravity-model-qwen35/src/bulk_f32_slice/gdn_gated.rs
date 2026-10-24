//! Existing GGUF recurrence → source grouped RMS/raw gamma/SiLU gate.
//! Observed Z or exact retained upstream; no model execution permit.
use super::gdn_recurrent::BulkGdnRecurrentOutput;
use super::*;
use crate::accelerator::SiluRegion;

pub struct ObservedBulkGdnGate<'a> {
    pub span: TeacherSpan,
    /// Original dense `[2048,teacher,1,1]` projected Z, not SiLU(Z).
    pub z: &'a [f32],
}
struct Inputs {
    z: Vec<f32>,
    _staging: Option<GlobalResourceLease>,
    _metadata: GlobalResourceLease,
    recurrent: BulkGdnRecurrentOutput,
    model: Arc<LoadedQwenModel>,
    upstream: Option<GgufGdnUpstreamOutput>,
}
pub struct BulkGdnGatedSlice {
    inputs: Arc<Inputs>,
}
impl BulkGdnGatedSlice {
    pub fn new(
        recurrent: BulkGdnRecurrentOutput,
        observed: ObservedBulkGdnGate<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        let coefficients = recurrent.coefficients();
        let model = coefficients.model().clone();
        let gamma = model.tensor_info("blk.0.ssm_norm.weight")?;
        if observed.span != *coefficients.span()
            || model.source_identity() != coefficients.source_identity()
            || model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || gamma.kind != zerogravity_core::quant::QuantKind::F32
            || gamma.shape != [128]
            || model.layers[0].ssm_norm.as_ref().map(Vec::len) != Some(128)
            || observed.z.len() != recurrent.core().len()
            || observed.z.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Malformed("bulk GDN gate source/issuer/shape".into()));
        }
        let admission = coefficients.admission();
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let staging = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: std::mem::size_of_val(observed.z) as u64,
            }],
        )?;
        let mut z = Vec::new();
        z.try_reserve_exact(observed.z.len())
            .map_err(|_| Error::Limit("bulk GDN gate allocation".into()))?;
        if z.capacity() != observed.z.len() {
            return Err(Error::Limit("bulk GDN gate capacity".into()));
        }
        z.extend_from_slice(observed.z);
        not_cancelled(cancelled)?;
        Ok(Self {
            inputs: Arc::new(Inputs {
                z,
                _staging: Some(staging),
                _metadata: metadata,
                recurrent,
                model,
                upstream: None,
            }),
        })
    }

    /// The recurrent consumer must retain this same actual QKV/Z producer.
    pub fn from_upstream(
        recurrent: BulkGdnRecurrentOutput,
        upstream: GgufGdnUpstreamOutput,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        let coefficients = recurrent.coefficients();
        if recurrent
            .upstream()
            .is_none_or(|actual| !actual.same_producer(&upstream))
            || coefficients.span() != upstream.descriptor().span()
            || upstream.z()?.len() != recurrent.core().len()
        {
            return Err(Error::State(
                "bulk GDN gate exact actual upstream producer".into(),
            ));
        }
        let model = coefficients.model().clone();
        let gamma = model.tensor_info("blk.0.ssm_norm.weight")?;
        if gamma.kind != zerogravity_core::quant::QuantKind::F32
            || gamma.shape != [128]
            || model.layers[0].ssm_norm.as_ref().map(Vec::len) != Some(128)
        {
            return Err(Error::Unsupported(
                "bulk GDN original gated norm source".into(),
            ));
        }
        let admission = coefficients.admission();
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        not_cancelled(cancelled)?;
        Ok(Self {
            inputs: Arc::new(Inputs {
                z: Vec::new(),
                _staging: None,
                _metadata: metadata,
                recurrent,
                model,
                upstream: Some(upstream),
            }),
        })
    }

    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkGdnGatedOutput> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        let input = &self.inputs;
        let coefficients = input.recurrent.coefficients();
        let admission = coefficients.admission();
        let words = input.recurrent.core().len();
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: (3 * words * std::mem::size_of::<f32>()) as u64,
                },
                // One returned source RMS head plus the identity-gamma stack.
                AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 1024,
                },
            ],
        )?;
        let gamma = input.model.layers[0].ssm_norm.as_deref().unwrap();
        let groups = coefficients.span().len() * 16;
        // Observe the existing unweighted boundary using exact identity weights;
        // the live weighted path still calls the original helper with raw gamma.
        let normalized = crate::source_grouped_rms_normalize(
            input.recurrent.core(),
            &[1.0; 128],
            groups,
            input.model.eps,
        )?;
        not_cancelled(cancelled)?;
        let weighted = crate::source_grouped_rms_normalize(
            input.recurrent.core(),
            gamma,
            groups,
            input.model.eps,
        )?;
        not_cancelled(cancelled)?;
        let gated = crate::source_silu(SiluRegion {
            source_identity: coefficients.source_identity(),
            operation: "blk.0.gdn_output_gate",
            input: match &input.upstream {
                Some(upstream) => upstream.z()?,
                None => &input.z,
            },
            multiplier: Some(&weighted),
        })?;
        if gated.iter().any(|value| !value.is_finite()) {
            return Err(Error::Malformed("nonfinite bulk GDN gated result".into()));
        }
        if normalized.capacity() != words
            || weighted.capacity() != words
            || gated.capacity() != words
        {
            return Err(Error::Limit("bulk GDN gated result capacity".into()));
        }
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        not_cancelled(cancelled)?;
        Ok(BulkGdnGatedOutput {
            completed: Arc::new(Completed {
                normalized,
                weighted,
                gated,
                _lease: lease,
                inputs: input.clone(),
            }),
        })
    }
}
struct Completed {
    normalized: Vec<f32>,
    weighted: Vec<f32>,
    gated: Vec<f32>,
    _lease: GlobalResourceLease,
    inputs: Arc<Inputs>,
}
/// Borrowed original-layout outputs with every producer/source lease retained.
#[derive(Clone)]
pub struct BulkGdnGatedOutput {
    completed: Arc<Completed>,
}
impl BulkGdnGatedOutput {
    pub fn recurrent(&self) -> &BulkGdnRecurrentOutput {
        &self.completed.inputs.recurrent
    }
    pub fn normalized(&self) -> &[f32] {
        &self.completed.normalized
    }
    pub fn weighted(&self) -> &[f32] {
        &self.completed.weighted
    }
    pub fn gated(&self) -> &[f32] {
        &self.completed.gated
    }
}
