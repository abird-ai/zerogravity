//! Connected bulk coefficient producer → existing GGUF AR recurrence region.
//! Explicit observations or actual retained upstream; never current roots.
use super::gdn_coefficients::{BulkGdnCoefficientContract, BulkGdnCoefficientOutput};
use super::*;

const HEADS: usize = 16;
const WIDTH: usize = 128;
const STEP: usize = HEADS * WIDTH;
const STATE: usize = HEADS * WIDTH * WIDTH;

/// Native dense `[key128,head16,teacher,1]` Q/K/V and
/// `[key128,value128,head16,1]` prior state. No hidden transpose/normalization.
pub struct ObservedBulkGdnInputs<'a> {
    pub span: TeacherSpan,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub previous_state: &'a [f32],
}

struct Inputs {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    previous_state: Vec<f32>,
    _staging: Option<GlobalResourceLease>,
    _metadata: GlobalResourceLease,
    coefficients: BulkGdnCoefficientOutput,
    upstream: Option<GgufGdnUpstreamOutput>,
}

pub struct BulkGdnRecurrentSlice {
    inputs: Arc<Inputs>,
}
impl BulkGdnRecurrentSlice {
    pub fn new(
        coefficients: BulkGdnCoefficientOutput,
        observed: ObservedBulkGdnInputs<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        let words = coefficients.span().len() * STEP;
        if coefficients.contract()
            != BulkGdnCoefficientContract::HaswellTinyBlasF32GgufCoefficientsV1
            || observed.span != *coefficients.span()
            || observed.query.len() != words
            || observed.key.len() != words
            || observed.value.len() != words
            || observed.previous_state.len() != STATE
            || observed
                .query
                .iter()
                .chain(observed.key)
                .chain(observed.value)
                .chain(observed.previous_state)
                .any(|v| !v.is_finite())
        {
            return Err(Error::Malformed(
                "bulk GDN observed issuer/layout/finite domain".into(),
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
        let bytes = (3 * words + STATE) * std::mem::size_of::<f32>();
        let staging = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: bytes as u64,
            }],
        )?;
        let copy = |values: &[f32]| -> Result<Vec<f32>> {
            not_cancelled(cancelled)?;
            let mut owned = Vec::new();
            owned
                .try_reserve_exact(values.len())
                .map_err(|_| Error::Limit("bulk GDN observed allocation".into()))?;
            if owned.capacity() != values.len() {
                return Err(Error::Limit("bulk GDN observed capacity".into()));
            }
            owned.extend_from_slice(values);
            Ok(owned)
        };
        let inputs = Inputs {
            query: copy(observed.query)?,
            key: copy(observed.key)?,
            value: copy(observed.value)?,
            previous_state: copy(observed.previous_state)?,
            _staging: Some(staging),
            _metadata: metadata,
            coefficients,
            upstream: None,
        };
        not_cancelled(cancelled)?;
        Ok(Self {
            inputs: Arc::new(inputs),
        })
    }

    /// Initial first64 state is private +0; every Q/K/V comes from the actual producer.
    pub fn from_upstream(
        coefficients: BulkGdnCoefficientOutput,
        upstream: GgufGdnUpstreamOutput,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        if coefficients.contract()
            != BulkGdnCoefficientContract::HaswellTinyBlasF32GgufCoefficientsV1
            || coefficients.span().base() != 0
            || coefficients.span().len() != 64
            || coefficients.span() != upstream.descriptor().span()
            || coefficients
                .prefix()
                .is_none_or(|prefix| !prefix.same_producer(upstream.prefix()))
            || upstream.query()?.len() != 131072
            || upstream.key()?.len() != 131072
            || upstream.value()?.len() != 131072
        {
            return Err(Error::State(
                "bulk GDN exact actual upstream/prefix/issuer".into(),
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
                query: Vec::new(),
                key: Vec::new(),
                value: Vec::new(),
                previous_state: Vec::new(),
                _staging: None,
                _metadata: metadata,
                coefficients,
                upstream: Some(upstream),
            }),
        })
    }

    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkGdnRecurrentOutput> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        let inputs = &self.inputs;
        let admission = inputs.coefficients.admission();
        let query = match &inputs.upstream {
            Some(upstream) => upstream.query()?,
            None => &inputs.query,
        };
        let key = match &inputs.upstream {
            Some(upstream) => upstream.key()?,
            None => &inputs.key,
        };
        let value = match &inputs.upstream {
            Some(upstream) => upstream.value()?,
            None => &inputs.value,
        };
        let core_words = query.len();
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: ((core_words + STATE) * std::mem::size_of::<f32>()) as u64,
            }],
        )?;
        let mut core = Vec::new();
        let mut state = Vec::new();
        core.try_reserve_exact(core_words)
            .map_err(|_| Error::Limit("bulk GDN core allocation".into()))?;
        state
            .try_reserve_exact(STATE)
            .map_err(|_| Error::Limit("bulk GDN state allocation".into()))?;
        if core.capacity() != core_words || state.capacity() != STATE {
            return Err(Error::Limit("bulk GDN result capacity".into()));
        }
        core.resize(core_words, 0.0);
        if inputs.upstream.is_some() {
            state.resize(STATE, 0.0);
        } else {
            state.extend_from_slice(&inputs.previous_state);
        }
        // This is one closed recurrence operator over original native axes.
        // Existing scalar math owns every decay/dot/FMA/output rounding point.
        for teacher in 0..inputs.coefficients.span().len() {
            let coefficients = &inputs.coefficients.teacher(teacher)?.coefficients;
            if coefficients.len() != HEADS {
                return Err(Error::State("bulk GDN coefficient head count".into()));
            }
            for (head, &coefficient) in coefficients.iter().enumerate() {
                not_cancelled(cancelled)?;
                let row = (teacher * HEADS + head) * WIDTH;
                crate::source_gdn_recurrent_step(
                    &mut state[head * WIDTH * WIDTH..(head + 1) * WIDTH * WIDTH],
                    &query[row..row + WIDTH],
                    &key[row..row + WIDTH],
                    &value[row..row + WIDTH],
                    coefficient,
                    &mut core[row..row + WIDTH],
                )?;
            }
        }
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        not_cancelled(cancelled)?;
        Ok(BulkGdnRecurrentOutput {
            completed: Arc::new(Completed {
                core,
                state,
                _lease: lease,
                inputs: inputs.clone(),
            }),
        })
    }
}

struct Completed {
    core: Vec<f32>,
    state: Vec<f32>,
    _lease: GlobalResourceLease,
    inputs: Arc<Inputs>,
}
/// Pure candidate buffers, retaining all original producers and observations.
#[derive(Clone)]
pub struct BulkGdnRecurrentOutput {
    completed: Arc<Completed>,
}
impl BulkGdnRecurrentOutput {
    pub(crate) fn upstream(&self) -> Option<&GgufGdnUpstreamOutput> {
        self.completed.inputs.upstream.as_ref()
    }
    pub fn coefficients(&self) -> &BulkGdnCoefficientOutput {
        &self.completed.inputs.coefficients
    }
    pub fn core(&self) -> &[f32] {
        &self.completed.core
    }
    pub fn state(&self) -> &[f32] {
        &self.completed.state
    }
}
