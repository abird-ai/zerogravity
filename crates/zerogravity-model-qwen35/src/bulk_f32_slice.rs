//! Pure explicit default-bulk F32 fragment, not a model producer/forward permit.
use crate::accelerator::program::{
    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
    PlanBuilder, ProgramNumericalProfile, RecordedOperatorSlice, TeacherSpan,
};
use crate::original_binding::{QwenMatrixRole, QwenNumericalProfile};
use crate::{Error, LoadedQwenModel, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

const META: u64 = 16384;

#[path = "bulk_f32_slice/gguf_prefix.rs"]
mod gguf_prefix;
pub(crate) use gguf_prefix::GgufPrefixOwner;
pub use gguf_prefix::{GgufPrefixOutput, GgufPrefixSlice};

#[path = "bulk_f32_slice/gdn_upstream.rs"]
mod gdn_upstream;
pub(crate) use gdn_upstream::GgufGdnUpstreamOwner;
pub use gdn_upstream::{GgufGdnUpstreamOutput, GgufGdnUpstreamSlice};

#[path = "bulk_f32_slice/gdn_coefficients.rs"]
mod gdn_coefficients;
pub use gdn_coefficients::{
    BulkGdnCoefficientContract, BulkGdnCoefficientOutput, BulkGdnCoefficientSlice,
};

#[path = "bulk_f32_slice/gdn_recurrent.rs"]
mod gdn_recurrent;
pub use gdn_recurrent::{BulkGdnRecurrentOutput, BulkGdnRecurrentSlice, ObservedBulkGdnInputs};

#[path = "bulk_f32_slice/gdn_gated.rs"]
mod gdn_gated;
pub use gdn_gated::{BulkGdnGatedOutput, BulkGdnGatedSlice, ObservedBulkGdnGate};
#[path = "bulk_f32_slice/gdn_ssm_out.rs"]
mod gdn_ssm_out;
pub(crate) use gdn_ssm_out::ZenSsmOutOwner;
pub use gdn_ssm_out::{BulkGdnSsmOutOutput, BulkGdnSsmOutSlice};
/// Existing host ledger only; no budget registration or outer RAM claim.
#[derive(Clone)]
pub struct BulkF32HostAdmission {
    pub arbiter: ResourceArbiter,
    pub resource: PhysicalDeviceKey,
    pub owner: ResourceOwner,
}
/// An explicit actual allocation view. External input is never claimed to be
/// current model state or a mathematical producer from the present Rust graph.
pub struct ExternalBulkF32<'a> {
    pub name: &'a str,
    pub layout: BulkF32Layout,
    pub backing: &'a [f32],
}

pub(crate) struct BulkOwner {
    pub(crate) model: Arc<LoadedQwenModel>,
    pub(crate) matrix: BulkF32Matrix,
    pub(crate) admission: BulkF32HostAdmission,
    _metadata: GlobalResourceLease,
}
pub(crate) enum LeftInput {
    External(Vec<f32>),
    Source,
}
pub(crate) enum RightInput {
    External(Vec<f32>),
    ProducedPrefix(GgufPrefixOutput),
}
pub(crate) struct BulkInputs {
    // Payloads precede staging; source/metadata owner remains last.
    left: LeftInput,
    right: RightInput,
    _staging: Option<GlobalResourceLease>,
    pub(crate) owner: Arc<BulkOwner>,
}
impl BulkInputs {
    pub(crate) fn values(&self, side: BulkF32Side) -> Result<&[f32]> {
        if side == BulkF32Side::Right {
            return match &self.right {
                RightInput::External(values) => Ok(values),
                RightInput::ProducedPrefix(prefix) => {
                    if !Arc::ptr_eq(prefix.model(), &self.owner.model)
                        || !matches!(self.owner.matrix.right().origin(), BulkF32Origin::RetainedNormalized { prefix: descriptor } if descriptor.as_ref() == prefix.descriptor())
                        || prefix.span() != self.owner.matrix.right().span()
                    {
                        return Err(Error::State(
                            "bulk exact retained normalized owner/issuer".into(),
                        ));
                    }
                    prefix.normalized()
                }
            };
        }
        match &self.left {
            LeftInput::External(values) => Ok(values),
            LeftInput::Source => {
                let BulkF32Origin::SourceMatrix(tensor) = self.owner.matrix.left().origin() else {
                    return Err(Error::State("bulk learned backing descriptor".into()));
                };
                tensor.validate_source(self.owner.model.as_ref())?;
                let bytes = self.owner.model.tensor_bytes(&tensor.name)?;
                // SAFETY: all F32 bit patterns are valid; source raw kind/length
                // and target little endian are checked. Returned borrow retains
                // the SAME immutable model allocation; no transposition/cache.
                let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
                if !cfg!(target_endian = "little")
                    || !prefix.is_empty()
                    || !suffix.is_empty()
                    || values.len() != self.owner.matrix.left().layout().backing_words()
                {
                    return Err(Error::Unsupported(
                        "bulk original F32 alignment/byte geometry".into(),
                    ));
                }
                Ok(values)
            }
        }
    }
}

pub struct BulkF32Slice {
    pub(crate) inputs: Arc<BulkInputs>,
}
impl BulkF32Slice {
    /// Exactly one immutable original GGUF model and actual consecutive IDs.
    /// These constructors confer pure-operator ownership, never SessionAuthority.
    #[allow(clippy::too_many_arguments)] // Explicit source, temporal and physical scope.
    pub fn observed_pair(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        left: ExternalBulkF32<'_>,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::bind(
            model,
            base,
            capacity,
            ids,
            None,
            Some(left),
            right,
            admission,
            cancelled,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn source_projection(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        role: QwenMatrixRole,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::bind(
            model,
            base,
            capacity,
            ids,
            Some(role),
            None,
            right,
            admission,
            cancelled,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn bind(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        source_role: Option<QwenMatrixRole>,
        external_left: Option<ExternalBulkF32<'_>>,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || admission.resource.backend != "host"
        {
            return Err(Error::Unsupported(
                "bulk pure GGUF F32 existing host scope".into(),
            ));
        }
        // Bounded original metadata is admitted BEFORE span/descriptor allocation.
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let span = TeacherSpan::new(
            model.source_identity(),
            base,
            capacity,
            model.geometry().vocab_size,
            ids,
        )?;
        Self::bind_prepared(
            model,
            span,
            source_role,
            external_left,
            right,
            admission,
            cancelled,
            metadata,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn bind_for_span(
        model: Arc<LoadedQwenModel>,
        span: TeacherSpan,
        source_role: Option<QwenMatrixRole>,
        external_left: Option<ExternalBulkF32<'_>>,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        Self::bind_prepared(
            model,
            span,
            source_role,
            external_left,
            right,
            admission,
            cancelled,
            metadata,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn bind_prepared(
        model: Arc<LoadedQwenModel>,
        span: TeacherSpan,
        source_role: Option<QwenMatrixRole>,
        external_left: Option<ExternalBulkF32<'_>>,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
        metadata: GlobalResourceLease,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || admission.resource.backend != "host"
            || span.source() != model.source_identity()
            || span
                .ids()
                .iter()
                .any(|&id| u64::from(id) >= model.geometry().vocab_size)
        {
            return Err(Error::Unsupported(
                "bulk source/span/profile/host scope".into(),
            ));
        }
        let left = match (source_role, &external_left) {
            (Some(role), None) => {
                let tensor = model.matrix_descriptor(role)?;
                BulkF32Operand::source_matrix(span.clone(), model.as_ref(), &tensor.name)?
            }
            (None, Some(value)) => {
                BulkF32Operand::external(span.clone(), BulkF32Side::Left, value.name, value.layout)?
            }
            _ => return Err(Error::State("bulk left source transport ambiguity".into())),
        };
        let rhs = BulkF32Operand::external(span, BulkF32Side::Right, right.name, right.layout)?;
        let matrix = BulkF32Matrix::new(left, rhs, BulkF32Contract::HaswellTinyBlasF32V1)?;
        for (operand, payload) in [
            (matrix.left(), external_left.as_ref().map(|v| v.backing)),
            (matrix.right(), Some(right.backing)),
        ] {
            if payload.is_some_and(|data| data.len() != operand.layout().backing_words()) {
                return Err(Error::Malformed(
                    "bulk exact external allocation backing".into(),
                ));
            }
        }
        let input_words = right
            .backing
            .len()
            .checked_add(external_left.as_ref().map_or(0, |v| v.backing.len()))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("bulk aggregate staging bytes".into()))?;
        // Both private external payloads share ONE exact staging ticket before
        // either allocation. Learned raw matrix remains borrowed from its owner.
        let staging = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: input_words as u64,
            }],
        )?;
        let owner = Arc::new(BulkOwner {
            model,
            matrix,
            admission,
            _metadata: metadata,
        });
        let copy = |data: &[f32]| -> Result<Vec<f32>> {
            not_cancelled(cancelled)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(data.len())
                .map_err(|_| Error::Limit("bulk external payload allocation".into()))?;
            if values.capacity() != data.len() {
                return Err(Error::Limit("bulk external actual capacity".into()));
            }
            values.extend_from_slice(data);
            not_cancelled(cancelled)?;
            Ok(values)
        };
        let left = match external_left {
            Some(value) => LeftInput::External(copy(value.backing)?),
            None => LeftInput::Source,
        };
        let inputs = Arc::new(BulkInputs {
            owner,
            left,
            right: RightInput::External(copy(right.backing)?),
            _staging: Some(staging),
        });
        // Raw source alignment and all logical finite words are subsequently
        // checked by the unchanged primitive before its own output allocation.
        inputs.values(BulkF32Side::Left)?;
        not_cancelled(cancelled)?;
        Ok(Self { inputs })
    }
    fn bind_prefix(
        prefix: GgufPrefixOutput,
        role: QwenMatrixRole,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::math::haswell_bulk_f32::verify_environment()?;
        let model = prefix.model().clone();
        let admission = prefix.admission().clone();
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let tensor = model.matrix_descriptor(role)?;
        let left =
            BulkF32Operand::source_matrix(prefix.span().clone(), model.as_ref(), &tensor.name)?;
        let right = BulkF32Operand::retained_normalized(prefix.descriptor().clone())?;
        let matrix = BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1)?;
        let owner = Arc::new(BulkOwner {
            model,
            matrix,
            admission,
            _metadata: metadata,
        });
        let inputs = Arc::new(BulkInputs {
            left: LeftInput::Source,
            right: RightInput::ProducedPrefix(prefix),
            _staging: None,
            owner,
        });
        inputs.values(BulkF32Side::Left)?;
        inputs.values(BulkF32Side::Right)?;
        not_cancelled(cancelled)?;
        Ok(Self { inputs })
    }
    pub fn matrix(&self) -> &BulkF32Matrix {
        &self.inputs.owner.matrix
    }
    pub fn record(&self) -> Result<RecordedOperatorSlice> {
        let mut builder = PlanBuilder::new_operator_slice(
            self.matrix().source(),
            ProgramNumericalProfile::GgufSourceV1,
        )?;
        let output = crate::forward::bulk_f32_slice(&mut builder, self.matrix())?;
        builder.finish_operator_slice(output)
    }
    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkF32Output> {
        self.execute_with_provider(None, cancelled)
    }
    pub fn execute_with_provider(
        &self,
        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<BulkF32Output> {
        not_cancelled(cancelled)?;
        if kernel.is_some() {
            return Err(Error::Unsupported(
                "unaware provider cannot execute Haswell matrix contract".into(),
            ));
        }
        let mut math = crate::host_math::HostMath::for_bulk_f32_slice(self, cancelled);
        let value = crate::forward::bulk_f32_slice(&mut math, self.matrix())?;
        not_cancelled(cancelled)?;
        Ok(BulkF32Output {
            value,
            shape: self.matrix().output().shape(),
            source: self.matrix().source(),
        })
    }
}
#[derive(Clone)]
pub struct BulkF32Output {
    value: crate::host_math::HostValue,
    shape: [usize; 4],
    source: crate::accelerator::SourceIdentity,
}
impl BulkF32Output {
    pub fn values(&self) -> Result<&[f32]> {
        self.value.as_slice()
    }
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn contract(&self) -> BulkF32Contract {
        BulkF32Contract::HaswellTinyBlasF32V1
    }
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<&[f32]> {
        self.values()?
            .get(range)
            .ok_or_else(|| Error::Malformed("bulk borrowed output range".into()))
    }
}
pub(crate) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("closed bulk matrix cancelled".into()));
    }
    Ok(())
}
