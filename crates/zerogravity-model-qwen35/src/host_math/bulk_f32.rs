//! Closed input/result backing under the SAME HostMath, no layer/token engine.
use super::*;
use crate::accelerator::program::{BulkF32Matrix, BulkF32Operand, BulkF32Side};
use crate::operator_slice::bulk_f32::{
    BulkF32Slice, BulkInputs, BulkOwner, ZenSsmOutOwner, not_cancelled,
};
use std::sync::atomic::AtomicBool;
use zerogravity_cpu::numerics::haswell_bulk_f32::{
    BulkF32Admission, BulkF32Completed, BulkF32Profile, BulkF32Request, F32View,
};
use zerogravity_cpu::numerics::zen_blas_q8_f32::ZenQ8F32Completed;

pub(super) struct BulkCall<'a> {
    slice: &'a BulkF32Slice,
    cancelled: Option<&'a AtomicBool>,
}
pub(super) enum BulkBacking {
    GgufGdnInput {
        owner: Arc<crate::operator_slice::bulk_f32::GgufGdnUpstreamOwner>,
    },
    GgufGdnProjection {
        completed: Box<super::gdn_upstream::UpstreamProjection>,
    },
    GgufGdnConvolution {
        completed: Box<super::gdn_upstream::UpstreamConvolution>,
    },
    GgufGdnQk {
        completed: Box<super::gdn_upstream::UpstreamQk>,
    },
    GgufEmbedding {
        completed: Box<super::gguf_prefix::PrefixEmbedding>,
    },
    GgufRms {
        completed: Box<super::gguf_prefix::PrefixRms>,
    },
    ZenInput {
        owner: Arc<ZenSsmOutOwner>,
    },
    ZenOutput {
        completed: ZenQ8F32Completed,
        _owner: Arc<ZenSsmOutOwner>,
    },
    Input {
        inputs: Arc<BulkInputs>,
        side: BulkF32Side,
    },
    Output {
        completed: BulkF32Completed,
        owner: Arc<BulkOwner>,
    },
}
impl BulkBacking {
    pub(super) fn values(&self) -> Result<&[f32]> {
        match self {
            Self::GgufGdnInput { owner } => owner.prefix.normalized(),
            Self::GgufGdnProjection { completed } => Ok(completed.values()),
            Self::GgufGdnConvolution { completed } => Ok(completed.values()),
            Self::GgufGdnQk { completed } => Ok(completed.values()),
            Self::GgufEmbedding { completed } => Ok(completed.values()),
            Self::GgufRms { completed } => Ok(completed.values()),
            Self::ZenInput { owner } => Ok(owner.gated.gated()),
            Self::ZenOutput { completed, .. } => Ok(completed.values()),
            Self::Input { inputs, side } => inputs.values(*side),
            Self::Output { completed, .. } => Ok(completed.values()),
        }
    }
    fn owner(&self) -> Option<&Arc<BulkOwner>> {
        match self {
            Self::Input { inputs, .. } => Some(&inputs.owner),
            Self::Output { owner, .. } => Some(owner),
            Self::GgufGdnInput { .. }
            | Self::GgufGdnProjection { .. }
            | Self::GgufGdnConvolution { .. }
            | Self::GgufGdnQk { .. }
            | Self::GgufEmbedding { .. }
            | Self::GgufRms { .. }
            | Self::ZenInput { .. }
            | Self::ZenOutput { .. } => None,
        }
    }
}
fn fp_environment() -> Result<()> {
    zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()
}
fn request_view<'a>(
    source: crate::accelerator::SourceIdentity,
    operand: &'a BulkF32Operand,
    values: &'a [f32],
) -> F32View<'a> {
    let layout = operand.layout();
    F32View {
        source,
        name: operand.name(),
        values,
        offset: layout.offset(),
        shape: layout.shape(),
        strides: layout.strides(),
    }
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub(crate) fn for_bulk_f32_slice(
        slice: &'engine BulkF32Slice,
        cancelled: Option<&'engine AtomicBool>,
    ) -> Self {
        let mut math = Self::for_operator_slice(slice.inputs.owner.model.as_ref(), None);
        math.bulk_call = Some(BulkCall { slice, cancelled });
        math
    }
    fn bulk_call(&self) -> Result<&BulkCall<'engine>> {
        if self.engine.is_some()
            || self.state.is_some()
            || self.region_kernel.is_some()
            || self.zen_ssm_out_call.is_some()
            || self.gguf_prefix_call.is_some()
            || self.gguf_gdn_upstream_call.is_some()
            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
        {
            return Err(Error::Unsupported(
                "bulk F32 private same-engine host slice only".into(),
            ));
        }
        self.bulk_call
            .as_ref()
            .ok_or_else(|| Error::Unsupported("bulk F32 has no bound temporal input issuer".into()))
    }
    pub(super) fn import_bulk_f32(&self, operand: &BulkF32Operand) -> Result<HostValue> {
        let call = self.bulk_call()?;
        not_cancelled(call.cancelled)?;
        fp_environment()?;
        let owner = &call.slice.inputs.owner;
        let expected = match operand.side() {
            BulkF32Side::Left => owner.matrix.left(),
            BulkF32Side::Right => owner.matrix.right(),
        };
        if operand != expected || operand.span().source() != self.source.source_identity() {
            return Err(Error::State(
                "bulk input exact bound owner/span/layout".into(),
            ));
        }
        let backing = BulkBacking::Input {
            inputs: call.slice.inputs.clone(),
            side: operand.side(),
        };
        if backing.values()?.len() != operand.layout().backing_words() {
            return Err(Error::State("bulk actual input storage extent".into()));
        }
        HostValue::from_bulk(
            backing,
            operand.layout().backing_words(),
            self.source.source_identity(),
        )
    }
    pub(super) fn execute_bulk_f32(
        &self,
        matrix: &BulkF32Matrix,
        left: &HostValue,
        right: &HostValue,
    ) -> Result<HostValue> {
        let call = self.bulk_call()?;
        let owner = &call.slice.inputs.owner;
        not_cancelled(call.cancelled)?;
        fp_environment()?;
        if matrix != &owner.matrix {
            return Err(Error::State("bulk exact bound matrix/temporal span".into()));
        }
        for (value, operand) in [(left, matrix.left()), (right, matrix.right())] {
            let backing = value.data.bulk.as_ref().ok_or_else(|| {
                Error::State("bulk input has no admitted original backing".into())
            })?;
            if backing
                .owner()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
                || value.source != Some(matrix.source())
                || value.profile != Some(ProgramNumericalProfile::GgufSourceV1)
                || value.precision != ActivationPrecision::F32
                || value.elements != operand.layout().backing_words()
                || value.view != TensorView::vector(0, value.elements)
                || !matches!(backing,BulkBacking::Input{side,..}if *side==operand.side())
            {
                return Err(Error::State(
                    "bulk foreign/partial/retagged input value".into(),
                ));
            }
        }
        let a = left.as_slice()?;
        let b = right.as_slice()?;
        let extent = |values: &[f32]| -> Result<std::ops::Range<usize>> {
            let start = values.as_ptr() as usize;
            let end = values
                .len()
                .checked_mul(4)
                .and_then(|n| start.checked_add(n))
                .ok_or_else(|| Error::Limit("bulk actual allocation range".into()))?;
            Ok(start..end)
        };
        let ar = extent(a)?;
        let br = extent(b)?;
        if ar.start < br.end && br.start < ar.end {
            return Err(Error::State(
                "bulk independent input backing aliases".into(),
            ));
        }
        // Unchanged source-derived primitive reserves actual result+scratch
        // BEFORE its Vec allocation. No old MatVec/provider fallback exists.
        let completed = zerogravity_cpu::numerics::haswell_bulk_f32::execute(
            BulkF32Request {
                profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
                source: matrix.source(),
                left: request_view(matrix.source(), matrix.left(), a),
                right: request_view(matrix.source(), matrix.right(), b),
            },
            BulkF32Admission {
                arbiter: &owner.admission.arbiter,
                resource: &owner.admission.resource,
                owner: &owner.admission.owner,
            },
            call.cancelled,
        )?;
        fp_environment()?;
        not_cancelled(call.cancelled)?;
        if completed.source() != matrix.source()
            || completed.shape() != matrix.output().shape()
            || completed.values().len() != matrix.output().backing_words()
        {
            return Err(Error::State(
                "bulk actual primitive result geometry/source".into(),
            ));
        }
        let value = HostValue::from_bulk(
            BulkBacking::Output {
                completed,
                owner: owner.clone(),
            },
            matrix.output().backing_words(),
            matrix.source(),
        )?;
        not_cancelled(call.cancelled)?;
        Ok(value)
    }
}
impl HostValue {
    pub(super) fn from_bulk(
        backing: BulkBacking,
        elements: usize,
        source: crate::accelerator::SourceIdentity,
    ) -> Result<Self> {
        if !source.is_bound() || backing.values()?.len() != elements {
            return Err(Error::State("bulk retained backing extent/source".into()));
        }
        Ok(Self {
            data: Arc::new(HostStorage {
                values: Vec::new(),
                #[cfg(all(feature = "source-binding", target_os = "linux"))]
                workspace: None,
                bulk: Some(backing),
            }),
            view: TensorView::vector(0, elements),
            elements,
            precision: ActivationPrecision::F32,
            source: Some(source),
            profile: Some(ProgramNumericalProfile::GgufSourceV1),
        })
    }
}
