//! Exact original generated-kernel constructors; no handles, math or commands.
use super::*;
use crate::resident_program::gemma_text::{
    Gemma4Bf16NormSource, Gemma4Bf16TextPlan, PlanOperation, ValueShape,
};
use cubecl::ir::{DeviceProperties, TargetProperties};
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;
use std::{collections::HashSet, sync::Arc};
use zerogravity_artifacts::source_weights::WindowWorkspace;

const MAX_KERNELS: usize = 1024;
const PREPARATION_METADATA_BYTES: usize = 1024 * 1024;
pub(in crate::resident_program::gemma_text) struct Specializations {
    pub(in crate::resident_program::gemma_text) kernels: Vec<Box<dyn CubeKernel>>,
    pub(in crate::resident_program::gemma_text) _metadata: WindowWorkspace,
}
struct Builder {
    properties: Arc<DeviceProperties>,
    target: Arc<TargetProperties>,
    ids: HashSet<KernelId>,
    kernels: Vec<Box<dyn CubeKernel>>,
}
impl Builder {
    fn add(&mut self, kernel: impl CubeKernel + 'static) -> Result<()> {
        let id = kernel.id();
        if self.ids.contains(&id) {
            return Ok(());
        }
        if self.kernels.len() == MAX_KERNELS {
            return Err(Error::Limit(
                "complete prepared kernel specialization count".into(),
            ));
        }
        self.ids.insert(id);
        self.kernels.push(Box::new(kernel));
        Ok(())
    }
    fn settings(&self, lanes: u32) -> KernelSettings {
        KernelSettings::new(
            CubeDim::new_1d(lanes).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
    }
}
fn buffer() -> BufferCompilationArg {
    BufferCompilationArg { inplace: None }
}
macro_rules! add {
    ($builder:ident, $kernel:path, $lanes:expr $(, $arg:expr)* $(,)?) => {
        $builder.add(<$kernel>::new($builder.settings($lanes), $builder.properties.clone(), $builder.target.clone(), $($arg),*))?
    };
}
fn finite(b: &mut Builder, n: usize) -> Result<()> {
    add!(
        b,
        crate::program_kernels::certify_finite_strided::CertifyFiniteStrided,
        32,
        buffer(),
        buffer(),
        (),
        (),
        1usize,
        n
    );
    Ok(())
}
fn copy(b: &mut Builder, rows: usize, width: usize, finite: bool) -> Result<()> {
    add!(
        b,
        crate::program_kernels::copy_strided_words::CopyStridedWords,
        32,
        buffer(),
        buffer(),
        buffer(),
        (),
        (),
        (),
        (),
        rows,
        width,
        finite
    );
    Ok(())
}
/// All execution rows use the same source-issued traversal. Runtime token,
/// position, causal extent, query M and PV contraction are runtime arguments;
/// only original compile-time fields become specialization keys here.
pub(in crate::resident_program::gemma_text) fn enumerate(
    client: &Client,
    recorded: &Gemma4Bf16TextPlan,
    storage: crate::resident_program::gemma_text::GemmaTextStorage,
) -> Result<Specializations> {
    let selection =
        crate::resident_program::gemma_text::storage::Selection::derive(recorded, storage)?;
    let plan = recorded.plan();
    if plan.numerical_profile() != zerogravity_core::accelerator::program::ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1
        || plan.prefix_capacity() != 64 || plan.nodes().len() > 8192 || plan.values().len() > 4096
    { return Err(Error::Unsupported("complete source-issued Gemma kernel inventory".into())); }
    let metadata = recorded
        .source()
        .source()
        .reserve_workspace(PREPARATION_METADATA_BYTES)?;
    let mut b = Builder {
        properties: client.properties_shared(),
        target: client.target_properties_shared(),
        ids: HashSet::new(),
        kernels: Vec::new(),
    };
    add!(
        b,
        crate::program_kernels::clear_step_status::ClearStepStatus,
        32,
        buffer()
    );
    // Certificates on every actual public value and both whole canonical KV slabs.
    for value in plan.values() {
        finite(&mut b, value.capacity_elements()?)?;
    }
    let row = recorded.call().row(recorded.call().rows() - 1)?;
    for (index, node) in plan.nodes().iter().enumerate() {
        let count = node
            .inputs
            .first()
            .map(|id| plan.values()[id.0 as usize].capacity_elements())
            .transpose()?
            .unwrap_or(0);
        match &node.operation {
            PlanOperation::Embed { .. } => {
                add!(
                    b,
                    crate::resident_program::gemma_text::transport::widen::Widen,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    1u32
                );
            }
            PlanOperation::Project { tensor } => {
                add!(
                    b,
                    crate::resident_program::gemma_text::transport::pack::Pack,
                    32,
                    buffer(),
                    buffer(),
                    buffer()
                );
                add!(b, crate::resident_program::gemma_text::transport::projection_status::ProjectionStatus, 32, buffer(), buffer());
                let (tiles, geometry) = recorded.projection_tiles_for_row(index, &row)?;
                if geometry.input_columns != tensor.columns || geometry.output_rows != tensor.rows {
                    return Err(Error::State(
                        "prepared exact source projection geometry".into(),
                    ));
                }
                let operand = selection.nodes[index]
                    .and_then(|i| selection.operands.get(i))
                    .ok_or_else(|| {
                        Error::State("prepared BF16 shared source operand absent".into())
                    })?;
                if !operand.matrix().same_binding(tiles.matrix()) {
                    return Err(Error::State("prepared BF16 source member differs".into()));
                }
                for rows in operand.launch_rows() {
                    add!(
                        b,
                        crate::bf16_projection_kernel::project_cpu_bf16::ProjectCpuBf16,
                        64,
                        buffer(),
                        buffer(),
                        buffer(),
                        buffer(),
                        buffer(),
                        buffer(),
                        rows,
                        tensor.columns
                    );
                }
            }
            PlanOperation::Rms { groups, .. } => {
                let (width, unit) = match recorded.normalization_source(index)? {
                    Gemma4Bf16NormSource::Direct(p) => (
                        usize::try_from(p.shape[0])
                            .map_err(|_| Error::Limit("prepared source gamma width".into()))?,
                        false,
                    ),
                    Gemma4Bf16NormSource::Unit { width } => (*width, true),
                };
                if width.checked_mul(*groups) != Some(count) {
                    return Err(Error::State("prepared exact grouped RMS extent".into()));
                }
                add!(
                    b,
                    rms::squares::Squares,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    count
                );
                add!(b, rms::sum::Sum, 32, buffer(), buffer(), buffer(), width);
                for stage in 0..4u32 {
                    add!(
                        b,
                        rms::scalar::Scalar,
                        32,
                        buffer(),
                        buffer(),
                        buffer(),
                        stage,
                        width,
                        *groups
                    );
                }
                add!(
                    b,
                    rms::normalize::Normalize,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    width,
                    count
                );
                add!(
                    b,
                    rms::weight::Weight,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    width,
                    count,
                    if unit { 0u32 } else { 1u32 }
                );
                add!(
                    b,
                    rms::store::Store,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    count
                );
            }
            PlanOperation::TextElementwise(op) => {
                let kind = match op {
                    TextElementwise::Scale(_) => 0u32,
                    TextElementwise::ScaleBf16Scalar(_) => 1,
                    TextElementwise::Add => 2,
                    TextElementwise::Multiply => 3,
                    TextElementwise::GeluTanh => 4,
                    TextElementwise::TanhSoftcap(_) => 5,
                };
                add!(
                    b,
                    elementwise::operation::Operation,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    (),
                    kind,
                    count
                );
            }
            PlanOperation::TextRotary { geometry, .. } => {
                let width = geometry.head_width;
                add!(
                    b,
                    rotary::angles::Angles,
                    32,
                    buffer(),
                    buffer(),
                    (),
                    width / 2
                );
                add!(
                    b,
                    crate::source_trig::sincos_angles::SincosAngles,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    width / 2
                );
                add!(
                    b,
                    rotary::apply::Apply,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    width,
                    count,
                    geometry.head_width
                );
            }
            PlanOperation::TextWindowQk { geometry, .. } => {
                add!(
                    b,
                    attention::qk::Qk,
                    64,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    (),
                    (),
                    (),
                    geometry.width,
                    8usize,
                    1usize,
                    64usize
                );
                add!(
                    b,
                    crate::resident_bindings::certify_causal::CertifyCausal,
                    32,
                    buffer(),
                    buffer(),
                    (),
                    geometry.query_heads,
                    plan.prefix_capacity()
                );
            }
            PlanOperation::AttentionSoftmax { .. } => {
                for maximum in [true, false] {
                    add!(
                        b,
                        attention::reduce::Reduce,
                        32,
                        buffer(),
                        buffer(),
                        buffer(),
                        (),
                        maximum,
                        64usize
                    );
                }
                add!(
                    b,
                    attention::exponentials::Exponentials,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    (),
                    8usize,
                    64usize
                );
                add!(
                    b,
                    attention::inverse::Inverse,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    8usize
                );
                add!(
                    b,
                    attention::probabilities::Probabilities,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    (),
                    8usize,
                    64usize
                );
            }
            PlanOperation::AttentionValue { geometry, .. } => {
                add!(
                    b,
                    attention::pv::Pv,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    (),
                    (),
                    (),
                    geometry.width,
                    8usize,
                    1usize,
                    64usize
                );
            }
            PlanOperation::AppendKv => {
                add!(
                    b,
                    crate::resident_bindings::append_kv::AppendKv,
                    32,
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    buffer(),
                    (),
                    count,
                    plan.prefix_capacity()
                );
                finite(&mut b, count * plan.prefix_capacity())?;
            }
            PlanOperation::View(view) => copy(&mut b, view.rows, view.width, true)?,
            PlanOperation::Contiguous => copy(&mut b, 1, count, true)?,
            PlanOperation::Observe { .. } => {
                if let ValueShape::CausalRows { heads, capacity } =
                    plan.values()[node.inputs[0].0 as usize]
                {
                    for active in 1..=capacity {
                        copy(&mut b, heads, active, false)?;
                    }
                }
            }
            PlanOperation::CheckFinite { .. } | PlanOperation::KvPrefixGuard { .. } => {}
            _ => {
                return Err(Error::Unsupported(
                    "operation has no complete prepared Gemma native specialization inventory"
                        .into(),
                ));
            }
        }
    }
    let vocabulary = plan.values()[plan
        .logits()
        .ok_or_else(|| Error::State("complete source plan has no logits".into()))?
        .0 as usize]
        .capacity_elements()?;
    let mut partials = crate::program_kernels::argmax_partial_count(vocabulary, 256)?;
    add!(
        b,
        crate::program_kernels::argmax_logits_numeric_zero::ArgmaxLogitsNumericZero,
        32,
        buffer(),
        buffer(),
        buffer(),
        (),
        (),
        vocabulary,
        256usize
    );
    while partials > 1 {
        add!(
            b,
            crate::program_kernels::argmax_partials_numeric_zero::ArgmaxPartialsNumericZero,
            32,
            buffer(),
            buffer(),
            buffer(),
            (),
            (),
            partials,
            vocabulary,
            256usize
        );
        partials = crate::program_kernels::argmax_partial_count(partials, 256)?;
    }
    add!(
        b,
        crate::program_kernels::finish_argmax::FinishArgmax,
        32,
        buffer(),
        buffer(),
        buffer(),
        (),
        vocabulary
    );
    Ok(Specializations {
        kernels: b.kernels,
        _metadata: metadata,
    })
}
