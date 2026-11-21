//! Cold source/geometry kernels. Row counts, positions and IDs remain runtime data.
use super::*;
use crate::resident_program::bf16_projection::PreparedBf16Projection;
use cubecl::{client::Client, prelude::*};
use cubecl_server::kernel::CubeKernel;
use zerogravity_core::accelerator::program::SourceOperatorKind;

pub(super) fn kernels(
    client: &Client,
    l: &layout::Layout,
    operands: &BTreeMap<String, Arc<PreparedBf16Projection>>,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    let mut out: Vec<Box<dyn CubeKernel>> = Vec::new();
    let b = || BufferCompilationArg { inplace: None };
    macro_rules! add {
        ($k:path,$lanes:expr,$($a:expr),* $(,)?)=>{out.push(Box::new(<$k>::new(
            KernelSettings::new(CubeDim::new_1d($lanes).into(),ExecutionMode::Unchecked,AddressType::U32),
            client.properties_shared(),client.target_properties_shared(),$($a),*
        )))}
    }
    add!(super::super::bf16_projection::pack::Pack, 32, b(), b(), b());
    add!(
        super::super::bf16_projection::widen::Widen,
        32,
        b(),
        b(),
        b(),
        crate::program_kernels::NONFINITE_VALUE
    );
    add!(
        super::super::bf16_projection::projection_status::ProjectionStatus,
        32,
        b(),
        b()
    );
    add!(
        super::kernels::residual::Residual,
        32,
        b(),
        b(),
        b(),
        b(),
        ()
    );
    add!(
        super::kernels::multiply::Multiply,
        32,
        b(),
        b(),
        b(),
        b(),
        ()
    );
    add!(
        crate::source_sigmoid::call_shift::CallShift,
        32,
        b(),
        b(),
        b(),
        ()
    );
    add!(
        crate::source_sigmoid::call::Call,
        32,
        b(),
        b(),
        b(),
        b(),
        b(),
        ()
    );
    for p in operands.values() {
        for rows in p.launch_rows() {
            add!(
                crate::bf16_projection_kernel::project_cpu_bf16::ProjectCpuBf16,
                64,
                b(),
                b(),
                b(),
                b(),
                b(),
                b(),
                rows,
                p.matrix().columns(),
                p.matrix().rows()
            );
        }
    }
    for node in l.plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor: d, .. } = &node.operation else {
            unreachable!()
        };
        match d.kind() {
            SourceOperatorKind::Projection { .. } => {
                let width = d.matrices().iter().map(|m| m.rows).sum::<usize>();
                for m in d.matrices() {
                    add!(
                        super::kernels::columns::Columns,
                        32,
                        b(),
                        b(),
                        b(),
                        (),
                        (),
                        (),
                        m.rows,
                        m.rows,
                        width
                    );
                }
            }
            SourceOperatorKind::Rms(g) | SourceOperatorKind::ResidualRms(g) => {
                out.extend(crate::source_rms::prepared::kernels(
                    client,
                    g.width,
                    1,
                    crate::source_rms::Gamma::OnePlusRaw,
                )?);
            }
            SourceOperatorKind::CausalConvSplit { geometry: g, .. } => {
                add!(
                    super::kernels::convolution::Convolution,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    (),
                    (),
                    g.qkv_channels,
                    g.z_channels
                );
            }
            SourceOperatorKind::GatedDelta { geometry: g, .. } => {
                let kh = g.key_heads;
                let vh = g.value_heads;
                let values = vh * 128;
                let state = vh * 16384;
                add!(
                    super::kernels::repeat_qkv::RepeatQkv,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    (),
                    kh,
                    vh
                );
                add!(
                    super::kernels::coefficients::Coefficients,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    (),
                    vh
                );
                add!(
                    crate::source_standalone_exp::normal_vector::NormalVector,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    vh
                );
                add!(
                    crate::source_gdn::recurrent::normalize_qk::NormalizeQk,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    kh,
                    vh,
                    true
                );
                add!(
                    crate::source_gdn::recurrent::decayed::Decayed,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    state,
                    false
                );
                for round in [false, true] {
                    add!(
                        crate::source_gdn::recurrent::key_reduce::KeyReduce,
                        32,
                        b(),
                        b(),
                        b(),
                        b(),
                        values,
                        round
                    );
                }
                add!(
                    crate::source_gdn::recurrent::delta::Delta,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    kh,
                    values
                );
                add!(
                    crate::source_gdn::recurrent::state_update::StateUpdate,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    kh,
                    state,
                    false
                );
                out.extend(crate::source_gdn::chunk::prepared::kernels(client, vh)?);
                out.extend(crate::source_rms::prepared::kernels(
                    client,
                    128,
                    vh,
                    crate::source_rms::Gamma::DirectF32AfterBf16,
                )?);
                let bf16 = d.parameters()[2].kind
                    == zerogravity_core::encoding::StoredTensorEncoding::Scalar(
                        zerogravity_core::encoding::ScalarDType::Bf16,
                    );
                add!(
                    crate::source_activation::gated::Gated,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    values,
                    bf16
                );
            }
            SourceOperatorKind::QkRmsRotaryGated(g) => {
                let qh = g.attention.query_heads;
                let kh = g.attention.kv_heads;
                add!(
                    super::kernels::split_attention::SplitAttention,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    (),
                    qh,
                    kh
                );
                add!(
                    crate::source_rotary::kernels::angles::Angles,
                    32,
                    b(),
                    b(),
                    (),
                    32usize
                );
                add!(
                    crate::source_trig::cpu::sincos_angles::SincosAngles,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    32usize
                );
                for heads in [qh, kh] {
                    let n = heads * 256;
                    out.extend(crate::source_rms::prepared::kernels(
                        client,
                        256,
                        heads,
                        crate::source_rms::Gamma::OnePlusRaw,
                    )?);
                    add!(
                        super::kernels::columns::Columns,
                        32,
                        b(),
                        b(),
                        b(),
                        (),
                        (),
                        (),
                        n,
                        n,
                        n
                    );
                    add!(
                        crate::source_rotary::kernels::apply::Apply,
                        32,
                        b(),
                        b(),
                        b(),
                        b(),
                        256usize,
                        n,
                        64usize
                    );
                }
            }
            SourceOperatorKind::KvAppend {
                key_width,
                packed_value,
                ..
            } => {
                add!(
                    super::kernels::append_kv::AppendKv,
                    32,
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    b(),
                    (),
                    (),
                    (),
                    key_width,
                    packed_value.row_stride,
                    l.plan.prefix_capacity()
                );
            }
            SourceOperatorKind::FlashAttention { geometry: g, .. } => {
                out.extend(crate::source_attention::rows::kernels(
                    client,
                    crate::source_attention::rows::Geometry::new(
                        g.query_heads,
                        g.kv_heads,
                        l.plan.prefix_capacity(),
                    )?,
                ));
            }
            SourceOperatorKind::SiluAndMul { width } => {
                add!(
                    super::kernels::silu_mul::SiluMul,
                    32,
                    b(),
                    b(),
                    b(),
                    (),
                    width
                );
            }
            SourceOperatorKind::EmbeddingRows
            | SourceOperatorKind::Sigmoid { .. }
            | SourceOperatorKind::Multiply { .. } => {}
            _ => {
                return Err(Error::Unsupported(
                    "text-execution cold kernel implementation absent".into(),
                ));
            }
        }
    }
    let mut ids = std::collections::HashSet::new();
    out.retain(|k| ids.insert(k.id()));
    if out.is_empty() || out.len() > 1024 {
        return Err(Error::Limit("text-execution unique kernel count".into()));
    }
    Ok(out)
}
