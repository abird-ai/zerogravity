//! Same generated CubeKernel constructors used by actual source operations.
use super::super::{PlanOperation, RecordedWindowToken, StateLayout, layout};
use super::*;
use cubecl::ir::{DeviceProperties, TargetProperties};
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;
use std::{collections::HashSet, sync::Arc};
use zerogravity_artifacts::source_weights::WindowWorkspace;
pub(in crate::resident_program::qwen_source) struct Specializations {
    pub kernels: Vec<Box<dyn CubeKernel>>,
    pub metadata: WindowWorkspace,
}
struct Builder {
    properties: Arc<DeviceProperties>,
    target: Arc<TargetProperties>,
    ids: HashSet<KernelId>,
    kernels: Vec<Box<dyn CubeKernel>>,
}
impl Builder {
    fn settings(&self, lanes: u32) -> KernelSettings {
        KernelSettings::new(
            CubeDim::new_1d(lanes).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
    }
    fn add(&mut self, k: impl CubeKernel + 'static) -> Result<()> {
        if self.ids.insert(k.id()) {
            if self.kernels.len() == 1024 {
                return Err(Error::Limit(
                    "source finite complete native catalogue".into(),
                ));
            }
            self.kernels.push(Box::new(k));
        }
        Ok(())
    }
}
fn buffer() -> BufferCompilationArg {
    BufferCompilationArg { inplace: None }
}
fn finite(b: &mut Builder, n: usize) -> Result<()> {
    b.add(
        crate::program_kernels::certify_finite_strided::CertifyFiniteStrided::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
            buffer(),
            (),
            (),
            1usize,
            n,
        ),
    )
}
fn copy(b: &mut Builder, rows: usize, width: usize) -> Result<()> {
    b.add(
        crate::program_kernels::copy_strided_words::CopyStridedWords::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
            buffer(),
            buffer(),
            (),
            (),
            (),
            (),
            rows,
            width,
            true,
        ),
    )
}
fn add_gdn_convolution(b: &mut Builder, channels: usize, initial: bool) -> Result<()> {
    b.add(<gdn::convolution::Convolution>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        channels,
        initial,
    ))
}

fn add_gdn_full_convolution(b: &mut Builder, channels: usize) -> Result<()> {
    b.add(<gdn::full_convolution::FullConvolution>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        channels,
    ))
}

fn add_gdn_normalize_qk(b: &mut Builder, kh: usize, vh: usize, cached: bool) -> Result<()> {
    b.add(<gdn::normalize_qk::NormalizeQk>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        kh,
        vh,
        cached,
    ))
}

fn add_gdn_initial_dot(b: &mut Builder, heads: usize) -> Result<()> {
    b.add(<gdn::initial_dot::InitialDot>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        heads,
    ))
}

fn add_gdn_decayed(b: &mut Builder, elements: usize, initial: bool) -> Result<()> {
    b.add(<gdn::decayed::Decayed>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        elements,
        initial,
    ))
}

fn add_gdn_key_reduce(b: &mut Builder, values: usize, round_bf16: bool) -> Result<()> {
    b.add(<gdn::key_reduce::KeyReduce>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        values,
        round_bf16,
    ))
}

fn add_gdn_delta(b: &mut Builder, kh: usize, values: usize) -> Result<()> {
    b.add(<gdn::delta::Delta>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        kh,
        values,
    ))
}

fn add_gdn_state_update(b: &mut Builder, kh: usize, elements: usize, initial: bool) -> Result<()> {
    b.add(<gdn::state_update::StateUpdate>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        kh,
        elements,
        initial,
    ))
}

fn add_gdn_initial_core(b: &mut Builder, kh: usize, values: usize) -> Result<()> {
    b.add(<gdn::initial_core::InitialCore>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        kh,
        values,
    ))
}

fn add_coefficients_biased_softplus(b: &mut Builder, heads: usize) -> Result<()> {
    b.add(<coefficients::biased_softplus::BiasedSoftplus>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        heads,
    ))
}

fn add_coefficients_log_decay(b: &mut Builder, heads: usize) -> Result<()> {
    b.add(<coefficients::log_decay::LogDecay>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        heads,
    ))
}

fn add_coefficients_sigmoid_shift(b: &mut Builder, elements: usize) -> Result<()> {
    b.add(<coefficients::sigmoid_shift::SigmoidShift>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        elements,
    ))
}

fn add_coefficients_sigmoid(b: &mut Builder, elements: usize) -> Result<()> {
    b.add(<coefficients::sigmoid::Sigmoid>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        elements,
    ))
}

fn add_coefficients_gated(b: &mut Builder, elements: usize, bf16_gamma: bool) -> Result<()> {
    b.add(<coefficients::gated::Gated>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        elements,
        bf16_gamma,
    ))
}

fn add_coefficients_ffn_silu_mul(b: &mut Builder, elements: usize) -> Result<()> {
    b.add(<coefficients::ffn_silu_mul::FfnSiluMul>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        elements,
    ))
}

fn add_attention_append(b: &mut Builder, slab: usize, position: usize) -> Result<()> {
    b.add(<attention::append::Append>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        slab,
        position,
    ))
}

fn add_attention_gate(b: &mut Builder, elements: usize) -> Result<()> {
    b.add(<attention::gate::Gate>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        elements,
    ))
}

fn add_source_rms_kernels_squares(b: &mut Builder, count: usize) -> Result<()> {
    b.add(<crate::source_rms::kernels::squares::Squares>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        count,
    ))
}

fn add_source_rms_kernels_sum(b: &mut Builder, width: usize) -> Result<()> {
    b.add(<crate::source_rms::kernels::sum::Sum>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        width,
    ))
}

fn add_source_rms_kernels_scalar(
    b: &mut Builder,
    stage: u32,
    width: usize,
    rows: usize,
) -> Result<()> {
    b.add(<crate::source_rms::kernels::scalar::Scalar>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        stage,
        width,
        rows,
    ))
}

fn add_source_rms_kernels_normalize(b: &mut Builder, width: usize, count: usize) -> Result<()> {
    b.add(<crate::source_rms::kernels::normalize::Normalize>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        width,
        count,
    ))
}

fn add_source_rms_kernels_weight(
    b: &mut Builder,
    width: usize,
    count: usize,
    mode: u32,
) -> Result<()> {
    b.add(<crate::source_rms::kernels::weight::Weight>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        width,
        count,
        mode,
    ))
}

fn add_source_rms_kernels_store(b: &mut Builder, count: usize) -> Result<()> {
    b.add(<crate::source_rms::kernels::store::Store>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        count,
    ))
}

fn add_source_attention_kernels_qk(
    b: &mut Builder,
    width: usize,
    heads: usize,
    kv_heads: usize,
    capacity: usize,
) -> Result<()> {
    b.add(<crate::source_attention::kernels::qk::Qk>::new(
        b.settings(64),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        (),
        (),
        (),
        width,
        heads,
        kv_heads,
        capacity,
    ))
}

fn add_source_attention_kernels_reduce(
    b: &mut Builder,
    is_max: bool,
    capacity: usize,
) -> Result<()> {
    b.add(<crate::source_attention::kernels::reduce::Reduce>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        (),
        is_max,
        capacity,
    ))
}

fn add_source_attention_kernels_exponentials(
    b: &mut Builder,
    heads: usize,
    capacity: usize,
) -> Result<()> {
    b.add(
        <crate::source_attention::kernels::exponentials::Exponentials>::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
            buffer(),
            buffer(),
            buffer(),
            (),
            heads,
            capacity,
        ),
    )
}

fn add_source_attention_kernels_inverse(b: &mut Builder, heads: usize) -> Result<()> {
    b.add(<crate::source_attention::kernels::inverse::Inverse>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        heads,
    ))
}

fn add_source_attention_kernels_probabilities(
    b: &mut Builder,
    heads: usize,
    capacity: usize,
) -> Result<()> {
    b.add(
        <crate::source_attention::kernels::probabilities::Probabilities>::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
            buffer(),
            buffer(),
            buffer(),
            (),
            heads,
            capacity,
        ),
    )
}

fn add_source_attention_kernels_pv(
    b: &mut Builder,
    width: usize,
    heads: usize,
    kv_heads: usize,
    capacity: usize,
) -> Result<()> {
    b.add(<crate::source_attention::kernels::pv::Pv>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        (),
        (),
        (),
        width,
        heads,
        kv_heads,
        capacity,
    ))
}

fn add_source_rotary_kernels_angles(b: &mut Builder, half: usize) -> Result<()> {
    b.add(<crate::source_rotary::kernels::angles::Angles>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        (),
        half,
    ))
}

fn add_source_rotary_kernels_apply(
    b: &mut Builder,
    width: usize,
    count: usize,
    rotation: usize,
) -> Result<()> {
    b.add(<crate::source_rotary::kernels::apply::Apply>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        width,
        count,
        rotation,
    ))
}

fn add_source_standalone_exp_normal_vector(b: &mut Builder, elements: usize) -> Result<()> {
    b.add(
        <crate::source_standalone_exp::normal_vector::NormalVector>::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
            buffer(),
            buffer(),
            buffer(),
            elements,
        ),
    )
}

fn add_source_trig_sincos_angles(b: &mut Builder, half: usize) -> Result<()> {
    b.add(<crate::source_trig::sincos_angles::SincosAngles>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        half,
    ))
}

fn add_transport_widen(b: &mut Builder) -> Result<()> {
    b.add(<super::super::transport::widen::Widen>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        crate::program_kernels::NONFINITE_VALUE,
    ))
}

fn add_transport_pack(b: &mut Builder) -> Result<()> {
    b.add(<crate::resident_program::bf16_projection::pack::Pack>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
    ))
}

fn add_transport_projection_status(b: &mut Builder) -> Result<()> {
    b.add(
        <crate::resident_program::bf16_projection::projection_status::ProjectionStatus>::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
            buffer(),
        ),
    )
}

fn add_transport_state_copy(b: &mut Builder, active: usize, capacity: usize) -> Result<()> {
    b.add(<super::super::transport::state_copy::StateCopy>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        active,
        capacity,
    ))
}

fn add_transport_add(b: &mut Builder, elements: usize) -> Result<()> {
    b.add(<super::super::transport::add::Add>::new(
        b.settings(32),
        b.properties.clone(),
        b.target.clone(),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
        elements,
    ))
}
fn rms(b: &mut Builder, width: usize, groups: usize, mode: crate::source_rms::Gamma) -> Result<()> {
    let count = width * groups;
    add_source_rms_kernels_squares(b, count)?;
    add_source_rms_kernels_sum(b, width)?;
    for stage in 0..4 {
        add_source_rms_kernels_scalar(b, stage, width, groups)?;
    }
    add_source_rms_kernels_normalize(b, width, count)?;
    add_source_rms_kernels_weight(b, width, count, mode.mode())?;
    if mode != crate::source_rms::Gamma::DirectF32AfterBf16 {
        add_source_rms_kernels_store(b, count)?;
    }
    Ok(())
}
/// Exact finite two-record union. Runtime input values never select compiler keys.
pub(in crate::resident_program::qwen_source) fn enumerate(
    client: &Client,
    records: &[Arc<RecordedWindowToken>; 2],
    readout: super::super::QwenSourceReadout,
) -> Result<Specializations> {
    if records[0].position() != 0
        || records[1].position() != 1
        || !records[0]
            .source()
            .source()
            .same_issuer(records[1].source().source())
    {
        return Err(Error::State(
            "original source initial/cached record pair".into(),
        ));
    }
    let metadata = records[0]
        .source()
        .source()
        .reserve_workspace(1024 * 1024)?;
    let mut b = Builder {
        properties: client.properties_shared(),
        target: client.target_properties_shared(),
        ids: HashSet::new(),
        kernels: Vec::new(),
    };
    b.add(
        crate::program_kernels::clear_step_status::ClearStepStatus::new(
            b.settings(32),
            b.properties.clone(),
            b.target.clone(),
            buffer(),
        ),
    )?;
    for record in records {
        let l = layout::Layout::derive(record)?;
        let plan = record.plan();
        for node in plan.nodes() {
            for id in &node.outputs {
                finite(&mut b, plan.values()[id.0 as usize].capacity_elements()?)?;
            }
            let count =
                |id: super::super::ValueId| plan.values()[id.0 as usize].capacity_elements();
            match &node.operation {
                PlanOperation::TypedEmbed { tensor, .. } => {
                    finite(&mut b, tensor.columns)?;
                    add_transport_widen(&mut b)?;
                }
                PlanOperation::TypedProject { tensor, .. } => {
                    let matrix = record.source().source().matrix(&tensor.name, tensor.axes)?;
                    let operand = crate::resident_program::bf16_projection::PreparedBf16Projection::new(
                        record.source().source().clone(), matrix,
                        crate::resident_program::bf16_projection::Bf16ProjectionStorage::Streamed,
                    )?;
                    if operand.max_launch_rows() > 2048 {
                        return Err(Error::Unsupported(
                            "finite full-K native source catalogue".into(),
                        ));
                    }
                    add_transport_pack(&mut b)?;
                    add_transport_projection_status(&mut b)?;
                    for rows in operand.launch_rows() {
                        b.add(
                            crate::bf16_projection_kernel::project_cpu_bf16::ProjectCpuBf16::new(
                                b.settings(64),
                                b.properties.clone(),
                                b.target.clone(),
                                buffer(),
                                buffer(),
                                buffer(),
                                buffer(),
                                buffer(),
                                buffer(),
                                rows,
                                tensor.columns,
                            ),
                        )?;
                    }
                }
                PlanOperation::TypedOrdinaryRms {
                    parameter,
                    contract,
                } => {
                    let n = count(node.inputs[0])?;
                    let width = contract.validate(parameter, n)?;
                    rms(
                        &mut b,
                        width,
                        n / width,
                        crate::source_rms::Gamma::OnePlusRaw,
                    )?;
                }
                PlanOperation::TypedTorchGdn { contract, phase } => {
                    let (channels, values, state) = contract.validate(plan.source())?;
                    let initial = *phase == TorchCpuGdnPhase::InitialOne;
                    let (kh, heads) = (contract.key_heads, contract.value_heads);
                    add_gdn_convolution(&mut b, channels, initial)?;
                    add_coefficients_biased_softplus(&mut b, heads)?;
                    add_coefficients_log_decay(&mut b, heads)?;
                    add_source_standalone_exp_normal_vector(&mut b, heads)?;
                    add_coefficients_sigmoid_shift(&mut b, heads)?;
                    add_coefficients_sigmoid(&mut b, heads)?;
                    add_gdn_normalize_qk(&mut b, kh, heads, !initial)?;
                    add_gdn_decayed(&mut b, state, initial)?;
                    if initial {
                        add_gdn_initial_dot(&mut b, heads)?;
                        add_gdn_initial_core(&mut b, kh, values)?;
                    } else {
                        add_gdn_key_reduce(&mut b, values, false)?;
                        add_gdn_key_reduce(&mut b, values, true)?;
                        add_gdn_delta(&mut b, kh, values)?;
                    }
                    add_gdn_state_update(&mut b, kh, state, initial)?;
                    rms(
                        &mut b,
                        128,
                        heads,
                        crate::source_rms::Gamma::DirectF32AfterBf16,
                    )?;
                    add_coefficients_gated(
                        &mut b,
                        values,
                        contract.norm.kind == zerogravity_core::quant::QuantKind::Bf16,
                    )?;
                }
                PlanOperation::TypedTorchFullConvolution => {
                    add_gdn_full_convolution(&mut b, count(node.inputs[2])?)?
                }
                PlanOperation::TypedTorchAttention { geometry, position } => {
                    let (q, kv, tokens) = geometry.validate(*position)?;
                    if *position != record.position() {
                        return Err(Error::State("source attention phase catalogue".into()));
                    }
                    add_source_rotary_kernels_angles(&mut b, 32)?;
                    add_source_trig_sincos_angles(&mut b, 32)?;
                    for n in [q, kv] {
                        copy(&mut b, 1, n)?;
                        add_source_rotary_kernels_apply(&mut b, 256, n, 64)?;
                    }
                    add_attention_append(&mut b, kv, *position)?;
                    add_source_attention_kernels_qk(
                        &mut b,
                        256,
                        geometry.query_heads,
                        geometry.kv_heads,
                        tokens,
                    )?;
                    for is_max in [true, false] {
                        add_source_attention_kernels_reduce(&mut b, is_max, tokens)?;
                    }
                    add_source_attention_kernels_exponentials(
                        &mut b,
                        geometry.query_heads,
                        tokens,
                    )?;
                    add_source_attention_kernels_inverse(&mut b, geometry.query_heads)?;
                    add_source_attention_kernels_probabilities(
                        &mut b,
                        geometry.query_heads,
                        tokens,
                    )?;
                    add_source_attention_kernels_pv(
                        &mut b,
                        256,
                        geometry.query_heads,
                        geometry.kv_heads,
                        tokens,
                    )?;
                    add_coefficients_sigmoid_shift(&mut b, q)?;
                    add_coefficients_sigmoid(&mut b, q)?;
                    add_attention_gate(&mut b, q)?;
                }
                PlanOperation::TypedSourceStateRead { binding, .. } => {
                    for n in binding.components() {
                        copy(&mut b, 1, *n)?;
                    }
                }
                PlanOperation::TypedSourceStateWrite { binding } => {
                    let state = l
                        .states
                        .iter()
                        .find(|r| r.slot == binding.slot())
                        .ok_or_else(|| Error::State("source catalogue state owner".into()))?;
                    let capacity = match state.layout {
                        StateLayout::Vector { elements } => elements,
                        StateLayout::Kv { slab, capacity } => slab * capacity,
                    };
                    for n in binding.components() {
                        add_transport_state_copy(&mut b, *n, capacity)?;
                    }
                }
                PlanOperation::View(view) => copy(&mut b, view.rows, view.width)?,
                PlanOperation::Contiguous => copy(&mut b, 1, count(node.inputs[0])?)?,
                PlanOperation::Silu { multiplied: true } => {
                    add_coefficients_ffn_silu_mul(&mut b, count(node.inputs[0])?)?
                }
                PlanOperation::Add => add_transport_add(&mut b, count(node.inputs[0])?)?,
                PlanOperation::Observe { .. } | PlanOperation::CheckFinite { .. } => {}
                _ => {
                    return Err(Error::Unsupported(
                        "missing generated source operation catalogue".into(),
                    ));
                }
            }
        }
        let logits = plan
            .logits()
            .ok_or_else(|| Error::State("source catalogue full logits absent".into()))?;
        finite(
            &mut b,
            plan.values()[logits.0 as usize].capacity_elements()?,
        )?;
        for state in &l.states {
            match state.layout {
                StateLayout::Vector { elements } => finite(&mut b, elements)?,
                StateLayout::Kv { slab, capacity } => finite(&mut b, slab * capacity)?,
            }
        }
    }
    if readout.natural() {
        let plan = records[0].plan();
        let vocabulary = plan.values()[plan
            .logits()
            .ok_or_else(|| Error::State("source greedy logits absent".into()))?
            .0 as usize]
            .capacity_elements()?;
        for kernel in crate::resident_greedy::numeric_zero_kernels(client, vocabulary)? {
            if b.ids.insert(kernel.id()) {
                if b.kernels.len() == 1024 {
                    return Err(Error::Limit(
                        "source finite complete native catalogue".into(),
                    ));
                }
                b.kernels.push(kernel);
            }
        }
    }
    Ok(Specializations {
        kernels: b.kernels,
        metadata,
    })
}
