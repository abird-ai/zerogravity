//! Synthetic role/wiring/lease checks; genuine trained qualification is separate.
use super::*;
use crate::accelerator::program::{CausalConvSiluContract, ProgramNumericalProfile};
use crate::operator_slice::gdn_conv::{ConvInputPhase, GdnConvSliceInput};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

fn conv_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_key_head_dim"] = json!(8);
    cfg["text_config"]["linear_value_head_dim"] = json!(8);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, index, fallback| {
            if name.ends_with(".conv1d.weight") {
                if index % 4 == 3 { 0x3f80 } else { 0 }
            } else {
                fallback
            }
        },
        |_| {},
    )
}
fn bf16(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn private<'a>(
    source: crate::accelerator::SourceIdentity,
    phase: ConvInputPhase,
    x: &'a [u8],
    h: &'a [u8],
) -> GdnConvSliceInput<'a> {
    GdnConvSliceInput::from_bf16_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        phase,
        x,
        h,
    )
    .unwrap()
}
#[test]
fn pure_window_region_matches_closed_helpers_and_preserves_canonical_history_bits() {
    let f = Fixture::new();
    let loaded = conv_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let baseline = reserved(&admission);
    let x = bf16((0..32).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }));
    let h = bf16((0..32).flat_map(|_| [1.0, -0.0, 2.0]));
    let output = ops
        .execute_gdn_convolution_silu_slice(
            0,
            private(source.source_identity(), ConvInputPhase::CachedTail, &x, &h),
            None,
        )
        .unwrap();
    assert_eq!(output.layer(), 0);
    assert_eq!(output.phase(), ConvInputPhase::CachedTail);
    assert_eq!(output.source_identity(), source.source_identity());
    // Last tap 1 and three zero weights make each positive input exact.
    same(output.input().unwrap(), output.raw().unwrap());
    let silu = zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
        source.source_identity(),
        output.raw().unwrap(),
    )
    .unwrap();
    assert_eq!(silu.len(), output.activated().unwrap().len());
    for (expected, actual) in silu.iter().zip(output.activated().unwrap()) {
        let bits = expected.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
        assert_eq!(actual.to_bits(), rounded);
    }
    for (channel, next) in output.next_history().unwrap().chunks_exact(3).enumerate() {
        assert_eq!(next[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(next[1], 2.0);
        assert_eq!(
            next[2].to_bits(),
            output.input().unwrap()[channel].to_bits()
        );
    }
    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(reserved(&admission), baseline + 2 * 960 * 1024 + 32 * 36);
    assert_eq!(
        snapshot
            .per_class
            .get(&AllocationClass::Output)
            .copied()
            .unwrap_or(0),
        32 * 36
    );
    assert_eq!(
        snapshot
            .per_class
            .get(&AllocationClass::TransferStaging)
            .copied()
            .unwrap_or(0),
        0
    );
    let (model, _, _) = bind(loaded);
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, false).is_err());
    drop(ops);
    drop(source);
    assert_eq!(output.raw().unwrap().len(), 32);
    assert!(reserved(&admission) > 0);
    drop(output);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn initial_padding_is_explicit_and_never_canonicalizes_signed_zero() {
    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
    let x = bf16([1.0; 32]);
    let zero = bf16([0.0; 96]);
    let negative = bf16([-0.0; 96]);
    private(id, ConvInputPhase::InitialPadding, &x, &zero);
    assert!(
        GdnConvSliceInput::from_bf16_bytes(
            id,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            ConvInputPhase::InitialPadding,
            &x,
            &negative
        )
        .is_err()
    );
    private(id, ConvInputPhase::CachedTail, &x, &negative);
    for bytes in [vec![0; 63], vec![0xff; 64]] {
        assert!(
            GdnConvSliceInput::from_bf16_bytes(
                id,
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
                ConvInputPhase::CachedTail,
                &bytes,
                &zero
            )
            .is_err()
        );
    }
}
#[test]
fn aggregate_pressure_cancel_and_live_output_never_leak_or_reuse_admission() {
    let f = Fixture::new();
    let loaded = conv_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let baseline = reserved(&admission);
    let x = bf16([1.0; 32]);
    let h = bf16([0.0; 96]);
    let input = private(
        source.source_identity(),
        ConvInputPhase::InitialPadding,
        &x,
        &h,
    );
    let cancelled = AtomicBool::new(true);
    assert!(
        ops.execute_gdn_convolution_silu_slice(0, input, Some(&cancelled))
            .is_err()
    );
    assert_eq!(reserved(&admission), baseline);
    let pressure = admission
        .arbiter
        .admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024 - baseline,
            }],
        )
        .unwrap();
    assert!(
        ops.execute_gdn_convolution_silu_slice(0, input, None)
            .is_err()
    );
    drop(pressure);
    assert_eq!(reserved(&admission), baseline);
    // Leave enough for BOTH input Vecs, but one byte short of the aggregate
    // three-result envelope. A failed second admission must retire the inputs.
    let input_reservation = 960 * 1024 + 32 * 16;
    let result_reservation = 960 * 1024 + 32 * 20;
    let held = admission
        .arbiter
        .admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024 - baseline - input_reservation - result_reservation + 1,
            }],
        )
        .unwrap();
    let pressured = reserved(&admission);
    assert!(
        ops.execute_gdn_convolution_silu_slice(0, input, None)
            .is_err()
    );
    assert_eq!(reserved(&admission), pressured);
    drop(held);
    assert_eq!(reserved(&admission), baseline);
    // Cancellation between actual operand import and the region, without clocks.
    cancelled.store(false, Ordering::Release);
    let mut math =
        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
    let after_import = reserved(&admission);
    assert_eq!(after_import, baseline + 960 * 1024 + 32 * 16);
    let parameter = ops
        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
        .unwrap();
    cancelled.store(true, Ordering::Release);
    assert!(
        math.causal_conv_silu_profiled(
            parameter.clone(),
            &activation,
            &history,
            CausalConvSiluContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    assert_eq!(reserved(&admission), after_import);
    cancelled.store(false, Ordering::Release);
    let values = math
        .causal_conv_silu_profiled(
            parameter,
            &activation,
            &history,
            CausalConvSiluContract::TorchCpuBf16Avx2,
        )
        .unwrap();
    let clone = values.activated.clone();
    assert!(clone.clone().into_vec().is_err());
    drop(values);
    assert_eq!(clone.as_slice().unwrap().len(), 32);
    drop(clone);
    assert_eq!(reserved(&admission), after_import);
    drop(activation);
    drop(history);
    assert_eq!(reserved(&admission), baseline);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn same_source_strided_values_foreign_issuer_and_unaware_provider_refuse() {
    struct Unaware;
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            panic!("typed convolution must refuse before any provider callback")
        }
    }
    let f = Fixture::new();
    let loaded = conv_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let baseline = reserved(&admission);
    let x = bf16([1.0; 32]);
    let h = bf16([0.0; 96]);
    let input = private(
        source.source_identity(),
        ConvInputPhase::InitialPadding,
        &x,
        &h,
    );
    let provider = Unaware;
    let unsupported =
        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
    assert!(unsupported.import_gdn_conv_input(input, 32).is_err());
    assert_eq!(reserved(&admission), baseline);
    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
    let strided = math
        .view(
            &history,
            TensorView {
                offset: 0,
                rows: 32,
                width: 1,
                row_stride: 2,
            },
        )
        .unwrap();
    let parameter = ops
        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
        .unwrap();
    let before = reserved(&admission);
    let error = math
        .causal_conv_silu_profiled(
            parameter.clone(),
            &strided,
            &history,
            CausalConvSiluContract::TorchCpuBf16Avx2,
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains("contiguous"));
    assert_eq!(reserved(&admission), before);
    let foreign_source = windows(&f, &loaded).0;
    let foreign_ops = bind_windows(foreign_source, &loaded);
    let mut foreign =
        crate::host_math::HostMath::for_window_operator_slice(&foreign_ops, None, None);
    assert!(
        foreign
            .causal_conv_silu_profiled(
                parameter,
                &activation,
                &history,
                CausalConvSiluContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    drop(foreign_ops);
    drop(strided);
    drop(activation);
    drop(history);
    assert_eq!(reserved(&admission), baseline);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn actual_role_plan_append_uses_same_builder_and_never_adds_state() {
    use crate::accelerator::program::{ActivationPrecision, PlanBuilder, PlanOperation};
    let fixture = Fixture::new();
    let loaded = conv_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let embedding = builder
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    // Synthetic embedding-produced operands solely exercise the actual source
    // roles/same-builder path; they are not claimed as trained QKV/cache inputs.
    let x = builder.view(&embedding, TensorView::vector(0, 32)).unwrap();
    let h = builder
        .view(&embedding, TensorView::vector(32, 96))
        .unwrap();
    let result = operators
        .append_gdn_convolution_silu_slice(&mut builder, 0, &x, &h)
        .unwrap();
    assert_eq!(
        builder.precision(&result.activated),
        ActivationPrecision::Bf16WidenedF32
    );
    let recorded = builder.finish_operator_slice(result.activated).unwrap();
    assert_eq!(recorded.plan().nodes().len(), 4);
    assert!(matches!(
        recorded.plan().nodes()[3].operation,
        PlanOperation::TypedCausalConvSilu { .. }
    ));
    assert!(recorded.plan().required_domains().is_empty());
    assert!(recorded.plan().proposed_domains().is_empty());
    assert_eq!(recorded.plan().frontier_delta(), 0);
    drop(recorded);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
