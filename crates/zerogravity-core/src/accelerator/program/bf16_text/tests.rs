use super::*;
use crate::session_root::SessionAuthority;

fn builder(source: SourceIdentity) -> PlanBuilder {
    PlanBuilder::new_with_profile(
        source,
        64,
        (0..15).map(|i| StateSlot::Kv(KvDomain(i))).collect(),
        true,
        ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1,
    )
    .unwrap()
}
fn embedding(source: SourceIdentity) -> SourceTensor {
    SourceTensor {
        source,
        name: "actual.embedding".into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns: 1536,
        rows: 262144,
    }
}
#[test]
fn call_keeps_full_prefill_contraction_and_real_attempt_without_publication() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let authority = SessionAuthority::new(source).unwrap();
    let before = authority.binding();
    let call = Bf16TextCall::bind(
        TextCallGeometry::new(source, 262144, 64).unwrap(),
        authority.begin_advance(4).unwrap(),
        Arc::from([4906, 2915, 886, 563]),
    )
    .unwrap();
    for index in 0..4 {
        let row = call.row(index).unwrap();
        assert_eq!(row.position(), index);
        assert_eq!(
            row.pv_geometry(512).unwrap(),
            Bf16TextPvGeometry {
                query_rows: 4,
                contraction: 4,
                active: index + 1,
                width: 512
            }
        );
        assert_eq!(
            row.projection_geometry(&embedding(source))
                .unwrap()
                .query_rows,
            4
        );
        assert_eq!(row.needs_logits(), index == 3);
    }
    assert_eq!(authority.binding(), before);
    assert!(call.row(4).is_err());
    assert!(
        call.require_current(SessionAuthority::new(source).unwrap().binding())
            .is_err()
    );
    assert!(
        Bf16TextCall::bind(
            TextCallGeometry::new(source, 262144, 64).unwrap(),
            authority.begin_advance(3).unwrap(),
            Arc::from([1, 2])
        )
        .is_err()
    );
    assert!(
        Bf16TextCall::bind(
            TextCallGeometry::new(source, 262144, 64).unwrap(),
            authority.begin_advance(1).unwrap(),
            Arc::from([262144])
        )
        .is_err()
    );
}
#[test]
fn cached_geometry_is_current_root_issued_and_stale_calls_refuse() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut authority = SessionAuthority::new(source).unwrap();
    let initial = authority.begin_advance(4).unwrap();
    authority.prepare_publish(initial).unwrap().commit();
    let call = Bf16TextCall::bind(
        TextCallGeometry::new(source, 262144, 64).unwrap(),
        authority.begin_advance(1).unwrap(),
        Arc::from([886]),
    )
    .unwrap();
    assert_eq!(
        call.row(0).unwrap().pv_geometry(256).unwrap(),
        Bf16TextPvGeometry {
            query_rows: 1,
            contraction: 5,
            active: 5,
            width: 256
        }
    );
    assert!(
        Bf16TextCall::bind(
            TextCallGeometry::new(source, 262144, 64).unwrap(),
            authority.begin_advance(2).unwrap(),
            Arc::from([1, 2])
        )
        .is_err()
    );
    authority
        .prepare_publish(authority.begin_advance(1).unwrap())
        .unwrap()
        .commit();
    assert!(call.require_current(authority.binding()).is_err());
}
#[test]
fn direct_gamma_profile_does_not_widen_qwen_ordinary_rms_or_analytic_scalars() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut b = builder(source);
    let x = b
        .embed(embedding(source), PlanBuilder::inputs().token)
        .unwrap();
    let gamma = vec![0.0; 1536];
    let n = b
        .grouped_rms(
            "actual.norm",
            &x,
            SourceVector {
                source,
                name: "actual.norm.weight",
                values: &gamma,
            },
            1,
            1e-6,
        )
        .unwrap();
    assert_eq!(n.precision(), ActivationPrecision::Bf16WidenedF32);
    assert_eq!(b.constants[0].values[0].to_bits(), 0); // direct zero, never1+gamma
    assert!(
        OrdinaryRmsContract::TorchCpuBf16Avx2
            .validate(
                &SourceParameter {
                    source,
                    name: "norm".into(),
                    kind: QuantKind::Bf16,
                    shape: vec![1536],
                    bytes: 3072
                },
                1536
            )
            .is_err()
    );
    assert!(
        b.grouped_rms(
            "wrong.eps",
            &x,
            SourceVector {
                source,
                name: "actual.norm.weight",
                values: &gamma
            },
            1,
            1e-5
        )
        .is_err()
    );
    assert!(
        PlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1
        )
        .is_err()
    );
    let mut analytic = PlanBuilder::new_with_profile(
        source,
        64,
        vec![StateSlot::Kv(KvDomain(0))],
        false,
        ProgramNumericalProfile::AnalyticF32TextV1,
    )
    .unwrap();
    let a = analytic
        .embed(embedding(source), PlanBuilder::inputs().token)
        .unwrap();
    assert!(
        analytic
            .text_elementwise("scale", TextElementwise::ScaleBf16Scalar(2.0), &[&a])
            .is_err()
    );
    assert!(
        b.text_elementwise("foreign", TextElementwise::Add, &[&x, &a])
            .is_err()
    );
}
#[test]
fn numeric_argmax_preserves_signed_zero_bits_and_lowest_id() {
    assert!(!Bf16TextGreedy::prefer((4, 0), (3, 0x80000000)).unwrap());
    assert!(Bf16TextGreedy::prefer((3, 0x80000000), (4, 0)).unwrap());
    assert!(Bf16TextGreedy::prefer((9, 0x00010000), (1, 0)).unwrap());
    assert!(!Bf16TextGreedy::prefer((9, 0x3f800000), (1, 0x3f800000)).unwrap());
    assert!(Bf16TextGreedy::prefer((0, 0x7fc00000), (1, 0)).is_err());
    assert!(Bf16TextGreedy::prefer((0, 1), (1, 0)).is_err());
}
