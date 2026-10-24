//! Real31B geometry recording and bounded synthetic execution of that schedule.
use super::*;

#[test]
fn full_dense31_plan_reuses_family_schedule_and_sixty_owning_domains() {
    let config = Gemma4Config::from_json(include_bytes!(
        "../../tests/fixtures/official-31b-it-config.json"
    ))
    .unwrap();
    let fixture = Fixture::geometry(Geometry::from_config(&config).unwrap(), false);
    let domains = fixture.geometry.domains();
    assert_eq!(domains, (0..60).map(KvDomain).collect::<Vec<_>>());
    assert!(fixture.matrix(Gemma4MatrixRole::Value(5)).is_err());
    assert!(fixture.matrix(Gemma4MatrixRole::Value(4)).is_ok());
    let mut math = PlanBuilder::new_with_profile(
        fixture.source,
        8,
        domains.into_iter().map(StateSlot::Kv).collect(),
        true,
        ProgramNumericalProfile::AnalyticF32TextV1,
    )
    .unwrap();
    let logits = forward(&fixture, &mut math, PlanBuilder::inputs(), true).unwrap();
    let plan = math.finish(logits).unwrap();
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::AppendKv))
            .count(),
        60
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::TextWindowQk { .. }))
            .count(),
        60
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TextWindowQk {
                    window: Some(1024),
                    ..
                }
            ))
            .count(),
        50
    );
    assert!(
        !plan
            .nodes()
            .iter()
            .any(|n| matches!(n.operation, PlanOperation::TextMoe { .. })
                || n.name.contains(".ple."))
    );
    assert!(plan.nodes().iter().any(|n|matches!(&n.operation,PlanOperation::Project{tensor} if tensor.name.ends_with("layers.59.mlp.down_proj.weight")&&tensor.columns==21504&&tensor.rows==5376)));
    assert!(plan.nodes().iter().any(|n|matches!(&n.operation,PlanOperation::Project{tensor} if tensor.name=="model.language_model.embed_tokens.weight"&&tensor.columns==5376&&tensor.rows==262144)));
    assert_eq!(
        plan.numerical_profile(),
        ProgramNumericalProfile::AnalyticF32TextV1
    );
}

fn tiny_dense31() -> Fixture {
    let config = Gemma4Config::from_json(include_bytes!(
        "../../tests/fixtures/official-31b-it-config.json"
    ))
    .unwrap();
    let mut g = Geometry::from_config(&config).unwrap();
    g.hidden = 4;
    g.vocab = 3;
    g.max_position = 8;
    g.window = 2;
    g.softcap = 3.0;
    for layer in &mut g.layers {
        layer.head_dim = 4;
        layer.query_heads = 2;
        layer.kv_heads = if layer.key_equals_value { 1 } else { 2 };
        layer.rope_rotate = if layer.key_equals_value { 2 } else { 4 };
        layer.feed_forward_width = 8;
    }
    let mut f = Fixture::geometry(g, true);
    for values in f.rows.values_mut() {
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.04;
        }
    }
    f.rows.insert(
        "model.language_model.embed_tokens.weight".into(),
        vec![1., 2., 3., 4., -2., 1., 0.5, -1., 0.5, -0.25, 1., 2.],
    );
    f
}

#[test]
fn dense31_nonzero_prefill_and_continuation_match_across_independent_roots() {
    let f = tiny_dense31();
    let mut a = ReferenceTextState::new(&f, &f.geometry.domains(), 8).unwrap();
    let mut b = ReferenceTextState::new(&f, &f.geometry.domains(), 8).unwrap();
    let mut root_a = SessionAuthority::new(f.source).unwrap();
    let mut root_b = SessionAuthority::new(f.source).unwrap();
    assert_ne!(root_a.binding().session(), root_b.binding().session());
    original::evaluate(&f, &f, &mut a, &mut root_a, &[0], false, None).unwrap();
    let split = original::evaluate(&f, &f, &mut a, &mut root_a, &[1, 2], true, None)
        .unwrap()
        .unwrap();
    let batch = original::evaluate(&f, &f, &mut b, &mut root_b, &[0, 1, 2], true, None)
        .unwrap()
        .unwrap();
    assert_eq!(split.as_slice().unwrap(), batch.as_slice().unwrap());
    assert!(split.as_slice().unwrap().iter().any(|x| x.abs() > 0.01));
    assert_eq!((root_a.binding().root(), root_b.binding().root()), (2, 1));
    for domain in f.geometry.domains() {
        assert_eq!(a.domain_tokens(domain).unwrap(), 3);
        assert_eq!(b.domain_tokens(domain).unwrap(), 3);
    }
    let cached_a = original::evaluate(&f, &f, &mut a, &mut root_a, &[1], true, None)
        .unwrap()
        .unwrap();
    let cached_b = original::evaluate(&f, &f, &mut b, &mut root_b, &[1], true, None)
        .unwrap()
        .unwrap();
    assert_eq!(cached_a.as_slice().unwrap(), cached_b.as_slice().unwrap());
}

#[test]
fn dense31_late_cancel_head_failure_and_retry_preserve_actual_current_root() {
    let f = tiny_dense31();
    let mut state = ReferenceTextState::new(&f, &f.geometry.domains(), 8).unwrap();
    let mut root = SessionAuthority::new(f.source).unwrap();
    original::evaluate(&f, &f, &mut state, &mut root, &[0], false, None).unwrap();
    let before = root.binding();
    let flag = AtomicBool::new(false);
    f.cancel_in_dense.store(true, Ordering::Release);
    assert!(original::evaluate(&f, &f, &mut state, &mut root, &[1, 2], true, Some(&flag)).is_err());
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(root.binding(), before);
    for domain in f.geometry.domains() {
        assert_eq!(state.domain_tokens(domain).unwrap(), 1);
    }
    f.cancel_in_dense.store(false, Ordering::Release);
    f.fail_head.store(true, Ordering::Release);
    assert!(original::evaluate(&f, &f, &mut state, &mut root, &[1, 2], true, None).is_err());
    assert_eq!(root.binding(), before);
    f.fail_head.store(false, Ordering::Release);
    let retried = original::evaluate(&f, &f, &mut state, &mut root, &[1, 2], true, None)
        .unwrap()
        .unwrap();
    let mut fresh = ReferenceTextState::new(&f, &f.geometry.domains(), 8).unwrap();
    let mut fresh_root = SessionAuthority::new(f.source).unwrap();
    let complete = original::evaluate(&f, &f, &mut fresh, &mut fresh_root, &[0, 1, 2], true, None)
        .unwrap()
        .unwrap();
    assert_eq!(retried.as_slice().unwrap(), complete.as_slice().unwrap());
    assert_eq!((root.binding().frontier(), state.frontier()), (3, 3));
}
