//! Analytic two-layer source; never a trained64-layer qualification.
use super::*;
use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
use std::sync::atomic::Ordering;

fn source(f: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(false)).unwrap();
    let c = &mut cfg["text_config"];
    c["num_hidden_layers"] = json!(2);
    c["full_attention_interval"] = json!(2);
    c["layer_types"] = json!(["linear_attention", "full_attention"]);
    c["hidden_size"] = json!(256);
    c["head_dim"] = json!(256);
    c["intermediate_size"] = json!(16);
    c["linear_num_key_heads"] = json!(1);
    c["linear_num_value_heads"] = json!(16);
    c["linear_key_head_dim"] = json!(128);
    c["linear_value_head_dim"] = json!(128);
    c["partial_rotary_factor"] = json!(0.25);
    c["rope_parameters"]["partial_rotary_factor"] = json!(0.25);
    c["rope_parameters"]["rope_theta"] = json!(10_000_000.);
    source_with_config(
        f,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, bits| {
            if name == "model.language_model.embed_tokens.weight"
                || name == "lm_head.weight"
                || name.ends_with("input_layernorm.weight")
            {
                bits
            } else {
                0
            }
        },
        |_| {},
    )
}

#[test]
fn completed_boundary_trace_uses_original_loop_and_refusal_never_publishes() {
    use crate::source::binding::WindowDiagnosticBoundary as B;
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, host) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut session = source.diagnostic_session(2, None).unwrap();
    let initial = session.binding();
    let baseline = reserved(&host);
    assert!(
        session
            .step_traced_with(11, None, |r| {
                assert_eq!(r.base(), initial);
                assert_eq!(r.boundary(B::Embedding)?.len(), 256);
                assert_eq!(r.boundary(B::Layer(0))?.len(), 256);
                assert_eq!(r.boundary(B::Layer(1))?.len(), 256);
                assert_eq!(r.boundary(B::FinalNorm)?.len(), 256);
                assert!(r.boundary(B::Layer(2)).is_err());
                assert!(r.prior_state(StateSlot::Kv(KvDomain(1)))?[0].is_empty());
                assert_eq!(r.state(StateSlot::Kv(KvDomain(1)))?[0].len(), 256);
                Err(Error::State("deliberate completed trace refusal".into()))
            })
            .is_err()
    );
    assert_eq!(session.binding(), initial);
    assert_eq!(reserved(&host), baseline);
    let first = session.step_traced_with(11, None, |_| Ok(())).unwrap();
    let second = session
        .step_with(13, None, |r| {
            assert!(r.boundary(B::Embedding).is_err());
            assert_eq!(r.prior_state(StateSlot::Kv(KvDomain(1)))?[0].len(), 256);
            assert_eq!(r.state(StateSlot::Kv(KvDomain(1)))?[0].len(), 512);
            Ok(())
        })
        .unwrap();
    assert_eq!(session.binding().frontier(), 2);
    assert_eq!(first.logits().unwrap(), second.logits().unwrap());
    drop((session, source, raw, first, second));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn synchronous_gdn_keeps_actual_outputs_after_transient_scratch_is_released() {
    use crate::accelerator::program::{
        ActivationPrecision, TorchCpuGdnInputs, TorchCpuGdnPhase, TorchCpuGdnValues,
    };
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, host) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut math = crate::host_math::HostMath::for_window_operator_slice(&source, None, None);
    let b = ActivationPrecision::Bf16WidenedF32;
    let qkv = math.window_zeros(2304, b).unwrap();
    let gate = math.window_zeros(2048, b).unwrap();
    let alpha = math.window_zeros(16, b).unwrap();
    let history = math.window_zeros(2304 * 3, b).unwrap();
    let state = math
        .window_zeros(16 * 128 * 128, ActivationPrecision::F32)
        .unwrap();
    let before = reserved(&host);
    let done = math
        .torch_cpu_bf16_gdn(
            &source.gdn_contract(0).unwrap(),
            TorchCpuGdnInputs {
                qkv: &qkv,
                gate: &gate,
                alpha: &alpha,
                beta: &alpha,
                history: &history,
                state: &state,
                phase: TorchCpuGdnPhase::InitialOne,
            },
        )
        .unwrap();
    let output_bytes = (2304 * 5 + 16 * 128 * 128 + 2048 * 2) * 4;
    assert_eq!(reserved(&host), before + output_bytes as u64);
    let TorchCpuGdnValues {
        convolution,
        activated,
        history: new_history,
        state: new_state,
        core,
        gated,
    } = done;
    drop((convolution, activated, core, gated));
    assert_eq!(reserved(&host), before + output_bytes as u64);
    assert_eq!(new_history.as_slice().unwrap().len(), 2304 * 3);
    drop(new_history);
    assert_eq!(reserved(&host), before + output_bytes as u64);
    drop(new_state);
    assert_eq!(reserved(&host), before);
    drop((qkv, gate, alpha, history, state, math));
    drop((source, raw));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn connected_full_schedule_initial_cached_and_independent_original_roots() {
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, admission) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let (normal, _, _) = bind(loaded);
    assert!(normal.new_session().is_err());
    assert!(normal.forward_plan(3, true, true).is_err());
    assert_eq!(normal.initialized_cpu_state_bytes(), 0);
    let mut a = source.diagnostic_session(3, None).unwrap();
    let mut b = source.diagnostic_session(3, None).unwrap();
    assert_eq!(a.domain_count(), 3);
    let initial = a.binding();
    let out = a.step(11, None).unwrap();
    assert_ne!(a.binding(), initial);
    assert_eq!(a.binding().frontier(), 1);
    assert_eq!(out.logits().unwrap(), &[-320.; 16]);
    assert!(!out.trained_numerical_qualified());
    assert!(!a.trained_numerical_qualified());
    assert_eq!(
        a.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0].len(),
        2304 * 3
    );
    assert_eq!(
        a.state(StateSlot::Recurrent(RecurrentDomain(0))).unwrap()[0].len(),
        16 * 128 * 128
    );
    assert_eq!(
        a.state(StateSlot::Kv(KvDomain(1)))
            .unwrap()
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>(),
        [256, 256]
    );
    let out2 = a.step(13, None).unwrap();
    let same = b.step(11, None).unwrap();
    assert_eq!(out2.logits().unwrap(), same.logits().unwrap());
    assert_ne!(a.binding().session(), b.binding().session());
    assert_eq!(a.binding().frontier(), 2);
    assert_eq!(b.binding().frontier(), 1);
    assert_eq!(a.state(StateSlot::Kv(KvDomain(1))).unwrap()[0].len(), 512);
    assert_eq!(b.state(StateSlot::Kv(KvDomain(1))).unwrap()[0].len(), 256);
    assert!(a.state(StateSlot::Kv(KvDomain(0))).is_err());
    drop(a);
    drop(b);
    drop(out);
    drop(same);
    drop(source);
    drop(raw);
    assert!(reserved(&admission) > 0);
    assert_eq!(out2.logits().unwrap(), &[-320.; 16]);
    drop(out2);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn complete_cpu_hook_refuses_foreign_or_swapped_original_parameter_issuer_before_math() {
    use crate::accelerator::program::{ActivationPrecision, TorchCpuGdnInputs, TorchCpuGdnPhase};
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, admission) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let (other, other_admission) = windows(&f, &loaded);
    let foreign = bind_windows(other.clone(), &loaded);
    let mut math = crate::host_math::HostMath::for_window_operator_slice(&source, None, None);
    let b = ActivationPrecision::Bf16WidenedF32;
    let q = math.window_zeros(2304, b).unwrap();
    let z = math.window_zeros(2048, b).unwrap();
    let a = math.window_zeros(16, b).unwrap();
    let h = math.window_zeros(2304 * 3, b).unwrap();
    let s = math
        .window_zeros(16 * 128 * 128, ActivationPrecision::F32)
        .unwrap();
    let before = reserved(&admission);
    let c = source.gdn_contract(0).unwrap();
    let mut swapped = c.clone();
    swapped.dt_bias = swapped.a_log.clone();
    for contract in [foreign.gdn_contract(0).unwrap(), swapped] {
        assert!(
            math.torch_cpu_bf16_gdn(
                &contract,
                TorchCpuGdnInputs {
                    qkv: &q,
                    gate: &z,
                    alpha: &a,
                    beta: &a,
                    history: &h,
                    state: &s,
                    phase: TorchCpuGdnPhase::InitialOne
                }
            )
            .is_err()
        );
        assert_eq!(reserved(&admission), before);
    }
    drop((q, z, a, h, s));
    drop((source, raw, other, foreign));
    assert_eq!(reserved(&admission), 0);
    assert_eq!(reserved(&other_admission), 0);
}

#[test]
fn actual_complete_cpu_validation_cancel_and_pressure_never_publish_partial_state() {
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, admission) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut s = source.diagnostic_session(2, None).unwrap();
    let before = s.binding();
    let baseline = reserved(&admission);
    let prior = s.state(StateSlot::Recurrent(RecurrentDomain(0))).unwrap()[0].to_vec();
    assert!(
        s.step_with(11, None, |r| {
            assert_eq!(r.base(), before);
            assert_eq!(r.logits()?, &[-320.; 16]);
            assert_eq!(r.state(StateSlot::Kv(KvDomain(1)))?[0].len(), 256);
            Err(Error::State(
                "deliberate completed diagnostic validation refusal".into(),
            ))
        })
        .is_err()
    );
    assert_eq!(s.binding(), before);
    assert_eq!(reserved(&admission), baseline);
    assert_eq!(
        s.state(StateSlot::Recurrent(RecurrentDomain(0))).unwrap()[0],
        prior
    );
    let cancelled = AtomicBool::new(false);
    assert!(
        s.step_with(11, Some(&cancelled), |_| {
            cancelled.store(true, Ordering::Release);
            Ok(())
        })
        .is_err()
    );
    assert_eq!(s.binding(), before);
    assert_eq!(reserved(&admission), baseline);
    assert!(s.step(11, Some(&cancelled)).is_err());
    assert_eq!(s.binding(), before);
    let mut pressure = Vec::new();
    while let Ok(t) = source.source().reserve_workspace(1024 * 1024) {
        pressure.push(t);
    }
    assert!(s.step(11, None).is_err());
    assert_eq!(s.binding(), before);
    drop(pressure);
    assert_eq!(reserved(&admission), baseline);
    let out = s.step(11, None).unwrap();
    assert_eq!(out.logits().unwrap(), &[-320.; 16]);
    assert!(s.step(16, None).is_err());
    assert_eq!(s.binding().frontier(), 1);
    drop(out);
    drop(s);
    drop(source);
    drop(raw);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn layer9_trace_cold_refuses_unrelated_source_without_payload_or_root_change() {
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, host) = windows(&f, &loaded);
    let operators = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut session = operators.diagnostic_session(2, None).unwrap();
    let root = session.binding();
    let before = reserved(&host);
    assert!(matches!(
        session.step_layer9_traced_with(11, None, |_| panic!("wrong source callback")),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(session.binding(), root);
    assert_eq!(reserved(&host), before);
    drop((session, operators, raw));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn initial_inner_prefix_refuses_wrong_actual_source_without_claim_or_allocation() {
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, host) = windows(&f, &loaded);
    let ops = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut session = ops.diagnostic_session(1, None).unwrap();
    let root = session.binding();
    let baseline = reserved(&host);
    assert!(matches!(
        session.observe_initial_core_with(11, None, |_| panic!("wrong source observer")),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(session.binding(), root);
    assert_eq!(reserved(&host), baseline);
    drop((session, ops, raw));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn initial_tape_reserves_before_copy_and_retains_same_source_until_drop() {
    use crate::source::binding::{InitialCoreStage as S, InitialCoreTape};
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, host) = windows(&f, &loaded);
    let ops = Arc::new(bind_windows(raw.clone(), &loaded));
    let before = reserved(&host);
    let tape = InitialCoreTape::prepare(&ops).unwrap();
    assert_eq!(reserved(&host), before + 1024 * 1024);
    {
        let mut t = tape.lock().unwrap();
        t.require_source(&ops).unwrap();
        assert!(t.capture(S::QueryKey, &[0.; 47]).is_err());
        assert!(t.capture(S::QueryKey, &[f32::NAN; 48]).is_err());
        t.capture(S::QueryKey, &[0.; 48]).unwrap();
        assert_eq!(t.values(S::QueryKey).unwrap().len(), 48);
        assert!(t.capture(S::QueryKey, &[0.; 48]).is_err());
        assert!(t.complete().is_err());
    }
    drop((ops, raw));
    assert!(reserved(&host) > 0);
    drop(tape);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn full_four_frame_session_preserves_same_math_and_actual_prior_root() {
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, admission) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut full = source.diagnostic_full_convolution_session(4, None).unwrap();
    let mut minimal = source.diagnostic_session(4, None).unwrap();
    assert_eq!(full.original_convolution_cache_frames(), 4);
    assert_eq!(
        full.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0].len(),
        2304 * 4
    );
    let mut previous = full.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0].to_vec();
    for (ordinal, token) in [11, 13, 11, 13].into_iter().enumerate() {
        let a = full
            .step_with(token, None, |read| {
                assert_eq!(read.base().frontier(), ordinal as u64);
                assert_eq!(
                    read.prior_state(StateSlot::Conv(ConvDomain(0)))?[0],
                    previous
                );
                assert_eq!(
                    read.state(StateSlot::Conv(ConvDomain(0)))?[0].len(),
                    2304 * 4
                );
                Ok(())
            })
            .unwrap();
        let b = minimal.step(token, None).unwrap();
        assert_eq!(a.logits().unwrap(), b.logits().unwrap());
        assert_eq!(
            full.state(StateSlot::Recurrent(RecurrentDomain(0)))
                .unwrap(),
            minimal
                .state(StateSlot::Recurrent(RecurrentDomain(0)))
                .unwrap()
        );
        assert_eq!(
            full.state(StateSlot::Kv(KvDomain(1))).unwrap(),
            minimal.state(StateSlot::Kv(KvDomain(1))).unwrap()
        );
        previous = full.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0].to_vec();
        let actual_last_three = previous
            .chunks_exact(4)
            .flat_map(|frames| frames[1..].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(
            actual_last_three,
            minimal.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0]
        );
        assert_ne!(a.binding().session(), b.binding().session());
        assert!(!a.trained_numerical_qualified());
    }
    assert!(full.step(11, None).is_err());
    drop((full, minimal, source, raw));
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn full_four_frame_completed_refusal_and_late_cancel_keep_all_state_then_retry() {
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, admission) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let mut session = source.diagnostic_full_convolution_session(2, None).unwrap();
    let mut survivor = source.diagnostic_full_convolution_session(2, None).unwrap();
    let initial = session.binding();
    let original = session.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0].to_vec();
    let baseline = reserved(&admission);
    assert!(
        session
            .step_with(11, None, |read| {
                assert_eq!(
                    read.state(StateSlot::Conv(ConvDomain(0)))?[0].len(),
                    2304 * 4
                );
                Err(Error::State("actual full-state validation refusal".into()))
            })
            .is_err()
    );
    assert_eq!(session.binding(), initial);
    assert_eq!(
        session.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0],
        original
    );
    assert_eq!(reserved(&admission), baseline);
    let cancel = AtomicBool::new(false);
    assert!(
        session
            .step_with(11, Some(&cancel), |_| {
                cancel.store(true, Ordering::Release);
                Ok(())
            })
            .is_err()
    );
    assert_eq!(session.binding(), initial);
    assert_eq!(
        session.state(StateSlot::Conv(ConvDomain(0))).unwrap()[0],
        original
    );
    assert_eq!(reserved(&admission), baseline);
    assert_eq!(survivor.binding().frontier(), 0);
    cancel.store(false, Ordering::Release);
    let result = session.step(11, Some(&cancel)).unwrap();
    let other = survivor.step(11, None).unwrap();
    assert_eq!(result.logits().unwrap(), other.logits().unwrap());
    assert_ne!(result.binding().session(), other.binding().session());
    drop((session, survivor, source, raw, other));
    assert!(reserved(&admission) > 0);
    drop(result);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn full_four_frame_transport_rejects_foreign_issuer_and_wrong_precision() {
    use crate::accelerator::program::ActivationPrecision;
    let f = Fixture::new();
    let loaded = source(&f);
    let (raw, admission) = windows(&f, &loaded);
    let source = Arc::new(bind_windows(raw.clone(), &loaded));
    let f2 = Fixture::new();
    let loaded2 = self::source(&f2);
    let (raw2, admission2) = windows(&f2, &loaded2);
    let source2 = Arc::new(bind_windows(raw2.clone(), &loaded2));
    let math = crate::host_math::HostMath::for_window_operator_slice(&source, None, None);
    let foreign = crate::host_math::HostMath::for_window_operator_slice(&source2, None, None);
    let b = ActivationPrecision::Bf16WidenedF32;
    let full = math.window_zeros(2304 * 4, b).unwrap();
    let minimal = math.window_zeros(2304 * 3, b).unwrap();
    let input = math.window_zeros(2304, b).unwrap();
    let wrong = math.window_zeros(2304, ActivationPrecision::F32).unwrap();
    let other = foreign.window_zeros(2304, b).unwrap();
    let before = reserved(&admission);
    assert!(
        math.window_full_convolution_cache(&full, &minimal, &wrong)
            .is_err()
    );
    assert!(
        math.window_full_convolution_cache(&full, &minimal, &other)
            .is_err()
    );
    assert!(
        math.window_full_convolution_cache(&minimal, &full, &input)
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    drop((full, minimal, input, wrong, other, math, foreign));
    drop((source, source2, raw, raw2));
    assert_eq!(reserved(&admission), 0);
    assert_eq!(reserved(&admission2), 0);
}
