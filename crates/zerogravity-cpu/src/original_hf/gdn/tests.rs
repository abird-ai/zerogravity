use super::*;
fn bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn tensor<'a>(bytes: &'a [u8], shape: &'a [u64], kind: QuantKind) -> SourceTensorRef<'a> {
    SourceTensorRef {
        kind,
        shape,
        packed_bytes: bytes,
    }
}

#[test]
fn exact_fallible_output_preserves_words_and_refuses_incomplete_or_excess_results() {
    for heads in [16, 32, 48] {
        let count = heads * 128;
        let output = recurrent::collect_exact_output(
            count,
            (0..count).map(|i| Ok(f32::from_bits(0x3f800000 + i as u32))),
        )
        .unwrap();
        assert_eq!(output.len(), count);
        assert_eq!(output.capacity(), count);
        for (i, value) in output.iter().enumerate() {
            assert_eq!(value.to_bits(), 0x3f800000 + i as u32);
        }
    }
    assert!(recurrent::collect_exact_output(0, std::iter::empty()).is_err());
    assert!(recurrent::collect_exact_output(1_048_577, std::iter::empty()).is_err());
    assert!(recurrent::collect_exact_output(3, [Ok(1.), Ok(2.)]).is_err());
    assert!(recurrent::collect_exact_output(1, [Ok(1.), Ok(2.)]).is_err());
    assert!(
        matches!(recurrent::collect_exact_output(2,[Ok(1.),Err(Error::State("original failure".into()))]),Err(Error::State(s))if s=="original failure")
    );
}

#[test]
fn true_sixteen_to_fortyeight_head_mapping_and_cached_prior_remain_private() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let c = 10240;
    let s = 48 * 128 * 128;
    let mut w = vec![0.; c * 4];
    for r in w.chunks_exact_mut(4) {
        r[3] = 1.;
    }
    let w = bf16(&w);
    let a = bf16(&[0.; 48]);
    let norm = bf16(&[1.; 128]);
    let mut qkv = vec![0.; c];
    for h in 0..16 {
        qkv[h * 128 + h] = 1.;
        qkv[2048 + h * 128 + h] = 1.;
    }
    for h in 0..48 {
        qkv[4096 + h * 128] = if h < 3 { 1. } else { 0. };
    }
    let history = vec![0.; c * 3];
    let prior = vec![0.; s];
    let gate = vec![0.; 6144];
    let alpha = [0.; 48];
    let call = |phase, state: &[f32]| {
        step(
            GdnInputs {
                source,
                key_heads: 16,
                value_heads: 48,
                qkv: &qkv,
                gate: &gate,
                alpha: &alpha,
                beta: &alpha,
                history: &history,
                state,
                phase,
            },
            GdnParameters {
                convolution: tensor(&w, &[10240, 1, 4], QuantKind::Bf16),
                convolution_name: "actual.conv.weight",
                a_log: tensor(&a, &[48], QuantKind::Bf16),
                dt_bias: tensor(&a, &[48], QuantKind::Bf16),
                norm: tensor(&norm, &[128], QuantKind::Bf16),
            },
        )
    };
    let initial = call(TorchCpuGdnPhase::InitialOne, &prior).unwrap();
    for (label, output) in [
        ("convolution", &initial.convolution),
        ("activated", &initial.activated),
        ("history", &initial.history),
        ("state", &initial.state),
        ("core", &initial.core),
        ("gated", &initial.gated),
    ] {
        assert_eq!(
            output.capacity(),
            output.len(),
            "initial H48 {label} exact owned capacity"
        );
    }
    assert!(
        initial.core[..384]
            .chunks_exact(128)
            .all(|h| h[0] > 0. && h[1..].iter().all(|x| *x == 0.))
    );
    assert!(initial.core[384..].iter().all(|x| *x == 0.));
    assert_eq!(&initial.core[..128], &initial.core[128..256]);
    assert_eq!(&initial.core[..128], &initial.core[256..384]);
    assert!(initial.gated.iter().all(|v| *v == 0.));
    assert!(initial.beta.iter().all(|v| *v == 0.5));
    let before = initial.state.clone();
    let cached = call(TorchCpuGdnPhase::CachedOne, &initial.state).unwrap();
    for (label, output) in [
        ("convolution", &cached.convolution),
        ("activated", &cached.activated),
        ("history", &cached.history),
        ("state", &cached.state),
        ("core", &cached.core),
        ("gated", &cached.gated),
    ] {
        assert_eq!(
            output.capacity(),
            output.len(),
            "cached H48 {label} exact owned capacity"
        );
    }
    assert_eq!(initial.state, before);
    assert!(cached.state.iter().all(|v| v.is_finite()));
    assert_ne!(cached.state, initial.state);
    assert!(call(TorchCpuGdnPhase::InitialOne, &initial.state).is_err());
    assert!(prior.iter().all(|v| v.to_bits() == 0));
}

fn retired_fixture() -> Vec<Vec<f32>> {
    let path =
        std::env::var("ZG_RETIRED_H48_GDN_FIXTURE").expect("explicit retired original fixture");
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() < 8 * 1024 * 1024);
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "1e873892bc9441014c2b12ab8a7dd8226e3571c8903b504e79d69c0757716be3"
    );
    let mut n = 8;
    assert_eq!(&bytes[..8], b"ZGH48V1\0");
    let mut field = || {
        let len = u32::from_le_bytes(bytes[n..n + 4].try_into().unwrap()) as usize;
        n += 4;
        let v = bytes[n..n + len * 4]
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect::<Vec<_>>();
        n += len * 4;
        v
    };
    let f = (0..17).map(|_| field()).collect::<Vec<_>>();
    assert_eq!(n, bytes.len());
    f
}

fn retired_step(f: &[Vec<f32>]) -> GdnCompleted {
    retired_step_profile(
        f,
        transcendentals::StandaloneExpProfile::ScalarLibmDiagnostic,
    )
}
fn retired_step_profile(
    f: &[Vec<f32>],
    profile: transcendentals::StandaloneExpProfile,
) -> GdnCompleted {
    let conv = bf16(&f[6]);
    let a = bf16(&f[7]);
    let dt = bf16(&f[8]);
    let norm = bf16(&f[9]);
    step_observed_with_exp_profile(
        GdnInputs {
            source: SourceIdentity::for_new_allocation().unwrap(),
            key_heads: 16,
            value_heads: 48,
            qkv: &f[0],
            gate: &f[1],
            alpha: &f[2],
            beta: &f[3],
            history: &f[4],
            state: &f[5],
            phase: TorchCpuGdnPhase::CachedOne,
        },
        GdnParameters {
            convolution: tensor(&conv, &[10240, 1, 4], QuantKind::Bf16),
            convolution_name: "model.language_model.layers.0.linear_attn.conv1d.weight",
            a_log: tensor(&a, &[48], QuantKind::Bf16),
            dt_bias: tensor(&dt, &[48], QuantKind::Bf16),
            norm: tensor(&norm, &[128], QuantKind::Bf16),
        },
        profile,
        None,
    )
    .unwrap()
}

#[test]
#[ignore = "requires ROOT-accepted retired H48 fixture; not a model launch"]
fn retired_h48_reference_comparison_is_explicit_and_exact() {
    let f = retired_fixture();
    let out = retired_step(&f);
    let mut mismatched = false;
    for (label, actual, expected) in [
        ("activated", &out.activated, &f[10]),
        ("beta", &out.beta, &f[11]),
        ("g", &out.log_decay, &f[12]),
        ("decay", &out.decay, &f[13]),
        ("core", &out.core, &f[14]),
        ("state", &out.state, &f[15]),
        ("gated", &out.gated, &f[16]),
    ] {
        assert_eq!(actual.len(), expected.len());
        let differences = actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i, a.to_bits(), b.to_bits()))
            .collect::<Vec<_>>();
        eprintln!(
            "actual H48 {label}: {}/{} differing words; first={:?}",
            differences.len(),
            actual.len(),
            differences.first()
        );
        mismatched |= !differences.is_empty();
    }
    assert!(
        !mismatched,
        "actual original H48 mismatch; no numerical qualification"
    );
}

#[test]
#[ignore = "requires retired H48 fixture; observed-decay boundary is not a producer fix"]
fn retired_h48_observed_exp_boundary_closes_entire_cached_state() {
    let f = retired_fixture();
    let out = retired_step(&f);
    let words = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    assert_eq!(words(&out.activated, &f[10]), 0);
    assert_eq!(words(&out.beta, &f[11]), 0);
    assert_eq!(words(&out.log_decay, &f[12]), 0);
    let changed_exp_heads = out
        .decay
        .iter()
        .zip(&f[13])
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let state_differences = out
        .state
        .chunks_exact(128 * 128)
        .zip(f[15].chunks_exact(128 * 128))
        .map(|(a, b)| words(a, b))
        .collect::<Vec<_>>();
    for (head, count) in state_differences.iter().enumerate() {
        assert!(
            *count == 0 || changed_exp_heads.contains(&head),
            "unexplained state head {head}"
        );
    }
    let repeat = |offset: usize| {
        (0..48)
            .flat_map(|h| {
                let start = offset + (h / 3) * 128;
                out.activated[start..start + 128].iter().copied()
            })
            .collect::<Vec<_>>()
    };
    let query = repeat(0);
    let key = repeat(2048);
    // This is the original standalone recurrence API with an observed input
    // boundary. It is deliberately NOT a replacement exp producer in step().
    let cut = recurrent::cached_step_128(recurrent::CachedGdn128Inputs {
        source: SourceIdentity::for_new_allocation().unwrap(),
        heads: 48,
        query: &query,
        key: &key,
        value: &out.activated[4096..],
        beta: &out.beta,
        decay: &f[13],
        previous_state: &f[5],
    })
    .unwrap();
    eprintln!(
        "actual H48 exp heads={changed_exp_heads:?}; state differences by head={state_differences:?}"
    );
    assert_eq!(
        words(&cut.state, &f[15]),
        0,
        "observed-exp cached state cut"
    );
    assert_eq!(
        words(&cut.output_bf16, &f[14]),
        0,
        "observed-exp cached output cut"
    );
    eprintln!(
        "observed-exp cut exact state {}/{} and core {}/{}; full exp producer remains separately unqualified",
        cut.state.len(),
        f[15].len(),
        cut.output_bf16.len(),
        f[14].len()
    );
}

#[test]
fn initial_observer_borrows_existing_computation_and_error_cannot_mutate_inputs() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let c = 10240;
    let mut w = vec![0.; c * 4];
    for row in w.chunks_exact_mut(4) {
        row[3] = 1.;
    }
    let w = bf16(&w);
    let a = bf16(&[0.; 48]);
    let norm = bf16(&[1.; 128]);
    let qkv = (0..c)
        .map(|i| ((i % 11) as f32 - 5.) / 32.)
        .collect::<Vec<_>>();
    let gate = vec![0.; 6144];
    let history = vec![0.; c * 3];
    let state = vec![0.; 48 * 128 * 128];
    let alpha = [0.; 48];
    let inputs = |phase| GdnInputs {
        source,
        key_heads: 16,
        value_heads: 48,
        qkv: &qkv,
        gate: &gate,
        alpha: &alpha,
        beta: &alpha,
        history: &history,
        state: &state,
        phase,
    };
    let params = || GdnParameters {
        convolution: tensor(&w, &[10240, 1, 4], QuantKind::Bf16),
        convolution_name: "analytic.conv.weight",
        a_log: tensor(&a, &[48], QuantKind::Bf16),
        dt_bias: tensor(&a, &[48], QuantKind::Bf16),
        norm: tensor(&norm, &[128], QuantKind::Bf16),
    };
    let default = step(inputs(TorchCpuGdnPhase::InitialOne), params()).unwrap();
    let mut seen = Vec::new();
    let mut contraction_tape = Vec::new();
    let mut observer = |stage, values: &[f32]| {
        assert!(values.iter().all(|v| v.is_finite()));
        seen.push((stage, values.len()));
        if matches!(
            stage,
            InitialCoreStage::ScaledQuery
                | InitialCoreStage::NormalizedKey
                | InitialCoreStage::QueryKey
        ) {
            contraction_tape.push((stage, values.to_vec()));
        }
        Ok(())
    };
    let observed = step_observed(
        inputs(TorchCpuGdnPhase::InitialOne),
        params(),
        Some(&mut observer),
    )
    .unwrap();
    for (a, b) in [
        (&default.core, &observed.core),
        (&default.gated, &observed.gated),
        (&default.state, &observed.state),
        (&default.history, &observed.history),
        (&default.convolution, &observed.convolution),
        (&default.activated, &observed.activated),
    ] {
        assert_eq!(a, b);
    }
    assert_eq!(seen.len(), 10);
    assert_eq!(seen[7], (InitialCoreStage::QueryKey, 48));
    let values = |stage| {
        contraction_tape
            .iter()
            .find(|(s, _)| *s == stage)
            .unwrap()
            .1
            .as_slice()
    };
    let scaled = values(InitialCoreStage::ScaledQuery);
    let key = values(InitialCoreStage::NormalizedKey);
    let dots = values(InitialCoreStage::QueryKey);
    assert_eq!(
        dots,
        reduction::padded64_qk_fma128(source, scaled, key).unwrap()
    );
    let products = scaled
        .iter()
        .zip(key)
        .map(|(q, k)| q * k)
        .collect::<Vec<_>>();
    assert_ne!(
        dots,
        reduction::sum_inner_128(source, &products).unwrap(),
        "actual actor must not regress to rounded-product SumKernel"
    );
    let before = qkv.clone();
    let mut refuse = |stage, _: &[f32]| {
        if stage == InitialCoreStage::NormalizedQuery {
            Err(Error::State("borrowed observer refused".into()))
        } else {
            Ok(())
        }
    };
    assert!(
        matches!(step_observed(inputs(TorchCpuGdnPhase::InitialOne),params(),Some(&mut refuse)),Err(Error::State(s))if s=="borrowed observer refused")
    );
    assert_eq!(qkv, before);
    assert!(state.iter().all(|v| v.to_bits() == 0));
    assert!(history.iter().all(|v| v.to_bits() == 0));
    assert!(matches!(
        step_observed(
            inputs(TorchCpuGdnPhase::CachedOne),
            params(),
            Some(&mut refuse)
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
#[ignore = "requires ROOT-accepted retired H48 fixture; no model launch"]
fn retired_h48_exp_profile_closes_both_exp_calls_and_entire_cached_state() {
    let fixture = retired_fixture();
    let actual = retired_step_profile(
        &fixture,
        transcendentals::StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1,
    );
    for (label, actual, expected) in [
        ("activated", &actual.activated, &fixture[10]),
        ("beta", &actual.beta, &fixture[11]),
        ("A-exp→log-decay", &actual.log_decay, &fixture[12]),
        ("standalone-decay", &actual.decay, &fixture[13]),
        ("core", &actual.core, &fixture[14]),
        ("state", &actual.state, &fixture[15]),
        ("gated", &actual.gated, &fixture[16]),
    ] {
        assert_eq!(actual.len(), expected.len(), "{label} source extent");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{label} word{index}");
        }
    }
    // Preserve the previous boundary as a negative control, never substitute
    // observed decay into the complete producer to make the new profile pass.
    let legacy = retired_step(&fixture);
    assert_eq!(
        legacy
            .decay
            .iter()
            .zip(&fixture[13])
            .filter(|(a, e)| a.to_bits() != e.to_bits())
            .count(),
        2
    );
    assert_eq!(
        legacy
            .state
            .iter()
            .zip(&fixture[15])
            .filter(|(a, e)| a.to_bits() != e.to_bits())
            .count(),
        5539
    );
}
