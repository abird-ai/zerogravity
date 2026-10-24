//! Analytic complete source-window attention/tail, not trained27B proof.
use super::*;
use crate::operator_slice::pure::PureSliceInput;

fn fixture_source(f: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(false)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["head_dim"] = json!(256);
    cfg["text_config"]["intermediate_size"] = json!(16);
    cfg["text_config"]["partial_rotary_factor"] = json!(0.25);
    cfg["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.25);
    cfg["text_config"]["rope_parameters"]["rope_theta"] = json!(10_000_000.0);
    source_with_config(
        f,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, bits| {
            if name.ends_with("input_layernorm.weight") {
                0x3e80 // Deliberate raw gamma1/4, matching the independent oracle.
            } else if name.ends_with(".self_attn.q_proj.weight")
                || name.ends_with(".self_attn.k_proj.weight")
            {
                0
            } else if name.ends_with(".self_attn.v_proj.weight")
                || name.ends_with("mlp.gate_proj.weight")
                || name.ends_with("mlp.up_proj.weight")
            {
                0x3b80
            } else if name.ends_with(".self_attn.o_proj.weight") {
                0x3b00
            } else if name.ends_with("mlp.down_proj.weight") {
                0x3d80
            } else {
                bits
            }
        },
        |_| {},
    )
}
fn input<'a>(s: &OriginalWindowOperators, bytes: &'a [u8]) -> PureSliceInput<'a> {
    PureSliceInput::from_bf16_bytes(
        s.source().source_identity(),
        s.program_numerical_profile(),
        bytes,
    )
    .unwrap()
}

#[test]
fn entire_attention_and_tail_compute_causal_prefix_and_full_vocabulary() {
    let f = Fixture::new();
    let loaded = fixture_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let raw = 0x3f80u16.to_le_bytes().repeat(256);
    let a = operators
        .execute_attention_output_tail(input(&operators, &raw), None, None)
        .unwrap();
    assert_eq!(a.layer(), 3);
    assert_eq!(a.prefix().tokens(), 1);
    assert_eq!(a.stage("normalized_input").unwrap(), &[1.25; 256]);
    assert_eq!(a.stage("context").unwrap(), &[1.25; 512]);
    assert_eq!(a.stage("gated_attention").unwrap(), &[0.625; 512]);
    assert_eq!(a.stage("attention_branch").unwrap(), &[0.625; 256]);
    assert_eq!(a.stage("post_attention").unwrap(), &[1.625; 256]);
    assert_eq!(a.stage("ffn_gated").unwrap(), &[1.21875; 16]);
    assert_eq!(a.full_vocabulary().unwrap(), &[-320.0; 16]);
    let prior = a.prefix().clone();
    let before = prior.values().unwrap().to_vec();
    let b = operators
        .execute_attention_output_tail(input(&operators, &raw), Some(&prior), None)
        .unwrap();
    assert_eq!(b.prefix().tokens(), 2);
    assert_eq!(b.stage("probabilities").unwrap(), &[0.5; 4]);
    assert_eq!(prior.values().unwrap(), before);
    assert_eq!(prior.tokens(), 1);
    assert_eq!(b.full_vocabulary().unwrap(), &[-320.0; 16]);
    drop(a);
    drop(prior);
    drop(operators);
    drop(source);
    assert!(reserved(&admission) > 0);
    assert_eq!(b.prefix().values().unwrap(), &[1.25; 512]);
    assert_eq!(b.full_vocabulary().unwrap(), &[-320.0; 16]);
    drop(b);
    assert_eq!(reserved(&admission), 0);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
}

#[test]
fn wrong_source_cancel_and_real_host_pressure_never_modify_prior() {
    let f = Fixture::new();
    let loaded = fixture_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let raw = 0x3f80u16.to_le_bytes().repeat(256);
    let baseline = reserved(&admission);
    assert!(
        operators
            .execute_attention_output_tail(
                input(&operators, &raw),
                None,
                Some(&AtomicBool::new(true))
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), baseline);
    let out = operators
        .execute_attention_output_tail(input(&operators, &raw), None, None)
        .unwrap();
    let prior = out.prefix().clone();
    let expected = prior.values().unwrap().to_vec();
    drop(out);
    let (other, other_admission) = windows(&f, &loaded);
    let foreign = bind_windows(other.clone(), &loaded);
    let before = reserved(&other_admission);
    assert!(
        foreign
            .execute_attention_output_tail(input(&foreign, &raw), Some(&prior), None)
            .is_err()
    );
    assert_eq!(reserved(&other_admission), before);
    let mut tickets = Vec::new();
    loop {
        let n = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if n == 0 {
            break;
        }
        tickets.push(
            source
                .reserve_workspace(n.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    assert!(
        operators
            .execute_attention_output_tail(input(&operators, &raw), Some(&prior), None)
            .is_err()
    );
    assert_eq!(prior.values().unwrap(), expected);
    assert_eq!(prior.tokens(), 1);
    drop(tickets);
    drop(prior);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
    drop(foreign);
    drop(other);
    assert_eq!(reserved(&other_admission), 0);
}

#[test]
fn source_geometry_is_checked_before_import_and_old_model_execution_stays_refused() {
    let f = Fixture::new();
    let loaded = fixture_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    assert!(operators.attention_tail_spec().is_ok());
    let (model, _, _) = bind(loaded);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, false).is_err());
    let raw = 0x3f80u16.to_le_bytes().repeat(255);
    let before = reserved(&admission);
    assert!(
        operators
            .execute_attention_output_tail(input(&operators, &raw), None, None)
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
}
