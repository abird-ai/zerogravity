use super::*;
use epoch::Cursor;

#[test]
fn equal_call_metadata_does_not_replace_distinct_record_identity() -> Result<()> {
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_model_qwen35::plan::PromptCall;
    use zerogravity_model_qwen35::plan::PromptSpan;
    use zerogravity_model_qwen35::plan::PromptWorkload;
    use zerogravity_model_qwen35::plan::QwenSourceProfile;
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        let span = PromptSpan::prefill(
            source,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            &[11; 47],
        )?;
        let cold = Arc::new(PromptCall::new(profile, span.clone())?);
        let same_call = Arc::new(PromptCall::new(profile, span)?);
        assert_eq!(cold, same_call); // SAME immutable teacher span, equal call.
        require_new_record(&same_call, &cold)?;
        assert!(require_new_record(&cold.clone(), &cold).is_err());
        // Root/attempt authority is deliberately absent from this metadata test.
    }
    Ok(())
}

fn recipe() -> (Vec<ValueStorageBinding>, Vec<ValueLifetime>) {
    // v0 -> v1 -> v2 -> v3; v2 overwrites v0 only after its final reader.
    (
        vec![
            ValueStorageBinding {
                slot: 0,
                epoch: 0,
                bytes: 512,
            },
            ValueStorageBinding {
                slot: 1,
                epoch: 0,
                bytes: 512,
            },
            ValueStorageBinding {
                slot: 0,
                epoch: 1,
                bytes: 512,
            },
            ValueStorageBinding {
                slot: 2,
                epoch: 0,
                bytes: 512,
            },
        ],
        vec![
            ValueLifetime {
                producer: 0,
                last_use: 1,
            },
            ValueLifetime {
                producer: 1,
                last_use: 2,
            },
            ValueLifetime {
                producer: 2,
                last_use: 3,
            },
            ValueLifetime {
                producer: 3,
                last_use: 4,
            },
        ],
    )
}
fn step(
    c: &mut Cursor,
    b: &[ValueStorageBinding],
    l: &[ValueLifetime],
    index: usize,
) -> Result<()> {
    let input = index.checked_sub(1).map(|id| ValueId(id as u32));
    let inputs = input.as_slice();
    let outputs = [ValueId(index as u32)];
    c.before_parts(b, l, index, inputs, &outputs)?;
    c.after_parts(b, index, &outputs)
}

#[test]
fn ordered_full_producer_and_certificates_allow_only_strict_lifetime_gap() {
    let (bindings, lives) = recipe();
    let mut c = Cursor::empty(4, 3, 4);
    c.begin().unwrap();
    for i in 0..4 {
        step(&mut c, &bindings, &lives, i).unwrap();
    }
    c.finish().unwrap();
    assert!(c.begin().is_err());
}

#[test]
fn partial_operation_or_missing_certificate_cannot_unlock_next_overwrite() {
    let (b, l) = recipe();
    let mut c = Cursor::empty(4, 3, 4);
    c.begin().unwrap();
    step(&mut c, &b, &l, 0).unwrap();
    c.before_parts(&b, &l, 1, &[ValueId(0)], &[ValueId(1)])
        .unwrap();
    // A successful split-K main is not completion of the whole producer.
    assert!(
        c.before_parts(&b, &l, 2, &[ValueId(1)], &[ValueId(2)])
            .is_err()
    );
    assert!(c.after_parts(&b, 1, &[ValueId(1)]).is_err());
    assert!(c.finish().is_err());
}

#[test]
fn same_operation_last_read_and_output_cannot_alias() {
    let (mut b, l) = recipe();
    b[1].slot = 0;
    b[1].epoch = 1;
    let mut c = Cursor::empty(4, 3, 4);
    c.begin().unwrap();
    step(&mut c, &b, &l, 0).unwrap();
    assert!(step(&mut c, &b, &l, 1).is_err());
}

#[test]
fn replay_old_logical_value_after_physical_overwrite_is_rejected() {
    let (b, l) = recipe();
    let mut c = Cursor::empty(4, 3, 4);
    c.begin().unwrap();
    for i in 0..3 {
        step(&mut c, &b, &l, i).unwrap();
    }
    assert!(
        c.before_parts(&b, &l, 3, &[ValueId(0)], &[ValueId(3)])
            .is_err()
    );
}

#[test]
fn changed_lifetime_epoch_or_incomplete_graph_refuses_before_reuse() {
    for mutation in 0..3 {
        let (mut b, mut l) = recipe();
        if mutation == 0 {
            l[0].last_use = 2;
        }
        if mutation == 1 {
            b[2].epoch = 0;
        }
        if mutation == 2 {
            b[2].epoch = 2;
        }
        let mut c = Cursor::empty(4, 3, 4);
        c.begin().unwrap();
        for i in 0..2 {
            step(&mut c, &b, &l, i).unwrap();
        }
        assert!(step(&mut c, &b, &l, 2).is_err());
    }
    let (b, l) = recipe();
    let mut c = Cursor::empty(4, 3, 4);
    c.begin().unwrap();
    step(&mut c, &b, &l, 0).unwrap();
    assert!(c.finish().is_err());
}

#[test]
fn delayed_or_duplicate_node_poison_never_becomes_a_fresh_epoch() {
    let (b, l) = recipe();
    let mut c = Cursor::empty(4, 3, 4);
    c.begin().unwrap();
    assert!(
        c.before_parts(&b, &l, 1, &[ValueId(0)], &[ValueId(1)])
            .is_err()
    );
    assert!(step(&mut c, &b, &l, 0).is_err());
    assert!(c.begin().is_err());
}

#[test]
fn private_pool_guard_is_confined_to_actual_begun_node() {
    let (b, l) = recipe();
    let mut c = Cursor::empty(4, 3, 4);
    assert!(c.require_active(0).is_err());
    c.begin().unwrap();
    assert!(c.require_active(0).is_err());
    c.before_parts(&b, &l, 0, &[], &[ValueId(0)]).unwrap();
    c.require_active(0).unwrap();
    assert!(c.require_active(1).is_err());
    c.after_parts(&b, 0, &[ValueId(0)]).unwrap();
    assert!(c.require_active(0).is_err());
    for i in 1..4 {
        step(&mut c, &b, &l, i).unwrap();
    }
    c.finish().unwrap();
    c.require_finished().unwrap();
    assert!(c.require_active(3).is_err());
}
