//! Host metadata guards only; no service, physical work, Ready or GPU fixture.
use super::*;
use zerogravity_core::device_events::DeviceEventLedger;

fn snapshot_fixture() -> Snapshot {
    let mut ledger = DeviceEventLedger::new(1, 32).unwrap();
    let ids = (0..4)
        .map(|_| ledger.admit(4, BufferRole::Scratch).unwrap())
        .collect::<Vec<_>>();
    Snapshot {
        leases: ids.iter().copied().map(|id| (id, 4)).collect(),
        descriptors: vec![(ids[0], 0, 4), (ids[1], 0, 4)],
        uninitialized: ids[1..].iter().copied().collect(),
        future: ids[2..].iter().copied().collect(),
        state_buffers: 2,
        values: 1,
        weights: 1,
        native: FixedPromptNativeRetention {
            images: 2,
            kernels: 3,
            native_charge_bytes: 16,
            image_admission_bytes: 32,
        },
        resident: 16,
        uploaded: 4,
    }
}

#[test]
fn retained_metadata_requires_every_actual_lease_and_extent() {
    let expected = snapshot_fixture();
    expected.require_same(&expected.clone()).unwrap();
    let id = *expected.leases.keys().next().unwrap();
    let mut missing = expected.clone();
    missing.leases.remove(&id);
    assert!(expected.require_same(&missing).is_err());
    let mut resized = expected.clone();
    *resized.leases.get_mut(&id).unwrap() += 1;
    assert!(expected.require_same(&resized).is_err());
}

#[test]
fn retained_cohort_preserves_descriptor_identity_order_and_view() {
    let expected = snapshot_fixture();
    let mut reordered = expected.clone();
    reordered.descriptors.reverse();
    assert!(expected.require_same(&reordered).is_err());
    let mut shifted = expected.clone();
    shifted.descriptors[0].1 += 1;
    assert!(expected.require_same(&shifted).is_err());
    let mut shortened = expected.clone();
    shortened.descriptors[0].2 -= 1;
    assert!(expected.require_same(&shortened).is_err());
    let mut omitted = expected.clone();
    omitted.descriptors.pop();
    assert!(expected.require_same(&omitted).is_err());
}

#[test]
fn unknown_cannot_promote_one_prepared_future_lease() {
    let expected = snapshot_fixture();
    require_future_uninitialized(&expected.future, &expected.uninitialized).unwrap();
    let mut promoted = expected.clone();
    let id = *promoted.future.iter().next().unwrap();
    promoted.uninitialized.remove(&id);
    assert!(expected.require_same(&promoted).is_err());
    assert!(require_future_uninitialized(&promoted.future, &promoted.uninitialized).is_err());
    assert!(require_future_uninitialized(&BTreeSet::new(), &expected.uninitialized).is_err());
    let mut newly_uninitialized = expected.clone();
    let id = *expected.leases.keys().next().unwrap();
    newly_uninitialized.uninitialized.insert(id);
    assert!(expected.require_same(&newly_uninitialized).is_err());
}

#[test]
fn retained_native_and_source_accounting_cannot_shrink_or_change() {
    let expected = snapshot_fixture();
    for field in 0..9 {
        let mut changed = expected.clone();
        match field {
            0 => changed.native.images += 1,
            1 => changed.native.kernels += 1,
            2 => changed.native.native_charge_bytes += 1,
            3 => changed.native.image_admission_bytes += 1,
            4 => changed.state_buffers += 1,
            5 => changed.values += 1,
            6 => changed.weights += 1,
            7 => changed.resident += 1,
            _ => changed.uploaded += 1,
        }
        assert!(expected.require_same(&changed).is_err());
    }
}

#[test]
fn only_exact_original_cut_errors_are_diagnostic_successes() {
    use FixedPromptModelPrefixFault as F;
    let unknown = Error::State(
        "deliberate full24 unknown after actual source Conv and canonical copy".into(),
    );
    let cancelled = Error::State("closed original cubin qualification cancelled".into());
    assert!(intended_error(&unknown, F::UnknownAfterConvolution));
    assert!(intended_error(&cancelled, F::CancelBeforeRecurrent));
    assert!(!intended_error(&unknown, F::CancelBeforeRecurrent));
    assert!(!intended_error(&cancelled, F::UnknownAfterConvolution));
    assert!(!intended_error(&unknown, F::AdmissionPressure));
    assert!(!intended_error(
        &Error::State("unrelated failure after cut".into()),
        F::UnknownAfterConvolution
    ));
    assert!(!intended_error(
        &Error::Limit("closed original cubin qualification cancelled".into()),
        F::CancelBeforeRecurrent
    ));
}

#[test]
fn existing_armer_modes_and_pressure_are_separate() {
    use FixedPromptModelPrefixFault as F;
    for (fault, armer, pressure) in [
        (F::AdmissionPressure, 0, true),
        (F::UnknownAfterConvolution, 1, false),
        (F::CancelBeforeRecurrent, 2, false),
        (F::UnknownAfterQkvMain, 3, false),
        (F::CancelBeforeQkvReducer, 4, false),
    ] {
        let mode = Mode::Fault {
            fault,
            probe: Arc::new(Probe::default()),
        };
        assert_eq!(mode.fault_mode(), armer);
        assert_eq!(mode.is_pressure(), pressure);
    }
    assert_eq!(Mode::Compare.fault_mode(), 0);
    assert!(!Mode::Compare.is_pressure());
}

#[test]
fn new_partial_faults_cold_refuse_old_cut_and_require_exact_intended_errors() {
    use FixedPromptModelPrefixFault as F;
    for fault in [F::UnknownAfterQkvMain, F::CancelBeforeQkvReducer] {
        assert!(require_fault_cut(FixedPromptPrefixKind::FirstThree, fault).is_err());
        require_fault_cut(FixedPromptPrefixKind::ThroughAttentionQkv, fault).unwrap();
    }
    for fault in [
        F::AdmissionPressure,
        F::UnknownAfterConvolution,
        F::CancelBeforeRecurrent,
    ] {
        require_fault_cut(FixedPromptPrefixKind::FirstThree, fault).unwrap();
    }
    let unknown = Error::State("deliberate canonical47 unknown after actual QKV main".into());
    let cancel = Error::State("closed original cubin qualification cancelled".into());
    assert!(intended_error(&unknown, F::UnknownAfterQkvMain));
    assert!(!intended_error(&unknown, F::CancelBeforeQkvReducer));
    assert!(intended_error(&cancel, F::CancelBeforeQkvReducer));
    assert!(!intended_error(&cancel, F::UnknownAfterQkvMain));
    assert!(!intended_error(
        &Error::State("different native error".into()),
        F::UnknownAfterQkvMain
    ));
    assert!(!intended_error(
        &Error::Limit("closed original cubin qualification cancelled".into()),
        F::CancelBeforeQkvReducer
    ));
}

#[test]
fn unknown_qkv_cannot_claim_no_enqueue_completed_reducer_or_promoted_output() {
    use FixedPromptModelPrefixFault as F;
    for fault in [F::UnknownAfterQkvMain, F::CancelBeforeQkvReducer] {
        require_partial_qkv_witness(fault, true, 1, true).unwrap();
        for (cut, enqueues, unpromoted) in [
            (false, 1, true),
            (true, 0, true),
            (true, 2, true),
            (true, 1, false),
        ] {
            assert!(require_partial_qkv_witness(fault, cut, enqueues, unpromoted).is_err());
        }
    }
}

#[test]
fn qk_kv_partial_faults_refuse_both_old_cuts_and_keep_exact_errors() {
    use FixedPromptModelPrefixFault as F;
    let cuts = [
        FixedPromptPrefixKind::FirstThree,
        FixedPromptPrefixKind::ThroughAttentionQkv,
    ];
    for (fault, code, message) in [
        (
            F::UnknownAfterQk,
            5,
            "deliberate canonical47 unknown after actual QK",
        ),
        (
            F::CancelBeforeKvAppend,
            6,
            "full24 original KV transport cancelled",
        ),
        (
            F::UnknownAfterKvAppend,
            7,
            "deliberate canonical47 unknown after actual KV append",
        ),
        (
            F::CancelBeforeKvTransport,
            8,
            "full24 original KV transport cancelled",
        ),
    ] {
        for cut in cuts {
            assert!(require_fault_cut(cut, fault).is_err());
        }
        require_fault_cut(FixedPromptPrefixKind::ThroughAttentionKv, fault).unwrap();
        let mode = Mode::Fault {
            fault,
            probe: Arc::new(Probe::default()),
        };
        assert_eq!(mode.fault_mode(), code);
        assert!(intended_error(&Error::State(message.into()), fault));
        assert!(!intended_error(
            &Error::State("unrelated failure".into()),
            fault
        ));
        assert!(!intended_error(&Error::Limit(message.into()), fault));
    }
}

#[test]
fn kv_unknown_witness_rejects_missing_extra_or_promoted_operations() {
    use FixedPromptModelPrefixFault as F;
    for (fault, kv) in [
        (F::UnknownAfterQk, 0),
        (F::CancelBeforeKvAppend, 0),
        (F::UnknownAfterKvAppend, 1),
        (F::CancelBeforeKvTransport, 1),
    ] {
        require_partial_kv_witness(fault, true, (1, kv, 0), true, true).unwrap();
        for (cut, counts, qk_unpromoted, kv_unpromoted) in [
            (false, (1, kv, 0), true, true),
            (true, (0, kv, 0), true, true),
            (true, (2, kv, 0), true, true),
            (true, (1, 2, 0), true, true),
            (true, (1, kv, 1), true, true),
            (true, (1, kv, 0), false, true),
            (true, (1, kv, 0), true, false),
        ] {
            assert!(
                require_partial_kv_witness(fault, cut, counts, qk_unpromoted, kv_unpromoted)
                    .is_err()
            );
        }
    }
}

#[test]
fn fa_faults_are_cold_restricted_and_witness_actual_native_delta() {
    use FixedPromptModelPrefixFault as F;
    use FixedPromptPrefixKind as K;
    for (fault, code, native) in [(F::UnknownAfterFa, 9, 1), (F::CancelBeforeFa, 10, 0)] {
        for old in [K::FirstThree, K::ThroughAttentionQkv, K::ThroughAttentionKv] {
            assert!(require_fault_cut(old, fault).is_err());
        }
        require_fault_cut(K::ThroughAttentionFa, fault).unwrap();
        let mode = Mode::Fault {
            fault,
            probe: Arc::new(Probe::default()),
        };
        assert_eq!(mode.fault_mode(), code);
        require_partial_fa_witness(fault, true, (1, 1, 1), native, true).unwrap();
        for (cut, counts, fa, unpromoted) in [
            (false, (1, 1, 1), native, true),
            (true, (0, 1, 1), native, true),
            (true, (1, 0, 1), native, true),
            (true, (1, 1, 0), native, true),
            (true, (1, 1, 2), native, true),
            (true, (1, 1, 1), 1 - native, true),
            (true, (1, 1, 1), native, false),
        ] {
            assert!(require_partial_fa_witness(fault, cut, counts, fa, unpromoted).is_err());
        }
    }
}
#[test]
fn fa_diagnostic_errors_cannot_accept_an_unrelated_fault() {
    use FixedPromptModelPrefixFault as F;
    let u = Error::State("deliberate canonical47 unknown after actual FA".into());
    let c = Error::State("closed original cubin qualification cancelled".into());
    assert!(intended_error(&u, F::UnknownAfterFa));
    assert!(intended_error(&c, F::CancelBeforeFa));
    assert!(!intended_error(&u, F::CancelBeforeFa));
    assert!(!intended_error(&c, F::UnknownAfterFa));
    assert!(!intended_error(
        &Error::Limit("deliberate canonical47 unknown after actual FA".into()),
        F::UnknownAfterFa
    ));
}

#[test]
fn layer3_partial_faults_require_exact_stage_counts_and_source_cut() {
    use FixedPromptModelPrefixFault as F;
    use FixedPromptPrefixKind as K;
    for (fault, code, counts) in [
        (F::UnknownAfterSigmoid, 11, [1, 0, 0, 0, 0, 0]),
        (F::CancelBeforeMultiply, 12, [1, 0, 0, 0, 0, 0]),
        (F::UnknownAfterAttentionOut, 13, [1, 1, 1, 0, 0, 0]),
        (F::CancelBeforeActivation, 14, [1, 1, 1, 1, 0, 0]),
        (F::UnknownAfterLayer3DownMain, 15, [1, 1, 1, 1, 1, 1]),
        (F::CancelBeforeLayer3DownReducer, 16, [1, 1, 1, 1, 1, 1]),
    ] {
        for old in [
            K::FirstThree,
            K::ThroughAttentionQkv,
            K::ThroughAttentionKv,
            K::ThroughAttentionFa,
        ] {
            assert!(require_fault_cut(old, fault).is_err());
        }
        require_fault_cut(K::ThroughLayer3, fault).unwrap();
        let mode = Mode::Fault {
            fault,
            probe: Arc::new(Probe::default()),
        };
        assert_eq!(mode.fault_mode(), code);
        require_tail_witness(fault, true, (1, 1, 1), 1, counts, true).unwrap();
        for (cut, prior, fa, unpromoted) in [
            (false, (1, 1, 1), 1, true),
            (true, (1, 1, 0), 1, true),
            (true, (1, 1, 1), 0, true),
            (true, (1, 1, 1), 1, false),
        ] {
            assert!(require_tail_witness(fault, cut, prior, fa, counts, unpromoted).is_err());
        }
        for index in 0..6 {
            let mut wrong = counts;
            wrong[index] += 1;
            assert!(require_tail_witness(fault, true, (1, 1, 1), 1, wrong, true).is_err());
        }
    }
}
#[test]
fn layer3_faults_never_accept_another_stage_or_error_class() {
    use FixedPromptModelPrefixFault as F;
    for (fault, message) in [
        (
            F::UnknownAfterSigmoid,
            "deliberate canonical47 unknown after actual Sigmoid",
        ),
        (
            F::UnknownAfterAttentionOut,
            "deliberate canonical47 unknown after actual AttentionOut",
        ),
        (
            F::UnknownAfterLayer3DownMain,
            "deliberate canonical47 unknown after actual layer3 Down main",
        ),
        (
            F::CancelBeforeMultiply,
            "closed original cubin qualification cancelled",
        ),
        (
            F::CancelBeforeActivation,
            "closed original cubin qualification cancelled",
        ),
        (
            F::CancelBeforeLayer3DownReducer,
            "closed original cubin qualification cancelled",
        ),
    ] {
        assert!(intended_error(&Error::State(message.into()), fault));
        assert!(!intended_error(&Error::Limit(message.into()), fault));
        assert!(!intended_error(
            &Error::State("unrelated error".into()),
            fault
        ));
    }
}
