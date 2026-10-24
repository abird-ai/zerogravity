//! Synthetic metadata guards only; no device or mathematical qualification.
use super::*;

fn copy(phase: VllmGdnPhase, slot: GdnCacheSlot, kind: CopyKind) -> Copy {
    let geometry = copy_geometry(phase, slot, kind).unwrap();
    let source: Arc<str> = "// cubecl.cuda.dynamic_shared_bytes=0\nsynthetic copy metadata".into();
    let bindings: Arc<[(usize, BufferIOAttr)]> = Arc::from([(4, BufferIOAttr::WriteOnly)]);
    Copy {
        launch: None,
        arithmetic: CopyArithmetic {
            kind,
            geometry,
            source,
            entry: "metadata_only",
            grid: [1, 1, 1],
            bindings,
        },
    }
}

fn instance(phase: VllmGdnPhase, slot: GdnCacheSlot, typed: bool) -> GdnArithmetic {
    instance_family(phase, slot, typed, false)
}
fn instance_family(
    phase: VllmGdnPhase,
    slot: GdnCacheSlot,
    typed: bool,
    two_b: bool,
) -> GdnArithmetic {
    let controls = metadata(phase, slot)
        .unwrap()
        .into_iter()
        .map(|(r, b)| (r, Arc::from(b)))
        .collect();
    let stages: Vec<_> = phase
        .stages()
        .iter()
        .map(|&stage| Stage {
            spec: if two_b {
                crate::vllm_cuda_gdn::kernel_qwen2b(phase, stage, typed).unwrap()
            } else if typed {
                kernel_with_typed_outputs(phase, stage).unwrap()
            } else {
                kernel(phase, stage).unwrap()
            },
            stage,
            launch: None,
        })
        .collect();
    let zero =
        (phase != VllmGdnPhase::CachedDecode).then(|| copy(phase, slot, CopyKind::ZeroInverse));
    let initial = (phase != VllmGdnPhase::CachedDecode).then(|| {
        copy(
            phase,
            slot,
            if phase.has_initial_state() {
                CopyKind::Gather
            } else {
                CopyKind::Zero
            },
        )
    });
    let scatter =
        (phase != VllmGdnPhase::CachedDecode).then(|| copy(phase, slot, CopyKind::Scatter));
    GdnArithmetic::new(
        phase,
        slot,
        typed,
        controls,
        &stages,
        &[],
        [zero.as_ref(), initial.as_ref(), scatter.as_ref()],
    )
    .unwrap()
}

#[test]
fn actual_slot_control_bytes_typed_output_and_zero_initialization_change_identity() {
    let actual = instance(VllmGdnPhase::Initial64, GdnCacheSlot::Four, true);
    actual.require_same(&actual.clone()).unwrap();
    let mut changed = actual.clone();
    Arc::make_mut(&mut Arc::make_mut(&mut changed.controls)[1].1)[0] ^= 1;
    assert!(actual.require_same(&changed).is_err());
    let mut changed = actual.clone();
    changed.inverse_zero.as_mut().unwrap().source = "changed zero-Ai implementation".into();
    assert!(actual.require_same(&changed).is_err());
    let mut changed = actual.clone();
    changed.inverse_zero.as_mut().unwrap().geometry.words -= 1;
    assert!(actual.require_same(&changed).is_err());
    assert!(
        actual
            .require_same(&instance(
                VllmGdnPhase::Initial64,
                GdnCacheSlot::Four,
                false
            ))
            .is_err()
    );
    let a = instance(VllmGdnPhase::CachedDecode, GdnCacheSlot::One, true);
    let b = instance(VllmGdnPhase::CachedDecode, GdnCacheSlot::Four, true);
    assert!(a.require_same(&b).is_err());
}

#[test]
fn missing_or_wrong_initialization_refuses_without_constructing_a_ready_or_permit() {
    let phase = VllmGdnPhase::InitialOne;
    let stages: Vec<_> = phase
        .stages()
        .iter()
        .map(|&stage| Stage {
            spec: kernel(phase, stage).unwrap(),
            stage,
            launch: None,
        })
        .collect();
    let controls = metadata(phase, GdnCacheSlot::One)
        .unwrap()
        .into_iter()
        .map(|(r, b)| (r, Arc::from(b)))
        .collect::<Arc<[(GdnBuffer, Arc<[u8]>)]>>();
    assert!(
        GdnArithmetic::new(
            phase,
            GdnCacheSlot::One,
            false,
            controls.clone(),
            &stages,
            &[],
            [None, None, None]
        )
        .is_err()
    );
    assert!(
        GdnArithmetic::new(
            phase,
            GdnCacheSlot::One,
            false,
            controls,
            &stages[..stages.len() - 1],
            &[],
            [None, None, None]
        )
        .is_err()
    );
    assert!(metadata(phase, GdnCacheSlot::Four).is_err());
    let continued = instance(VllmGdnPhase::ContinuedTwo, GdnCacheSlot::Four, true);
    assert_eq!(continued.initial.unwrap().kind, CopyKind::Gather);
    let initial = instance(VllmGdnPhase::InitialOne, GdnCacheSlot::One, true);
    assert_eq!(initial.initial.unwrap().kind, CopyKind::Zero);
}

#[test]
fn fixed_old_metadata_owners_fit_existing_flat_admission_allowance() {
    for (phase, slot) in [
        (VllmGdnPhase::InitialOne, GdnCacheSlot::One),
        (VllmGdnPhase::Initial64, GdnCacheSlot::Four),
        (VllmGdnPhase::ContinuedTwo, GdnCacheSlot::Four),
        (VllmGdnPhase::CachedDecode, GdnCacheSlot::One),
        (VllmGdnPhase::CachedDecode, GdnCacheSlot::Four),
        (VllmGdnPhase::Initial47, GdnCacheSlot::One),
    ] {
        let value = instance(phase, slot, true);
        assert!(value.logical_metadata_bytes() <= 64 * 1024);
        assert_eq!(
            value.controls.iter().map(|(_, b)| b.len()).sum::<usize>(),
            if phase == VllmGdnPhase::CachedDecode {
                4
            } else {
                16421
            }
        );
        assert!(format!("{value:?}").len() < 256);
    }
}

#[test]
fn bound_instance_derives_actual_family_and_checks_all_stages_and_coordinates() {
    for phase in [VllmGdnPhase::InitialOne, VllmGdnPhase::Initial64] {
        let slot = if phase == VllmGdnPhase::InitialOne {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let two = instance_family(phase, slot, true, true);
        assert_eq!(two.family(), &GdnFamilyIdentity::original2b());
        two.require_instance(phase, slot, true).unwrap();
        assert!(two.require_same(&instance(phase, slot, true)).is_err());
        assert!(two.require_instance(phase, slot, false).is_err());
        assert!(
            two.require_instance(VllmGdnPhase::CachedDecode, slot, true)
                .is_err()
        );
        let wrong_slot = if slot == GdnCacheSlot::One {
            GdnCacheSlot::Four
        } else {
            GdnCacheSlot::One
        };
        assert!(two.require_instance(phase, wrong_slot, true).is_err());
        assert!(two.logical_metadata_bytes() <= 64 * 1024);
    }
    let phase = VllmGdnPhase::InitialOne;
    let slot = GdnCacheSlot::One;
    let mut stages: Vec<_> = phase
        .stages()
        .iter()
        .map(|&stage| Stage {
            spec: crate::vllm_cuda_gdn::kernel_qwen2b(phase, stage, true).unwrap(),
            stage,
            launch: None,
        })
        .collect();
    // Prep has the same original code in both families; identity must still
    // reject a silently mixed declared family before any device operation.
    stages[1].spec = kernel_with_typed_outputs(phase, GdnStage::Prep).unwrap();
    let controls = metadata(phase, slot)
        .unwrap()
        .into_iter()
        .map(|(r, b)| (r, Arc::from(b)))
        .collect();
    let zero = copy(phase, slot, CopyKind::ZeroInverse);
    let initial = copy(phase, slot, CopyKind::Zero);
    let scatter = copy(phase, slot, CopyKind::Scatter);
    assert!(
        GdnArithmetic::new(
            phase,
            slot,
            true,
            controls,
            &stages,
            &[],
            [Some(&zero), Some(&initial), Some(&scatter)]
        )
        .is_err()
    );
}
