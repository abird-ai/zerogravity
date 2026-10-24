use super::*;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    EmbeddingContract, MathematicalBackend, MatrixAxes, PlanBuilder, ProgramNumericalProfile,
    RecordedOperatorSlice, SourceTensor,
};
use crate::quant::QuantKind;

fn tensor(source: SourceIdentity, kind: QuantKind) -> SourceTensor {
    SourceTensor {
        source,
        name: "embedding".into(),
        kind,
        axes: MatrixAxes::OutputInput,
        columns: 16,
        rows: 8,
    }
}
fn chain(count: usize, bf16: bool) -> RecordedOperatorSlice {
    assert!(count > 0);
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut b = PlanBuilder::new_operator_slice(
        source,
        if bf16 {
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        } else {
            ProgramNumericalProfile::GgufSourceV1
        },
    )
    .unwrap();
    let mut value = if bf16 {
        b.embed_profiled(
            tensor(source, QuantKind::Bf16),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap()
    } else {
        b.embed(tensor(source, QuantKind::F32), PlanBuilder::inputs().token)
            .unwrap()
    };
    for index in 1..count {
        value = if bf16 {
            b.torch_cpu_bf16_add(&format!("v{index}"), &value, &value)
                .unwrap()
        } else {
            b.silu(&format!("v{index}"), &value, None).unwrap()
        };
    }
    b.finish_operator_slice(value).unwrap()
}
fn layouts(plan: &ForwardPlan, bf16: bool) -> Vec<ValueStorageLayout> {
    plan.values()
        .iter()
        .map(|s| ValueStorageLayout {
            format: if bf16 {
                ValueStorageFormat::Bf16
            } else {
                ValueStorageFormat::F32
            },
            elements: s.capacity_elements().unwrap(),
        })
        .collect()
}
fn limits() -> ValueStorageLimits {
    ValueStorageLimits {
        max_nodes: 64,
        max_values: 64,
        max_slots: 64,
        max_slot_bytes: 4096,
        max_total_bytes: 64 * 4096,
        alignment: 64,
        reuse_policy: ValueStorageReusePolicy::BestFit,
    }
}
fn reuse() -> ValueStorageRetention<'static> {
    ValueStorageRetention::Reuse {
        escaped: &[],
        pinned: &[],
        taps: &[],
    }
}

#[test]
fn actual_builder_chain_reuses_only_after_last_consumer_and_never_in_place() {
    let slice = chain(6, false);
    let plan = slice.plan();
    let storage = plan
        .plan_value_storage(&layouts(plan, false), reuse(), limits())
        .unwrap();
    assert_eq!(storage.slots().len(), 2);
    assert_eq!(storage.total_bytes(), 128);
    assert_eq!(
        storage.lifetimes()[0],
        ValueLifetime {
            producer: 0,
            last_use: 1
        }
    );
    assert_eq!(storage.bindings()[0].slot, storage.bindings()[2].slot);
    assert_eq!(
        (storage.bindings()[0].epoch, storage.bindings()[2].epoch),
        (0, 1)
    );
    for node in 1..6 {
        let input = storage
            .binding_at(node, ValueId(node as u32 - 1), DeviceAccess::Read)
            .unwrap();
        let output = storage
            .binding_at(node, ValueId(node as u32), DeviceAccess::Write)
            .unwrap();
        assert_ne!(input.slot, output.slot);
    }
    assert!(
        storage
            .binding_at(1, ValueId(2), DeviceAccess::Read)
            .is_err()
    );
    assert!(
        storage
            .binding_at(1, ValueId(0), DeviceAccess::Write)
            .is_err()
    );
    assert!(
        storage
            .binding_at(6, ValueId(5), DeviceAccess::Write)
            .is_err()
    );
}

#[test]
fn retained_all_is_distinct_from_source_owned_pins_and_taps() {
    let slice = chain(6, false);
    let plan = slice.plan();
    let layout = layouts(plan, false);
    let all = plan
        .plan_value_storage(&layout, ValueStorageRetention::RetainAll, limits())
        .unwrap();
    assert_eq!((all.slots().len(), all.total_bytes()), (6, 384));
    assert!(all.taps().is_empty());
    let pin = [ValueId(0)];
    let pinned = plan
        .plan_value_storage(
            &layout,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &pin,
                taps: &[],
            },
            limits(),
        )
        .unwrap();
    assert_eq!((pinned.slots().len(), pinned.total_bytes()), (3, 192));
    assert_eq!(pinned.lifetimes()[0].last_use, plan.nodes().len());
    let tap_values = [ValueId(0), ValueId(3)];
    let tapped = plan
        .plan_value_storage(
            &layout,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &[],
                taps: &tap_values,
            },
            limits(),
        )
        .unwrap();
    assert_eq!((tapped.slots().len(), tapped.total_bytes()), (2, 256));
    assert_eq!(
        tapped.taps(),
        &[
            ValueStorageTap {
                value: ValueId(0),
                after_node: 0,
                bytes: 64,
                reserved_bytes: 64
            },
            ValueStorageTap {
                value: ValueId(3),
                after_node: 3,
                bytes: 64,
                reserved_bytes: 64
            },
        ]
    );
    // Tap metadata does not extend a scratch value or issue its D2D/completion.
    assert_eq!(tapped.lifetimes()[0].last_use, 1);
}

#[test]
fn late_pinned_output_gets_a_new_slot_but_escape_can_reuse_early_dead_storage() {
    let slice = chain(5, false);
    let plan = slice.plan();
    let layout = layouts(plan, false);
    let late = [ValueId(3)];
    let pinned = plan
        .plan_value_storage(
            &layout,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &late,
                taps: &[],
            },
            limits(),
        )
        .unwrap();
    let dedicated = pinned.bindings()[3];
    assert!(pinned.slots()[dedicated.slot].dedicated);
    assert_eq!(dedicated.epoch, 0);
    for value in [0, 1, 2, 4] {
        assert_ne!(dedicated.slot, pinned.bindings()[value].slot);
    }
    assert_eq!((pinned.slots().len(), pinned.total_bytes()), (3, 192));
    let escaped = plan
        .plan_value_storage(
            &layout,
            ValueStorageRetention::Reuse {
                escaped: &late,
                pinned: &[],
                taps: &[],
            },
            limits(),
        )
        .unwrap();
    assert_eq!(escaped.bindings()[3].slot, escaped.bindings()[1].slot);
    assert_eq!(escaped.bindings()[3].epoch, 1);
    assert!(!escaped.slots()[escaped.bindings()[3].slot].dedicated);
    assert_eq!((escaped.slots().len(), escaped.total_bytes()), (2, 128));
    let mut hostile = pinned;
    hostile.dedicated[3] = false;
    assert!(hostile.validate().is_err());
}

#[test]
fn exact_aligned_policy_never_grows_or_borrows_a_different_native_size_class() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut b =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let mut value = b
        .embed(tensor(source, QuantKind::F32), PlanBuilder::inputs().token)
        .unwrap();
    let mut columns = 16;
    for (index, rows) in [8, 32, 4, 16].into_iter().enumerate() {
        value = b
            .project(
                SourceTensor {
                    source,
                    name: format!("projection{index}"),
                    kind: QuantKind::F32,
                    axes: MatrixAxes::OutputInput,
                    columns,
                    rows,
                },
                &value,
            )
            .unwrap();
        columns = rows;
    }
    let slice = b.finish_operator_slice(value).unwrap();
    let plan = slice.plan();
    let layout = layouts(plan, false);
    let growing = plan.plan_value_storage(&layout, reuse(), limits()).unwrap();
    assert_eq!((growing.slots().len(), growing.total_bytes()), (2, 192));
    assert_eq!(growing.bindings()[0].slot, growing.bindings()[2].slot);
    assert!(
        growing.bindings()[0].bytes < growing.slots()[growing.bindings()[0].slot].reserved_bytes
    );
    let exact = plan
        .plan_value_storage(
            &layout,
            reuse(),
            ValueStorageLimits {
                reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
                ..limits()
            },
        )
        .unwrap();
    assert_eq!((exact.slots().len(), exact.total_bytes()), (3, 256));
    assert_ne!(exact.bindings()[0].slot, exact.bindings()[2].slot);
    for binding in exact.bindings() {
        assert_eq!(
            limits().aligned(binding.bytes).unwrap(),
            exact.slots()[binding.slot].reserved_bytes
        );
    }
    let mut hostile = growing;
    hostile.limits.reuse_policy = ValueStorageReusePolicy::ExactAlignedSize;
    assert!(hostile.validate().is_err());
}

#[test]
fn fork_and_late_residual_keep_all_actual_node_operands_nonaliasing() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut b =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let original = b
        .embed(tensor(source, QuantKind::F32), PlanBuilder::inputs().token)
        .unwrap();
    let left = b.silu("left", &original, None).unwrap();
    let right = b.silu("right", &original, None).unwrap();
    let merged = b.add("merged", &left, &right).unwrap();
    let final_value = b.add("residual", &merged, &original).unwrap();
    let slice = b.finish_operator_slice(final_value).unwrap();
    let plan = slice.plan();
    let storage = plan
        .plan_value_storage(&layouts(plan, false), reuse(), limits())
        .unwrap();
    assert_eq!(storage.lifetimes()[0].last_use, 4);
    for (index, node) in plan.nodes().iter().enumerate() {
        for input in &node.inputs {
            for output in &node.outputs {
                assert_ne!(
                    storage
                        .binding_at(index, *input, DeviceAccess::Read)
                        .unwrap()
                        .slot,
                    storage
                        .binding_at(index, *output, DeviceAccess::Write)
                        .unwrap()
                        .slot
                );
            }
        }
    }
    assert_eq!(storage.slots().len(), 4);
}

#[test]
fn production_order_not_numeric_value_id_controls_reuse_epochs() {
    let slice = chain(5, false);
    let mut plan = slice.plan().clone();
    let rename = |id: &mut ValueId| {
        id.0 = match id.0 {
            0 => 2,
            2 => 0,
            other => other,
        };
    };
    for node in &mut plan.nodes {
        let topology = &mut **node;
        for id in topology.inputs.iter_mut().chain(&mut topology.outputs) {
            rename(id);
        }
    }
    let storage = plan
        .plan_value_storage(&layouts(&plan, false), reuse(), limits())
        .unwrap();
    assert_eq!(storage.lifetimes()[2].producer, 0);
    assert_eq!(storage.lifetimes()[0].producer, 2);
    assert_eq!(storage.bindings()[2].slot, storage.bindings()[0].slot);
    assert_eq!(
        (storage.bindings()[2].epoch, storage.bindings()[0].epoch),
        (0, 1)
    );
}

#[test]
fn duplicate_future_missing_and_out_of_range_ssa_producers_refuse() {
    let slice = chain(4, false);
    for mutation in 0..5 {
        let mut plan = slice.plan().clone();
        match mutation {
            0 => plan.nodes[0].inputs.push(ValueId(1)),
            1 => plan.nodes[1].outputs[0] = ValueId(0),
            2 => plan.nodes[1].outputs.clear(),
            3 => plan.nodes[1].outputs[0] = ValueId(u32::MAX),
            _ => plan.nodes[1].inputs[0] = ValueId(u32::MAX),
        }
        assert!(
            plan.plan_value_storage(&layouts(&plan, false), reuse(), limits())
                .is_err()
        );
    }
}

#[test]
fn binding_metadata_cannot_turn_an_equal_plan_clone_into_the_same_issuer() {
    let slice = chain(4, false);
    let plan = slice.plan();
    let storage = plan
        .plan_value_storage(&layouts(plan, false), reuse(), limits())
        .unwrap();
    assert!(storage.require_plan(plan).is_ok());
    assert!(storage.require_plan(&plan.clone()).is_err());
}

#[test]
fn physical_precision_count_overflow_and_format_reinterpretation_refuse() {
    let f32 = chain(5, false);
    assert!(
        f32.plan()
            .plan_value_storage(&layouts(f32.plan(), true), reuse(), limits())
            .is_err()
    );
    let bf16 = chain(5, true);
    let plan = bf16.plan();
    let all_bf16 = layouts(plan, true);
    assert_eq!(
        plan.plan_value_storage(&all_bf16, reuse(), limits())
            .unwrap()
            .slots()
            .len(),
        2
    );
    let mut mixed = all_bf16.clone();
    mixed[2].format = ValueStorageFormat::F32;
    let storage = plan.plan_value_storage(&mixed, reuse(), limits()).unwrap();
    assert_ne!(storage.bindings()[0].slot, storage.bindings()[2].slot);
    for invalid in [0, usize::MAX] {
        let mut wrong = all_bf16.clone();
        wrong[0].elements = invalid;
        assert!(plan.plan_value_storage(&wrong, reuse(), limits()).is_err());
        assert!(
            ValueStorageLayout {
                format: ValueStorageFormat::F32,
                elements: invalid
            }
            .bytes()
            .is_err()
        );
    }
    assert!(
        plan.plan_value_storage(&all_bf16[..4], reuse(), limits())
            .is_err()
    );
}

#[test]
fn exact_total_alignment_slots_and_metadata_limits_include_tap_destinations() {
    let slice = chain(6, false);
    let plan = slice.plan();
    let layout = layouts(plan, false);
    let exact = ValueStorageLimits {
        max_slots: 2,
        max_total_bytes: 128,
        ..limits()
    };
    assert_eq!(
        plan.plan_value_storage(&layout, reuse(), exact)
            .unwrap()
            .total_bytes(),
        128
    );
    for wrong in [
        ValueStorageLimits {
            max_slots: 1,
            ..exact
        },
        ValueStorageLimits {
            max_total_bytes: 127,
            ..exact
        },
        ValueStorageLimits {
            max_slot_bytes: 63,
            ..exact
        },
        ValueStorageLimits {
            max_values: 5,
            ..exact
        },
        ValueStorageLimits {
            max_nodes: 5,
            ..exact
        },
        ValueStorageLimits {
            alignment: 3,
            ..exact
        },
        ValueStorageLimits {
            alignment: 0,
            ..exact
        },
    ] {
        assert!(plan.plan_value_storage(&layout, reuse(), wrong).is_err());
    }
    assert!(
        plan.plan_value_storage(
            &layout,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &[],
                taps: &[ValueId(0)]
            },
            exact
        )
        .is_err()
    );
    assert!(
        plan.plan_value_storage(&layout, ValueStorageRetention::RetainAll, exact)
            .is_err()
    );
}

#[test]
fn invalid_retention_and_hostile_live_slot_epoch_extent_mutations_refuse() {
    let slice = chain(5, false);
    let plan = slice.plan();
    let layout = layouts(plan, false);
    for mode in [
        ValueStorageRetention::Reuse {
            escaped: &[ValueId(0)],
            pinned: &[ValueId(0)],
            taps: &[],
        },
        ValueStorageRetention::Reuse {
            escaped: &[],
            pinned: &[],
            taps: &[ValueId(1), ValueId(1)],
        },
        ValueStorageRetention::Reuse {
            escaped: &[ValueId(u32::MAX)],
            pinned: &[],
            taps: &[],
        },
    ] {
        assert!(plan.plan_value_storage(&layout, mode, limits()).is_err());
    }
    for mutation in 0..6 {
        let mut storage = plan.plan_value_storage(&layout, reuse(), limits()).unwrap();
        match mutation {
            0 => storage.bindings[1].slot = storage.bindings[0].slot,
            1 => storage.bindings[2].epoch = storage.bindings[0].epoch,
            2 => storage.bindings[2].bytes -= 1,
            3 => storage.slots[0].reserved_bytes -= 1,
            4 => storage.total_bytes -= 1,
            _ => storage.lifetimes[0].last_use = 0,
        }
        assert!(storage.validate().is_err());
    }
}

#[test]
fn unpublished_head_and_published_logits_escape_are_explicit_separate_cases() {
    let slice = chain(5, false);
    let plan = slice.plan();
    let layout = layouts(plan, false);
    let pin_head = plan
        .plan_value_storage(
            &layout,
            ValueStorageRetention::Reuse {
                escaped: &[slice.output()],
                pinned: &[],
                taps: &[],
            },
            limits(),
        )
        .unwrap();
    assert_eq!(pin_head.lifetimes()[4].last_use, plan.nodes().len());
    let mut published = plan.clone();
    published.logits = Some(slice.output());
    let implicit = published
        .plan_value_storage(&layout, reuse(), limits())
        .unwrap();
    assert_eq!(implicit.lifetimes()[4].last_use, published.nodes().len());
}
