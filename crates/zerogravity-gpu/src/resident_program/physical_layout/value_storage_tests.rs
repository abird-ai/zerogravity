use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{
        MathematicalBackend, MatrixAxes, PlanBuilder, ProgramNumericalProfile, SourceTensor,
        ValueStorageFormat, ValueStorageLimits, ValueStorageRetention, ValueStorageReusePolicy,
    },
};
use zerogravity_core::quant::QuantKind;

#[test]
fn physical_adapter_refuses_growing_slots_and_preserves_explicit_exact_classes() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut builder =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let value = builder
        .embed(
            SourceTensor {
                source,
                name: "embedding".into(),
                kind: QuantKind::F32,
                axes: MatrixAxes::OutputInput,
                columns: 16,
                rows: 8,
            },
            PlanBuilder::inputs().token,
        )
        .unwrap();
    let large = builder
        .project(
            SourceTensor {
                source,
                name: "large".into(),
                kind: QuantKind::F32,
                axes: MatrixAxes::OutputInput,
                columns: 16,
                rows: 256,
            },
            &value,
        )
        .unwrap();
    let small = builder
        .project(
            SourceTensor {
                source,
                name: "small".into(),
                kind: QuantKind::F32,
                axes: MatrixAxes::OutputInput,
                columns: 256,
                rows: 16,
            },
            &large,
        )
        .unwrap();
    let slice = builder.finish_operator_slice(small).unwrap();
    let plan = slice.plan();
    let layouts = plan
        .values()
        .iter()
        .map(|shape| {
            ResidentBufferLayout::new(
                ResidentPhysicalType::F32,
                shape.capacity_elements().unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let limits = ValueStorageLimits {
        max_nodes: 3,
        max_values: 3,
        max_slots: 3,
        max_slot_bytes: 1024,
        max_total_bytes: 2048,
        alignment: 512,
        reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
    };
    let retention = ValueStorageRetention::Reuse {
        escaped: &[],
        pinned: &[],
        taps: &[],
    };
    let storage = ResidentBufferLayout::plan_values(plan, &layouts, retention, limits).unwrap();
    assert_eq!(storage.total_bytes(), 1536);
    assert_eq!(storage.slots().len(), 2);
    assert!(
        storage
            .slots()
            .iter()
            .all(|slot| slot.format == ValueStorageFormat::F32)
    );
    for binding in storage.bindings() {
        assert_eq!(
            binding.bytes.div_ceil(512) * 512,
            storage.slots()[binding.slot].reserved_bytes
        );
    }
    assert!(
        ResidentBufferLayout::plan_values(
            plan,
            &layouts,
            retention,
            ValueStorageLimits {
                reuse_policy: ValueStorageReusePolicy::BestFit,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        ResidentBufferLayout::plan_values(
            plan,
            &layouts,
            retention,
            ValueStorageLimits {
                max_values: 2,
                max_slots: 2,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        ResidentBufferLayout::plan_values(
            plan,
            &layouts,
            retention,
            ValueStorageLimits {
                max_nodes: 2,
                ..limits
            }
        )
        .is_err()
    );
    let too_many = vec![layouts[0]; 4];
    assert!(ResidentBufferLayout::plan_values(plan, &too_many, retention, limits).is_err());
    // Physical BF16 is explicit and cannot truncate actual F32 precision.
    let compact = plan
        .values()
        .iter()
        .map(|shape| {
            ResidentBufferLayout::new(
                ResidentPhysicalType::Bf16,
                shape.capacity_elements().unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(ResidentBufferLayout::plan_values(plan, &compact, retention, limits).is_err());
}
