//! Synthetic transport only; no trained convolution or model qualification.
use super::*;
fn fixture(dtype: &str, shape: &[u64], bytes: usize) -> Fixture {
    let mut fixture = Fixture::new();
    let raw = (0..bytes / 2)
        .flat_map(|i| (0x3f80u16 + (i % 8) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    fixture.second = safetensors("conv", dtype, shape, &raw);
    fixture.index = serde_json::to_vec(
        &json!({"metadata":{"total_size":64+raw.len()},"weight_map":{"matrix":A,"conv":B}}),
    )
    .unwrap();
    fs::write(fixture.path.join(B), &fixture.second).unwrap();
    fixture
}
#[test]
fn full_original_rank3_and_partial_range_keep_truthful_metadata_and_owner() {
    let f = fixture("BF16", &[32, 1, 4], 256);
    let host = admission(64 * 1024 * 1024, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let before = reserved(&host);
    assert!(source.parameter("conv").is_err());
    let raw = source.conv4_parameter("conv").unwrap();
    let descriptor = raw.source_parameter();
    assert_eq!(descriptor.shape, [32, 1, 4]);
    assert_eq!(descriptor.bytes, 256);
    raw.validate_source_parameter(&descriptor).unwrap();
    let partial = source.read_parameter(&raw, 4..8).unwrap();
    assert!(!partial.is_complete());
    assert!(partial.require_complete(&raw).is_err());
    assert_eq!(partial.parameter().tensor().shape, [32, 1, 4]);
    assert_eq!(partial.element_range(), 4..8);
    drop(partial);
    assert_eq!(reserved(&host), before);
    let full = source.read_parameter(&raw, 0..128).unwrap();
    full.require_complete(&raw).unwrap();
    assert_eq!(full.as_slice().len(), 256);
    assert!(full.charged_staging_bytes() >= 256);
    assert!(source.read_parameter(&raw, 0..129).is_err());
    assert!(source.read_parameter(&raw, 4..4).is_err());
    let foreign = f.plan(&host).unwrap().open().unwrap();
    assert!(foreign.read_parameter(&raw, 0..128).is_err());
    assert!(
        full.require_complete(&foreign.conv4_parameter("conv").unwrap())
            .is_err()
    );
    drop(foreign);
    drop(descriptor);
    drop(raw);
    drop(source);
    assert_eq!(full.as_slice().len(), 256);
    assert!(reserved(&host) > 0);
    drop(full);
    assert_eq!(reserved(&host), 0);
}
#[test]
fn declared_binding_preserves_rank_and_rejects_forged_metadata_without_staging() {
    for (dtype, shape) in [("BF16", vec![32, 1, 4]), ("F32", vec![64])] {
        let f = fixture(dtype, &shape, 256);
        let host = admission(64 * 1024 * 1024, "host");
        let source = f.plan(&host).unwrap().open().unwrap();
        let foreign = f.plan(&host).unwrap().open().unwrap();
        let raw = if shape.len() == 3 {
            assert!(source.parameter("conv").is_err());
            source.conv4_parameter("conv").unwrap()
        } else {
            source.parameter("conv").unwrap()
        };
        let descriptor = raw.source_parameter();
        let baseline = reserved(&host);
        let bound = source.bind_declared_parameter(&descriptor).unwrap();
        assert_eq!(bound.source_parameter(), descriptor);
        assert_eq!(reserved(&host), baseline);
        for field in 0..5 {
            let mut forged = descriptor.clone();
            match field {
                0 => forged.source = foreign.source_identity(),
                1 => forged.name = "matrix".into(),
                2 => {
                    forged.kind = zerogravity_core::encoding::StoredTensorEncoding::Scalar(
                        zerogravity_core::encoding::ScalarDType::F16,
                    )
                }
                3 => forged.shape = vec![bound.elements() as u64, 1],
                4 => forged.bytes += 2,
                _ => unreachable!(),
            }
            assert!(source.bind_declared_parameter(&forged).is_err());
            assert!(source.read_declared_parameter(&forged, None).is_err());
            assert_eq!(reserved(&host), baseline);
        }
        let cancelled = AtomicBool::new(true);
        assert!(
            source
                .read_declared_parameter(&descriptor, Some(&cancelled))
                .is_err()
        );
        assert_eq!(reserved(&host), baseline);
        let window = source.read_declared_parameter(&descriptor, None).unwrap();
        assert!(window.is_complete());
        assert_eq!(window.parameter().source_parameter(), descriptor);
        assert_eq!(window.as_slice().len(), 256);
        drop((bound, raw, source, foreign));
        assert!(reserved(&host) > 0);
        assert_eq!(window.parameter().tensor().shape, shape);
        drop(window);
        assert_eq!(reserved(&host), 0);
    }
}

#[test]
fn geometry_corruption_cancel_and_pressure_refuse_without_leases() {
    for (dtype, shape, bytes) in [
        ("F32", vec![32, 1, 4], 512),
        ("BF16", vec![128], 256),
        ("BF16", vec![32, 4, 1], 256),
        ("BF16", vec![16, 1, 4], 128),
    ] {
        let f = fixture(dtype, &shape, bytes);
        let host = admission(64 * 1024 * 1024, "host");
        let source = f.plan(&host).unwrap().open().unwrap();
        assert!(source.conv4_parameter("conv").is_err());
        drop(source);
        assert_eq!(reserved(&host), 0);
    }
    let f = fixture("BF16", &[32, 1, 4], 256);
    let host = admission(64 * 1024 * 1024, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let raw = source.conv4_parameter("conv").unwrap();
    let baseline = reserved(&host);
    let cancelled = AtomicBool::new(true);
    assert!(
        source
            .read_parameter_cancellable(&raw, 0..128, &cancelled)
            .is_err()
    );
    assert_eq!(reserved(&host), baseline);
    let remaining = 64 * 1024 * 1024 - baseline;
    let pressure = host
        .arbiter
        .admit(
            &host.resource,
            host.owner.clone(),
            &[zerogravity_core::resource_arbiter::AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: remaining,
            }],
        )
        .unwrap();
    assert!(source.read_parameter(&raw, 0..128).is_err());
    drop(pressure);
    assert_eq!(reserved(&host), baseline);
    let mut corrupt = f.second.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    fs::write(f.path.join(B), corrupt).unwrap();
    assert!(source.read_parameter(&raw, 0..128).is_err());
    assert_eq!(reserved(&host), baseline);
    drop(raw);
    drop(source);
    assert_eq!(reserved(&host), 0);
}
