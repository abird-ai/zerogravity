use super::*;
#[test]
fn closed_profiles_have_exact_real_provider_scopes() {
    let old = FixedPromptLifecycleProfile::FiniteTwo;
    let new = FixedPromptLifecycleProfile::BenchmarkEight;
    assert_eq!(
        (old.session_limit(), old.provider_bytes(), old.host_bytes()),
        (2, 2304 * MIB, 3072 * MIB)
    );
    assert_eq!(
        (new.session_limit(), new.provider_bytes(), new.host_bytes()),
        (8, 4096 * MIB, 4096 * MIB)
    );
    assert_eq!(new.metadata_bytes(), 2 * MIB);
    assert_eq!(old.metadata_bytes(), 0);
    assert_eq!(PENDING, 2);
    new.require_device_cap(5120 * MIB as u64).unwrap();
    for cap in [
        0,
        4096 * MIB as u64,
        5120 * MIB as u64 - 1,
        5120 * MIB as u64 + 1,
    ] {
        assert!(new.require_device_cap(cap).is_err());
    }
}
#[test]
fn fourteen_real_roots_fit_the_same_alignment_inventory() {
    assert_eq!(ROOT_BYTES, 32_784_384);
    let (device, host) = eight_bounds().unwrap();
    assert_eq!(device, 2_373_980_984);
    assert_eq!(host, 3_466_318_232);
    assert!(device < FixedPromptLifecycleProfile::BenchmarkEight.provider_bytes());
    assert!(host < FixedPromptLifecycleProfile::BenchmarkEight.host_bytes());
    // The old source envelope and its refusal boundaries are unchanged.
    assert_eq!(
        require_gates(PROVIDER, HOST).unwrap(),
        (2304 * MIB, 3072 * MIB)
    );
    assert!(require_gates(Some(4096 * MIB), Some(4096 * MIB)).is_err());
}
#[test]
fn metadata_inventory_refuses_unbounded_or_wrong_root_shapes() {
    require_metadata_counts(
        64,
        48,
        std::mem::size_of::<crate::resident_inference::Domain>(),
    )
    .unwrap();
    for (domains, planes, bytes) in [
        (65, 48, 1),
        (64, 47, 1),
        (64, 49, 1),
        (64, 48, 1025),
        (usize::MAX, 48, 1),
        (64, usize::MAX, 1),
        (64, 48, usize::MAX),
    ] {
        assert!(require_metadata_counts(domains, planes, bytes).is_err());
    }
}
