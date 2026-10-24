//! Complete semantic fixtures; no trained source or physical execution grant.
use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{FullModelProfile, ShortHfWorkload},
};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
#[path = "fixture.rs"]
mod fixture;
use fixture::{call, record};

fn metadata(
    source: SourceIdentity,
    bytes: u64,
) -> (
    ResourceArbiter,
    PhysicalDeviceKey,
    zerogravity_core::resource_arbiter::HostWorkspace,
    std::sync::Weak<Vec<u8>>,
) {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "prepared-binder-test").unwrap();
    arbiter.register(key.clone(), bytes).unwrap();
    let lease = arbiter
        .admit(
            &key,
            ResourceOwner::new("original-source", "binder-metadata").unwrap(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes,
            }],
        )
        .unwrap();
    let original = Arc::new(vec![1]);
    let witness = Arc::downgrade(&original);
    (
        arbiter,
        key,
        lease.into_host_workspace(source, original).unwrap(),
        witness,
    )
}
#[test]
fn all_shapes_share_complete_sites_while_every_real_invocation_stays_distinct() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for workload in [
            ShortHfWorkload::Prompt47,
            ShortHfWorkload::Prompt57,
            ShortHfWorkload::Prompt62,
            ShortHfWorkload::Prompt67,
            ShortHfWorkload::Prompt72,
        ] {
            for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                let first = call(source, profile, workload, base, 11);
                let template = Arc::new(record(&first));
                let mut validations = 0;
                let analysis = analyze_record(&first, &template, |op| {
                    validations += 1;
                    assert_eq!(op.call(), first.as_vllm_call());
                    Ok(())
                })?;
                assert_eq!(validations, 254);
                assert_eq!(
                    (
                        analysis.rms.len(),
                        analysis.projections.len(),
                        analysis.gdn.len(),
                        analysis.attention.len()
                    ),
                    (49, 114, 18, 6)
                );
                let (_arbiter, _key, metadata, _witness) = metadata(source, 512 * 1024);
                assert!(
                    PreparedShortHfModel::metadata_bytes(&template, &analysis)? <= metadata.bytes()
                );
                let prepared = Arc::new(PreparedShortHfModel::new(
                    template.clone(),
                    analysis,
                    metadata,
                )?);
                for id in [353, 353, 78] {
                    let current = call(source, profile, workload, base, id);
                    let bound = template.bind_short_hf_call(&current)?;
                    prepared.require_call(&current, &bound)?;
                    assert_ne!(template.call(), bound.call());
                    assert!(prepared.require_call(&first, &bound).is_err());
                    assert_eq!(validations, 254);
                    for (&layer, site) in &prepared.analysis.gdn {
                        assert_eq!(
                            typed(&bound, site.nodes[0])?.layer().unwrap().index(),
                            layer
                        );
                        assert_eq!(typed(&bound, site.nodes[0])?.call(), current.as_vllm_call());
                    }
                    for site in prepared.analysis.attention.values() {
                        let node = &bound.plan().nodes()[site.nodes[3]];
                        require_state_read(node, typed(&bound, site.nodes[3])?)?;
                    }
                }
                let independent = record(&first);
                assert!(prepared.require_call(&first, &independent).is_err());
            }
        }
    }
    Ok(())
}
#[test]
fn failed_source_join_refuses_and_prepared_lease_survives_cache_owner_drop() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let first = call(
        source,
        FullModelProfile::Qwen08,
        ShortHfWorkload::Prompt47,
        None,
        11,
    );
    let record = Arc::new(record(&first));
    let mut members = 0;
    assert!(
        analyze_record(&first, &record, |_| {
            members += 1;
            if members == 7 {
                Err(Error::State("actual original member join failed".into()))
            } else {
                Ok(())
            }
        })
        .is_err()
    );
    assert_eq!(members, 7);
    let analysis = analyze_record(&first, &record, |_| Ok(()))?;
    let required = PreparedShortHfModel::metadata_bytes(&record, &analysis)?;
    let (short_arbiter, short_key, short_metadata, short_owner) =
        metadata(source, required as u64 - 1);
    assert!(PreparedShortHfModel::new(record.clone(), analysis, short_metadata).is_err());
    assert_eq!(short_arbiter.snapshot(&short_key)?.reserved, 0);
    assert!(short_owner.upgrade().is_none());
    let analysis = analyze_record(&first, &record, |_| Ok(()))?;
    let (arbiter, key, metadata, witness) = metadata(source, 512 * 1024);
    let cached = Arc::new(PreparedShortHfModel::new(record, analysis, metadata)?);
    let current = cached.clone();
    drop(cached);
    assert!(witness.upgrade().is_some());
    assert_eq!(arbiter.snapshot(&key)?.reserved, 512 * 1024);
    drop(current);
    assert!(witness.upgrade().is_none());
    assert_eq!(arbiter.snapshot(&key)?.reserved, 0);
    Ok(())
}
