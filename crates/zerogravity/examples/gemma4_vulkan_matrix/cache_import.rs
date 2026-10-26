//! Publish the explicitly transferred SAME approved manifest through CacheStore.
use super::io::*;
use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity::cache::{
    CacheLimits, CacheOptions, CacheStore, CommitOutcome, Manifest, TransferAdmission,
};
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
const SNAPSHOT: &str = "ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1";
pub fn publish(stage: &Path) -> Proof<()> {
    need(stage.is_absolute(), "explicit transferred source stage")?;
    let _context = zerogravity::gemma4::execution::Context::new_vulkan_qualification()?;
    let _control = ResourceArbiter::process_shared().admit(
        &PhysicalDeviceKey::new("host", "system-ram")?,
        ResourceOwner::new("approved-source-transfer", "manifest-control")?,
        &[zerogravity::resource_arbiter::AllocationRequest {
            kind: zerogravity::resource_arbiter::AllocationClass::Scratch,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let raw = read(&stage.join("manifest.json"), 8192, None)?;
    let manifest = Manifest::from_json(&raw, &CacheLimits::default())?;
    need(
        manifest.digest()? == SNAPSHOT && manifest.entries().len() == 8,
        "exact original managed manifest/source/provenance",
    )?;
    let admission = TransferAdmission {
        arbiter: ResourceArbiter::process_shared(),
        resource: PhysicalDeviceKey::new("host", "system-ram")?,
        owner: ResourceOwner::new("approved-source-transfer", "Gemma4E2B-exact-manifest")?,
    };
    let store = CacheStore::open(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let mut tx = store.begin(
        manifest.clone(),
        &admission,
        Arc::new(AtomicBool::new(false)),
    )?;
    for entry in manifest.entries() {
        let mut file = open(&stage.join("files").join(entry.path()))?;
        let before = stamp(&file)?;
        need(
            before[2] == entry.bytes(),
            "same transferred original source extent",
        )?;
        tx.import_object(entry.path(), &mut file)?;
        need(
            stamp(&file)? == before,
            "same held imported source before/after",
        )?;
    }
    match tx.commit(None)? {
        CommitOutcome::Durable(snapshot) => need(
            snapshot.digest() == SNAPSHOT,
            "same durable canonical source snapshot",
        )?,
        CommitOutcome::DurabilityUnknown { .. } => {
            return Err("original source publication durability unknown; preserve and stop".into());
        }
    }
    let held = store.open_snapshot(SNAPSHOT)?;
    need(
        held.manifest().digest()? == SNAPSHOT,
        "actual CacheStore-issued verified SnapshotLease",
    )?;
    println!(
        "GEMMA_VULKAN_EXACT_SOURCE_IMPORTED snapshot={SNAPSHOT} files=8 full_payload_SHA=verified canonical_cache=true new_download=false"
    );
    Ok(())
}
