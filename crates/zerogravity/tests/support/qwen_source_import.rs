//! Offline import of the approved bytes through the original CacheStore.
use super::source::need;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity::cache::{
    CacheLimits, CacheOptions, CacheStore, CommitOutcome, FileRole, Manifest, ManifestEntry,
    ManifestSource, TransferAdmission,
};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity::{Error, Result};
pub fn import(label: &str, directory: &Path) -> Result<PathBuf> {
    let host = TransferAdmission {
        arbiter: ResourceArbiter::process_shared(),
        resource: PhysicalDeviceKey::new("host", "system-ram")?,
        owner: ResourceOwner::new("source-text-diagnostic", "approved-offline-import")?,
    };
    host.arbiter
        .register_matching(host.resource.clone(), 16 * 1024 * 1024 * 1024)?;
    let _control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let limits = CacheLimits::default();
    let manifest = match label {
        "2b" => Manifest::from_json(
            include_bytes!("../../src/qwen35_cuda/source2b.json"),
            &limits,
        )?,
        "08" => {
            let pin: serde_json::Value =
                serde_json::from_slice(include_bytes!("../../src/qwen35_cuda/source08.json"))
                    .map_err(|e| Error::Malformed(e.to_string()))?;
            let rows = pin["files"]
                .as_array()
                .ok_or_else(|| Error::Malformed("approved source inventory".into()))?;
            let mut entries = Vec::new();
            for row in rows {
                let name = row["name"]
                    .as_str()
                    .ok_or_else(|| Error::Malformed("source name".into()))?;
                let role = if name.ends_with(".safetensors") {
                    FileRole::Weights
                } else if name == "model.safetensors.index.json" {
                    FileRole::Index
                } else if name == "LICENSE" {
                    FileRole::License
                } else if matches!(
                    name,
                    "tokenizer.json" | "tokenizer_config.json" | "vocab.json" | "merges.txt"
                ) {
                    FileRole::Tokenizer
                } else if name.ends_with("config.json") {
                    FileRole::Config
                } else {
                    FileRole::Auxiliary
                };
                // No provenance upgrade: the legacy Git-blob inventory remains
                // the authority; the new cache entry reports Unspecified.
                entries.push(ManifestEntry::new(
                    name,
                    row["sha256"]
                        .as_str()
                        .ok_or_else(|| Error::Malformed("source digest".into()))?,
                    row["bytes"]
                        .as_u64()
                        .ok_or_else(|| Error::Malformed("source extent".into()))?,
                    role,
                )?);
            }
            Manifest::new(
                ManifestSource::new(
                    "huggingface",
                    "Qwen/Qwen3.5-0.8B",
                    "2fc06364715b967f1860aea9cf38778875588b17",
                )?,
                entries,
                &limits,
            )?
        }
        _ => return Err(Error::Unsupported("approved source import fixture".into())),
    };
    let digest = manifest.digest()?;
    let store = CacheStore::open(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let mut transaction = store.begin(manifest.clone(), &host, Arc::new(AtomicBool::new(false)))?;
    for entry in manifest.entries() {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(0o400000 | 0o2000000)
            .open(directory.join(entry.path()))?;
        need(
            file.metadata()?.is_file() && file.metadata()?.len() == entry.bytes(),
            "original transferred source member",
        )?;
        transaction.import_object(entry.path(), &mut file)?;
    }
    match transaction.commit(None)? {
        CommitOutcome::Durable(snapshot) => {
            need(snapshot.digest() == digest, "same durable source manifest")?
        }
        CommitOutcome::DurabilityUnknown { .. } => {
            return Err(Error::State(
                "source cache durability unknown; preserve import".into(),
            ));
        }
    }
    let snapshot = store.open_snapshot(&digest)?;
    need(
        snapshot.manifest() == &manifest,
        "genuine CacheStore-issued complete approved source",
    )?;
    let root = zerogravity::cache::paths::resolve_root(
        None,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?
    .join("snapshots")
    .join(&digest)
    .join("files");
    println!(
        "SOURCE_MODEL_IMPORTED model={label} snapshot={digest} files={} root={} new_download=false",
        manifest.entries().len(),
        root.display()
    );
    Ok(root)
}
