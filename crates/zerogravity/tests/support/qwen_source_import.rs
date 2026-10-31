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
    let manifest = manifest(label)?;
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

pub fn manifest(label: &str) -> Result<Manifest> {
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
    Ok(manifest)
}

pub fn canonical_root(label: &str) -> Result<PathBuf> {
    let digest = manifest(label)?.digest()?;
    Ok(zerogravity::cache::paths::resolve_root(
        None,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?
    .join("snapshots")
    .join(digest)
    .join("files"))
}
pub fn require_canonical(label: &str, directory: &Path) -> Result<()> {
    need(
        directory == canonical_root(label)?,
        "native source requires canonical CacheStore publication",
    )
}
pub fn open_native(
    label: &str,
    directory: &Path,
) -> Result<Arc<zerogravity::cache::SnapshotLease>> {
    require_canonical(label, directory)?;
    let manifest = manifest(label)?;
    let store = CacheStore::open(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let snapshot = store.open_snapshot(&manifest.digest()?)?;
    need(
        snapshot.manifest() == &manifest,
        "same canonical source manifest/provenance",
    )?;
    Ok(Arc::new(snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_source_paths_refuse_staging_and_another_snapshot() {
        for label in ["08", "2b"] {
            let root = canonical_root(label).unwrap();
            require_canonical(label, &root).unwrap();
            for bad in [
                PathBuf::from("/tmp/source-inputs"),
                root.join("../files"),
                root.parent().unwrap().to_path_buf(),
            ] {
                assert!(require_canonical(label, &bad).is_err());
            }
        }
        assert!(canonical_root("27b").is_err());
    }
    #[test]
    fn original_schema_and_legacy_provenance_survive_cache_conversion() {
        let small = manifest("08").unwrap();
        assert_eq!(
            small.digest().unwrap(),
            "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c"
        );
        assert_eq!(small.entries().len(), 11);
        assert!(small.entries().iter().all(|e| matches!(
            e.digest_provenance(),
            zerogravity::cache::DigestProvenance::Unspecified
        )));
        let two = manifest("2b").unwrap();
        assert_eq!(
            two.digest().unwrap(),
            "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52"
        );
        assert_eq!(two.entries().len(), 12);
    }
}
