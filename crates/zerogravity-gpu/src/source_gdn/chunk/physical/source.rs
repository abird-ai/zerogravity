//! Synthetic source anchor for operator ownership, never trained-model evidence.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    path::{Path, PathBuf},
};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};
use zerogravity_core::resource_arbiter::{
    GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

pub(super) struct Anchor {
    path: PathBuf,
    _control: GlobalResourceLease,
}
impl Drop for Anchor {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
pub(super) fn open() -> Result<(Arc<Anchor>, VerifiedTensorSource)> {
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", "system-ram")?;
    arbiter.register_matching(
        key.clone(),
        crate::hip::native::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES,
    )?;
    let owner = ResourceOwner::new("source-gdn-chunk-operator", "synthetic-source-anchor")?;
    let control = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 64 * 1024,
        }],
    )?;
    let path = std::env::temp_dir().join(format!("zg-gdn-chunk-anchor-{}", std::process::id()));
    std::fs::create_dir(&path)?;
    let anchor = Arc::new(Anchor {
        path,
        _control: control,
    });
    let header = br#"{"anchor":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    let filename = "anchor.safetensors";
    std::fs::write(anchor.path.join(filename), &bytes)?;
    let shard = ExpectedSourceShard::new(
        filename.into(),
        Artifact::from_file(File::open(anchor.path.join(filename))?, Path::new(filename))?,
        bytes.len() as u64,
        format!("{:x}", Sha256::digest(&bytes)),
    )?;
    let index = br#"{"metadata":{"total_size":4},"weight_map":{"anchor":"anchor.safetensors"}}"#;
    let source = SourceWeights::prepare_verified_windows(
        index,
        &format!("{:x}", Sha256::digest(index)),
        vec![shard],
        WindowOpenLimits {
            max_source_bytes: 1024,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 1,
            scan_chunk_bytes: 1024,
        },
        &TransferAdmission {
            arbiter,
            resource: key,
            owner,
        },
    )?
    .open()?;
    Ok((anchor, source))
}
