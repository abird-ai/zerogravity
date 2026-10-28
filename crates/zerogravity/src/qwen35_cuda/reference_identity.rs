//! Semantic identity of the authenticated historical CUDA reference.
//!
//! Staging receipts are audit provenance, not execution authority. The identity
//! used by parity evidence is recomputed from the exact held current manifest,
//! the model's capture recipe and the checked-in reference contract catalogue.
//! Bundle payloads are separately authenticated by the reference loader before
//! any evidence can be produced.

use super::{held_directory::Directory, *};
use crate::qwen35::plan::QwenSourceProfile;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicBool;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReferenceImplementationIdentity {
    pub(super) capture_recipe_sha256: String,
    pub(super) bundle_manifest_sha256: String,
    pub(super) reference_contract_sha256: String,
}

#[derive(Deserialize)]
struct ManifestHeader {
    schema: String,
}

fn current_reference_contract_sha256() -> String {
    format!(
        "{:x}",
        Sha256::digest(include_bytes!("bundle_migration_pins.json"))
    )
}

pub(super) fn inspect_held(
    manifest: &[u8],
    profile: QwenSourceProfile,
) -> Result<ReferenceImplementationIdentity> {
    let header: ManifestHeader = serde_json::from_slice(manifest).map_err(|error| {
        Error::Malformed(format!("Qwen CUDA native manifest identity: {error}"))
    })?;
    if header.schema != "zerogravity.cuda-kernels.v2" {
        return Err(Error::Unsupported(
            "reference identity requires current CUDA bundle schema".into(),
        ));
    }
    let capture = crate::qwen35::source::binding::ReferenceKernelCapture::for_profile(profile);
    Ok(ReferenceImplementationIdentity {
        capture_recipe_sha256: capture.recipe_sha256().into(),
        bundle_manifest_sha256: format!("{:x}", Sha256::digest(manifest)),
        reference_contract_sha256: current_reference_contract_sha256(),
    })
}

pub(super) fn inspect(
    directory: &Path,
    profile: QwenSourceProfile,
    cancelled: Option<&AtomicBool>,
) -> Result<ReferenceImplementationIdentity> {
    let root = Directory::open(directory)?;
    let manifest = root.bounded("cuda-kernels.json", 64 * 1024, cancelled)?;
    inspect_held(&manifest, profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_manifest_and_current_contract_define_reference_identity() {
        let first = br#"{"schema":"zerogravity.cuda-kernels.v2"}
"#;
        let second = br#"{"schema":"zerogravity.cuda-kernels.v2","changed":true}
"#;
        let a = inspect_held(first, QwenSourceProfile::Qwen08).unwrap();
        let again = inspect_held(first, QwenSourceProfile::Qwen08).unwrap();
        let changed = inspect_held(second, QwenSourceProfile::Qwen08).unwrap();
        assert_eq!(a, again);
        assert_ne!(a.bundle_manifest_sha256, changed.bundle_manifest_sha256);
        assert_eq!(
            a.reference_contract_sha256,
            current_reference_contract_sha256()
        );
        assert_eq!(a.reference_contract_sha256.len(), 64);
        assert!(
            a.reference_contract_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
        assert!(inspect_held(br#"{"schema":"wrong"}"#, QwenSourceProfile::Qwen08).is_err());
    }
}
