//! Unverified derived-artifact lineage. Never substitutes an original source.
use zerogravity_core::{encoding::StoredTensorEncoding, identity::ArtifactDigest};

/// Lossy conversion requires independent source, recipe and held-out quality checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedModelManifest {
    pub original: ArtifactDigest,
    pub derived: ArtifactDigest,
    pub quantizer_recipe: ArtifactDigest,
    pub calibration: ArtifactDigest,
    pub held_out_quality: ArtifactDigest,
    pub encoding: StoredTensorEncoding,
}
