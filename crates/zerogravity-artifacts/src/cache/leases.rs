use super::{FileRole, Manifest, ManifestEntry, TransferAdmission};
use crate::SourceWeights;
use crate::artifact::{Artifact, copy_verified_file};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

/// Pins exact opened, read-only object descriptors and their verified manifest.
/// Ref replacement cannot retarget these handles. Each actual open description
/// holds a shared advisory pin: ordinary dup/try_clone and inherited child
/// descriptors retain it after this wrapper drops. Release follows final close,
/// never a wrapper-owned explicit unlock. Protected v2 maintenance observes
/// those pins; legacy v1 roots do not support reclamation.
/// A borrowed file supports positioned reads without shared cursor mutation.
#[derive(Debug)]
pub struct SnapshotLease {
    pub(crate) manifest: Manifest,
    pub(crate) files: BTreeMap<String, File>,
}

impl SnapshotLease {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn file(&self, logical_path: &str) -> Result<&File> {
        self.files
            .get(logical_path)
            .ok_or_else(|| Error::Malformed("file is absent from snapshot".into()))
    }

    fn entry(&self, logical_path: &str) -> Result<&ManifestEntry> {
        self.manifest
            .entries()
            .iter()
            .find(|entry| entry.path() == logical_path)
            .ok_or_else(|| Error::Malformed("file is absent from snapshot manifest".into()))
    }

    /// Retain this lease while hashing and reparsing the actual owned source
    /// copy from its original opened descriptor. The logical filename supplies
    /// the checked format hint; a digest-named blob path is never reopened.
    pub fn source_weights(
        &self,
        logical_path: &str,
        admission: &TransferAdmission,
    ) -> Result<SourceWeights> {
        let entry = self.entry(logical_path)?;
        if entry.role() != FileRole::Weights {
            return Err(Error::Unsupported(
                "cache source must be a weights entry".into(),
            ));
        }
        let parsed = Artifact::from_file(
            self.file(logical_path)?.try_clone()?,
            Path::new(logical_path),
        )?;
        SourceWeights::from_verified_artifact(parsed, entry.bytes(), entry.digest(), admission)
    }

    /// Bounded config/tokenizer/control copy verified against the manifest
    /// after copying, before a caller's existing parser sees any bytes.
    pub fn read_verified(&self, logical_path: &str, maximum: u64) -> Result<Vec<u8>> {
        let entry = self.entry(logical_path)?;
        copy_verified_file(
            self.file(logical_path)?,
            entry.bytes(),
            entry.digest(),
            maximum,
        )
    }
}
