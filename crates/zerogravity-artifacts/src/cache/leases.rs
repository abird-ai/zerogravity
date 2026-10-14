use super::Manifest;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fs::File;

/// Pins exact opened, read-only object descriptors and their verified manifest.
/// Ref replacement cannot retarget these handles. No eviction exists in A1.
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
}
