//! Original descriptor capabilities, not paths reopened after authentication.
#[cfg(feature = "qwen35-hip")]
use super::Error;
#[cfg(feature = "qwen35-cuda-qualification")]
use super::source_inventory::digest;
use super::{Result, source_inventory::Spec};
use std::{fs::File, sync::atomic::AtomicBool};

pub(crate) enum Inputs<'a> {
    #[cfg(feature = "qwen35-cuda-qualification")]
    Directory(&'a super::held_directory::Directory),
    #[cfg(feature = "qwen35-hip")]
    Snapshot(&'a zerogravity_artifacts::cache::SnapshotLease),
}
impl Inputs<'_> {
    pub(super) fn file(&self, name: &str) -> Result<File> {
        match self {
            #[cfg(feature = "qwen35-cuda-qualification")]
            Self::Directory(dir) => dir.file(name),
            #[cfg(feature = "qwen35-hip")]
            Self::Snapshot(snapshot) => Ok(snapshot.file(name)?.try_clone()?),
        }
    }
    pub(super) fn bounded(
        &self,
        name: &str,
        cap: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "qwen35-cuda-qualification")]
            Self::Directory(dir) => dir.bounded(name, cap, cancelled),
            #[cfg(feature = "qwen35-hip")]
            Self::Snapshot(_) => super::held_file::HeldFile::from_file(self.file(name)?)?
                .read_bounded(cap, None, cancelled),
        }
    }
    pub(super) fn read(
        &self,
        name: &str,
        bytes: usize,
        sha: Option<&[u8; 32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "qwen35-cuda-qualification")]
            Self::Directory(dir) => dir.read(name, bytes, sha, cancelled),
            #[cfg(feature = "qwen35-hip")]
            Self::Snapshot(_) => super::held_file::HeldFile::from_file(self.file(name)?)?
                .read_exact_bounded(bytes, 32 * 1024 * 1024, sha, cancelled),
        }
    }
    pub(super) fn require_inventory(
        &self,
        spec: &Spec,
        members: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        super::not_cancelled(cancelled)?;
        match self {
            #[cfg(feature = "qwen35-cuda-qualification")]
            Self::Directory(dir) => {
                // Legacy receipt remains mandatory; a snapshot is a different
                // original capability, never a relaxation of this path.
                if spec.family == "0.8b" {
                    drop(dir.read(
                        "source-receipt.json",
                        spec.inventory.len(),
                        Some(&digest(spec.manifest_sha)),
                        cancelled,
                    )?);
                }
                let _ = members;
            }
            #[cfg(feature = "qwen35-hip")]
            Self::Snapshot(snapshot) => {
                let manifest = snapshot.manifest();
                if manifest.entries().len() != members
                    || manifest.digest()? != spec.snapshot_sha
                    || manifest.source().provider() != "huggingface"
                    || manifest.source().repository() != spec.repository
                    || manifest.source().revision() != spec.revision
                {
                    return Err(Error::State(
                        "canonical original source membership/provenance differs".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    #[cfg(feature = "qwen35-hip")]
    pub(super) fn retains_tokenizer(&self) -> bool {
        matches!(self, Self::Snapshot(_))
    }
}
