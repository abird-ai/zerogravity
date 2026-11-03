//! Held directory capability shared by Qwen CUDA source loaders.
//! Symlink traversal and mutable-path reopening are refused.

use super::held_file::{HeldFile, open_absolute_directory};
use crate::{Error, Result};
use rustix::fs::{Mode, OFlags, openat};
use std::{fs::File, path::Path, sync::atomic::AtomicBool};

fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|error| std::io::Error::from(error).into())
}

pub(crate) struct Directory(File);
impl Directory {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        Ok(Self(open_absolute_directory(path)?))
    }

    fn held(&self, name: &str) -> Result<HeldFile> {
        zerogravity_artifacts::cache::paths::validate_relative_path(name)?;
        let mut directory = self.0.try_clone()?;
        let parts = name.split('/').collect::<Vec<_>>();
        if parts.is_empty() {
            return Err(Error::Malformed("Qwen CUDA source member path".into()));
        }
        for part in &parts[..parts.len() - 1] {
            directory = os(openat(
                &directory,
                *part,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            ))?
            .into();
        }
        let file: File = os(openat(
            &directory,
            parts[parts.len() - 1],
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
        ))?
        .into();
        HeldFile::from_file(file)
    }

    pub(crate) fn file(&self, name: &str) -> Result<File> {
        Ok(self.held(name)?.into_file())
    }

    pub(crate) fn read(
        &self,
        name: &str,
        expected: usize,
        digest: Option<&[u8; 32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        self.held(name)?
            .read_exact_bounded(expected, 32 * 1024 * 1024, digest, cancelled)
    }

    pub(crate) fn bounded(
        &self,
        name: &str,
        cap: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        self.held(name)?.read_bounded(cap, None, cancelled)
    }
}
