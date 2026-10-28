//! Held directory capability shared by Qwen CUDA source loaders.
//! Symlink traversal and mutable-path reopening are refused.

use super::*;
use rustix::fs::{Mode, OFlags, open, openat};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    os::unix::fs::MetadataExt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const MIB: usize = 1024 * 1024;

fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Qwen CUDA source load cancelled".into()));
    }
    Ok(())
}

fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|error| std::io::Error::from(error).into())
}

pub(crate) struct Directory(File);
impl Directory {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute() {
            return Err(Error::Malformed(
                "Qwen CUDA source directory must be absolute".into(),
            ));
        }
        Ok(Self(
            os(open(
                path,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            ))?
            .into(),
        ))
    }

    pub(crate) fn file(&self, name: &str) -> Result<File> {
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
        if !file.metadata()?.is_file() {
            return Err(Error::Malformed(
                "Qwen CUDA regular source file required".into(),
            ));
        }
        Ok(file)
    }

    pub(crate) fn read(
        &self,
        name: &str,
        expected: usize,
        digest: Option<&[u8; 32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        let mut file = self.file(name)?;
        let before = file.metadata()?;
        if before.len() != expected as u64 || expected == 0 || expected > 32 * MIB {
            return Err(Error::Limit(
                "Qwen CUDA exact bounded immutable file".into(),
            ));
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(expected)
            .map_err(|_| Error::Limit("Qwen CUDA source file allocation".into()))?;
        raw.resize(expected, 0);
        for chunk in raw.chunks_mut(64 * 1024) {
            not_cancelled(cancelled)?;
            file.read_exact(chunk)?;
        }
        let mut tail = [0];
        if file.read(&mut tail)? != 0 {
            return Err(Error::State(
                "Qwen CUDA source file grew during read".into(),
            ));
        }
        let after = file.metadata()?;
        let identity = |metadata: &std::fs::Metadata| {
            (
                metadata.dev(),
                metadata.ino(),
                metadata.len(),
                metadata.mtime(),
                metadata.mtime_nsec(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            )
        };
        if identity(&before) != identity(&after)
            || digest.is_some_and(|sha| Sha256::digest(&raw)[..] != sha[..])
            || raw.capacity() > expected
        {
            return Err(Error::State(
                "Qwen CUDA complete held-file SHA/identity/capacity differs".into(),
            ));
        }
        not_cancelled(cancelled)?;
        Ok(raw)
    }

    pub(crate) fn bounded(
        &self,
        name: &str,
        cap: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        let length = usize::try_from(self.file(name)?.metadata()?.len())
            .map_err(|_| Error::Limit("Qwen CUDA source file length".into()))?;
        if length == 0 || length > cap {
            return Err(Error::Limit("Qwen CUDA metadata bound".into()));
        }
        self.read(name, length, None, cancelled)
    }
}
