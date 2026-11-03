//! Descriptor-bound regular-file reads for Qwen CUDA qualification.
//!
//! Paths are resolved exactly once with NOFOLLOW. Type, extent, digest and
//! before/after inode metadata are checked on that same descriptor.

use crate::{Error, Result};
#[cfg(any(test, feature = "qwen35-cuda-qualification"))]
use rustix::fs::{Mode, OFlags, open, openat};
use sha2::{Digest, Sha256};
#[cfg(any(test, feature = "qwen35-cuda-qualification"))]
use std::path::{Component, Path};
use std::{
    fs::File,
    os::unix::fs::{FileExt, MetadataExt},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    device: u64,
    inode: u64,
    bytes: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}

fn stamp(metadata: &std::fs::Metadata) -> Stamp {
    Stamp {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        mtime: metadata.mtime(),
        mtime_ns: metadata.mtime_nsec(),
        ctime: metadata.ctime(),
        ctime_ns: metadata.ctime_nsec(),
    }
}

#[cfg(any(test, feature = "qwen35-cuda-qualification"))]
fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|error| std::io::Error::from(error).into())
}

fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Qwen CUDA held-file read cancelled".into()));
    }
    Ok(())
}

#[cfg(any(test, feature = "qwen35-cuda-qualification"))]
fn absolute_parts(path: &Path) -> Result<Vec<&std::ffi::OsStr>> {
    if !path.is_absolute() {
        return Err(Error::Malformed(
            "Qwen CUDA held path must be absolute".into(),
        ));
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(part) => parts.push(part),
            _ => {
                return Err(Error::Malformed(
                    "Qwen CUDA held path must be canonical absolute components".into(),
                ));
            }
        }
    }
    Ok(parts)
}

#[cfg(any(test, feature = "qwen35-cuda-qualification"))]
fn open_absolute(path: &Path, leaf: OFlags, allow_root: bool) -> Result<File> {
    let parts = absolute_parts(path)?;
    if parts.is_empty() && !allow_root {
        return Err(Error::Malformed(
            "Qwen CUDA held file requires a filename".into(),
        ));
    }
    let mut current: File = os(open(
        "/",
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    ))?
    .into();
    for (index, part) in parts.iter().enumerate() {
        let flags = if index + 1 == parts.len() {
            leaf
        } else {
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
        };
        current = os(openat(&current, *part, flags, Mode::empty()))?.into();
    }
    Ok(current)
}

#[cfg(feature = "qwen35-cuda-qualification")]
pub(crate) fn open_absolute_directory(path: &Path) -> Result<File> {
    open_absolute(
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        true,
    )
}

pub(crate) struct HeldFile {
    file: File,
    stamp: Stamp,
}

impl HeldFile {
    #[cfg(any(test, feature = "qwen35-cuda-qualification"))]
    pub(crate) fn open_absolute(path: &Path) -> Result<Self> {
        Self::from_file(open_absolute(
            path,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            false,
        )?)
    }

    pub(crate) fn from_file(file: File) -> Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Malformed(
                "Qwen CUDA held descriptor must be a regular file".into(),
            ));
        }
        Ok(Self {
            file,
            stamp: stamp(&metadata),
        })
    }

    #[cfg(feature = "qwen35-cuda-qualification")]
    pub(crate) const fn len(&self) -> u64 {
        self.stamp.bytes
    }

    #[cfg(feature = "qwen35-cuda-qualification")]
    pub(crate) fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }

    #[cfg(feature = "qwen35-cuda-qualification")]
    pub(crate) fn into_file(self) -> File {
        self.file
    }

    pub(crate) fn verify_unchanged(&self) -> Result<()> {
        if stamp(&self.file.metadata()?) != self.stamp {
            return Err(Error::State(
                "Qwen CUDA held descriptor identity changed".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn read_exact_bounded(
        self,
        expected: usize,
        maximum: usize,
        expected_sha256: Option<&[u8; 32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        if expected == 0 || expected > maximum || self.stamp.bytes != expected as u64 {
            return Err(Error::Limit(
                "Qwen CUDA held-file exact bounded extent differs".into(),
            ));
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(expected)
            .map_err(|_| Error::Limit("Qwen CUDA held-file allocation".into()))?;
        raw.resize(expected, 0);
        let mut offset = 0u64;
        for chunk in raw.chunks_mut(64 * 1024) {
            not_cancelled(cancelled)?;
            let mut filled = 0usize;
            while filled < chunk.len() {
                let count = self.file.read_at(
                    &mut chunk[filled..],
                    offset
                        .checked_add(filled as u64)
                        .ok_or_else(|| Error::Limit("Qwen CUDA held-file offset".into()))?,
                )?;
                if count == 0 {
                    return Err(Error::State(
                        "Qwen CUDA held file truncated during read".into(),
                    ));
                }
                filled += count;
            }
            offset = offset
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::Limit("Qwen CUDA held-file offset".into()))?;
        }
        let mut tail = [0u8; 1];
        if self.file.read_at(&mut tail, expected as u64)? != 0 {
            return Err(Error::State("Qwen CUDA held file grew during read".into()));
        }
        self.verify_unchanged()?;
        if expected_sha256.is_some_and(|expected| Sha256::digest(&raw)[..] != expected[..]) {
            return Err(Error::State("Qwen CUDA held-file SHA256 differs".into()));
        }
        not_cancelled(cancelled)?;
        Ok(raw)
    }

    pub(crate) fn read_bounded(
        self,
        maximum: usize,
        expected_sha256: Option<&[u8; 32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        let expected = usize::try_from(self.stamp.bytes)
            .map_err(|_| Error::Limit("Qwen CUDA held-file length".into()))?;
        self.read_exact_bounded(expected, maximum, expected_sha256, cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        os::unix::fs::symlink,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(1);

    fn root(label: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zg-held-file-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn absolute_open_refuses_symlink_and_detects_descriptor_mutation() {
        let root = root("nofollow");
        let file = root.join("evidence.bin");
        std::fs::write(&file, b"abcd").unwrap();
        let alias = root.join("alias.bin");
        symlink(&file, &alias).unwrap();
        assert!(HeldFile::open_absolute(&alias).is_err());

        let real_parent = root.join("real");
        std::fs::create_dir(&real_parent).unwrap();
        let nested = real_parent.join("nested.bin");
        std::fs::write(&nested, b"abcd").unwrap();
        let parent_alias = root.join("parent-alias");
        symlink(&real_parent, &parent_alias).unwrap();
        assert!(HeldFile::open_absolute(&parent_alias.join("nested.bin")).is_err());

        let held = HeldFile::open_absolute(&file).unwrap();
        std::fs::write(&file, b"wxyz").unwrap();
        assert!(held.verify_unchanged().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_read_uses_one_held_descriptor() {
        let root = root("bounded");
        let file = root.join("member.bin");
        std::fs::write(&file, b"exact").unwrap();
        let raw = HeldFile::open_absolute(&file)
            .unwrap()
            .read_bounded(5, Some(&Sha256::digest(b"exact").into()), None)
            .unwrap();
        assert_eq!(raw, b"exact");
        std::fs::remove_dir_all(root).unwrap();
    }
}
