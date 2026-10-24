//! Held directory capabilities; no symlink traversal or mutable-path reopening.
use super::*;
use rustix::fs::{Mode, OFlags, open, openat};
use sha2::{Digest, Sha256};
use std::{ffi::OsStr, fs::File, io::Read, os::unix::fs::MetadataExt};

pub(crate) struct Directory(File);
fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|e| std::io::Error::from(e).into())
}
impl Directory {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute() {
            return Err(Error::Malformed(
                "canonical47 directory must be absolute".into(),
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
            return Err(Error::Malformed("canonical47 regular file required".into()));
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
                "canonical47 exact bounded immutable file".into(),
            ));
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(expected)
            .map_err(|_| Error::Limit("canonical47 file allocation".into()))?;
        raw.resize(expected, 0);
        for chunk in raw.chunks_mut(64 * 1024) {
            not_cancelled(cancelled)?;
            file.read_exact(chunk)?;
        }
        let mut tail = [0];
        if file.read(&mut tail)? != 0 {
            return Err(Error::State("canonical47 file grew during read".into()));
        }
        let after = file.metadata()?;
        let identity = |m: &std::fs::Metadata| {
            (
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            )
        };
        if identity(&before) != identity(&after)
            || digest.is_some_and(|sha| Sha256::digest(&raw)[..] != sha[..])
            || raw.capacity() > expected
        {
            return Err(Error::State(
                "canonical47 complete held-file SHA/identity/capacity differs".into(),
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
            .map_err(|_| Error::Limit("canonical47 file length".into()))?;
        if length == 0 || length > cap {
            return Err(Error::Limit("canonical47 metadata bound".into()));
        }
        self.read(name, length, None, cancelled)
    }
}
pub(crate) fn digest(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64, "compiled SHA256 literal");
    let mut out = [0; 32];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let digit = |c: u8| match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("compiled SHA256 literal"),
        };
        out[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    out
}
pub(crate) fn native_root(
    explicit: Option<&Path>,
    xdg: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.is_absolute() {
            return Err(Error::Malformed(
                "native artifacts root must be absolute".into(),
            ));
        }
        return Ok(path.to_owned());
    }
    let base = if let Some(path) = xdg.map(Path::new).filter(|p| p.is_absolute()) {
        path.to_owned()
    } else {
        home.map(Path::new)
            .filter(|p| p.is_absolute())
            .ok_or_else(|| Error::State("absolute HOME required for native artifact cache".into()))?
            .join(".cache")
    };
    Ok(base.join("abird/zerogravity/native/canonical47-v1"))
}
