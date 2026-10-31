//! Non-destructive advisory-pin observations using independent open descriptions.
use super::source::need;
use std::{
    fs::{File, OpenOptions, TryLockError},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
use zerogravity::Result;

pub struct Pins(Vec<(PathBuf, u64, u64)>);
fn open(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(0o400000 | 0o2000000 | 0o4000)
        .open(path)?)
}
impl Pins {
    pub fn new(label: &str, directory: &Path) -> Result<Self> {
        super::source_import::require_canonical(label, directory)?;
        let mut files = Vec::new();
        for member in super::source_import::manifest(label)?.entries() {
            let path = directory.join(member.path());
            let file = open(&path)?;
            let m = file.metadata()?;
            need(
                m.is_file() && m.len() == member.bytes(),
                "same canonical pin-probe member",
            )?;
            files.push((path, m.dev(), m.ino()));
        }
        Ok(Self(files))
    }
    pub fn require(&self, pinned: bool) -> Result<()> {
        for (path, dev, ino) in &self.0 {
            let file = open(path)?;
            let m = file.metadata()?;
            need(
                (m.dev(), m.ino()) == (*dev, *ino),
                "unchanged pin-probe descriptor binding",
            )?;
            let locked = match file.try_lock() {
                Ok(()) => false,
                Err(TryLockError::WouldBlock) => true,
                Err(TryLockError::Error(e)) => return Err(e.into()),
            };
            need(locked == pinned, "canonical source descriptor pin lifetime")?;
            // Close only this independent probe. Never unlock a reader's description.
            drop(file);
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}
