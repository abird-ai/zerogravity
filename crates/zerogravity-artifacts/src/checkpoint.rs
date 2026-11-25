//! Explicit cold checkpoint file IO. Integrity is not authentication.
//! The caller owns compatibility, allocation admission and state publication.

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// One bounded streaming buffer and fixed control/hash scratch. The model
/// owner admits this before starting IO; payload allocations are separate.
pub const IO_WORKSPACE_BYTES: u64 = 128 * 1024;
const BUFFER_BYTES: usize = 64 * 1024;
const DIGEST_BYTES: u64 = 32;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct Temp(Option<PathBuf>);
impl Drop for Temp {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = fs::remove_file(path);
        }
    }
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

/// Publish a complete file by same-directory rename. Unique private staging
/// never truncates another writer's file (or a caller's existing .part file).
/// Last successful rename wins. A failed directory sync after rename reports
/// an error, not a guarantee that the prior file survived.
pub fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    if path.file_name().is_none() {
        return Err(Error::Malformed(
            "checkpoint destination requires a filename".into(),
        ));
    }
    let directory = parent(path);
    let (temp, file) = loop {
        let id = NEXT_TEMP
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit("checkpoint temporary IDs exhausted".into()))?;
        let candidate = directory.join(format!(".zg-checkpoint-{}-{id}.part", std::process::id()));
        // A caller can legally choose any filename, including a staging name.
        if candidate.file_name() == path.file_name() {
            continue;
        }
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => break (Temp(Some(candidate)), file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    let mut writer = BufWriter::with_capacity(BUFFER_BYTES, file);
    write(&mut writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    let mut temp = temp;
    fs::rename(temp.0.as_ref().expect("owned staging file"), path)?;
    temp.0 = None;
    #[cfg(unix)]
    File::open(directory)?.sync_all()?;
    Ok(())
}

/// Held regular-file descriptor, with no symlink/FIFO traversal on Linux.
pub fn open_regular(path: &Path) -> Result<File> {
    #[cfg(target_os = "linux")]
    let file = File::from(
        rustix::fs::open(
            path,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map_err(io::Error::from)?,
    );
    #[cfg(not(target_os = "linux"))]
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::Malformed("checkpoint is not a regular file".into()));
    }
    Ok(file)
}

struct HashWriter<'a> {
    writer: &'a mut BufWriter<File>,
    hash: Sha256,
    remaining: u64,
}
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint exceeds declared length",
            ));
        }
        let n = self.writer.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.remaining -= n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// Write exactly the caller's bounded payload plus its SHA256 trailer, using
/// the same atomic publication primitive as other checkpoint formats.
pub fn write_checked(
    path: &Path,
    payload_bytes: u64,
    write: impl FnOnce(&mut dyn Write) -> Result<()>,
) -> Result<()> {
    payload_bytes
        .checked_add(DIGEST_BYTES)
        .ok_or_else(|| Error::Limit("checkpoint length overflow".into()))?;
    atomic_write(path, |writer| {
        let mut hashing = HashWriter {
            writer,
            hash: Sha256::new(),
            remaining: payload_bytes,
        };
        write(&mut hashing)?;
        if hashing.remaining != 0 {
            return Err(Error::Malformed(
                "checkpoint shorter than declared length".into(),
            ));
        }
        let digest = hashing.hash.finalize();
        writer.write_all(&digest)?;
        Ok(())
    })
}

struct HashReader<'a> {
    reader: io::Take<&'a mut BufReader<File>>,
    hash: Sha256,
}
impl Read for HashReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.reader.read(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
}

/// Decode privately from one bounded descriptor; return the candidate only
/// after every declared byte and the integrity trailer have been verified.
/// A digest detects corruption, not an attacker who can rewrite the file.
pub fn read_checked<T>(
    path: &Path,
    max_payload_bytes: u64,
    decode: impl FnOnce(&mut dyn Read, u64) -> Result<T>,
) -> Result<T> {
    let file = open_regular(path)?;
    let payload_bytes = file
        .metadata()?
        .len()
        .checked_sub(DIGEST_BYTES)
        .ok_or_else(|| Error::Malformed("checkpoint integrity trailer absent".into()))?;
    if payload_bytes > max_payload_bytes {
        return Err(Error::Limit("checkpoint exceeds file budget".into()));
    }
    let mut reader = BufReader::with_capacity(BUFFER_BYTES, file);
    let mut hashing = HashReader {
        reader: (&mut reader).take(payload_bytes),
        hash: Sha256::new(),
    };
    let candidate = decode(&mut hashing, payload_bytes)?;
    if hashing.reader.limit() != 0 {
        return Err(Error::Malformed("checkpoint payload extent differs".into()));
    }
    let actual = hashing.hash.finalize();
    let mut expected = [0u8; DIGEST_BYTES as usize];
    reader.read_exact(&mut expected)?;
    let mut extra = [0u8; 1];
    if actual.as_slice() != expected || reader.read(&mut extra)? != 0 {
        return Err(Error::Malformed(
            "checkpoint integrity/extent mismatch".into(),
        ));
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn atomic_checked_io_preserves_destination_and_rejects_corruption() {
        let dir = std::env::temp_dir().join(format!("zg-checkpoint-io-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());
        let path = dir.join("state");
        let legacy_part = path.with_extension("part");
        fs::write(&legacy_part, b"another owner's file").unwrap();
        let read = || {
            read_checked(&path, 8, |reader, len| {
                let mut bytes = vec![0; len as usize];
                reader.read_exact(&mut bytes)?;
                Ok(bytes)
            })
        };
        write_checked(&path, 4, |w| {
            w.write_all(b"root")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(read().unwrap(), b"root");
        assert!(
            write_checked(&path, 4, |w| {
                w.write_all(b"bad")?;
                Ok(())
            })
            .is_err()
        );
        assert!(
            write_checked(&path, 4, |w| {
                w.write_all(b"oversize")?;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(read().unwrap(), b"root");
        assert_eq!(fs::read(&legacy_part).unwrap(), b"another owner's file");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        let mut file = fs::read(&path).unwrap();
        file[0] ^= 1;
        fs::write(&path, &file).unwrap();
        assert!(read().is_err());
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::symlink;
            let alias = dir.join("alias");
            symlink(&path, &alias).unwrap();
            assert!(open_regular(&alias).is_err());
        }
    }
}
