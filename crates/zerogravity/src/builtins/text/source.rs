// Source-specific helpers moved from CLI serving.
use super::*;
use std::io::Read;
#[derive(PartialEq, Eq)]
struct FileStamp {
    len: u64,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    mtime: i64,
    #[cfg(unix)]
    mtime_nsec: i64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

fn file_stamp(meta: &std::fs::Metadata) -> Result<FileStamp, ServiceError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(FileStamp {
            len: meta.len(),
            dev: meta.dev(),
            ino: meta.ino(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
            ctime: meta.ctime(),
            ctime_nsec: meta.ctime_nsec(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(FileStamp {
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// Read the exact approved tokenizer extent from one held, no-follow
/// descriptor. The allocation is capped at the pinned length, the descriptor
/// must be at EOF immediately after it, and the pre/post descriptor stamp must
/// agree. SHA-256 is checked by `QwenTokenizer::from_approved_bytes` on these
/// same bytes before the parser runs.
pub fn read_approved_tokenizer(path: &Path) -> Result<Vec<u8>, ServiceError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW: do not authenticate a symlink target while claiming the
        // held path identity.
        options.custom_flags(0o400000);
    }
    let mut file = options
        .open(path)
        .map_err(|_| ServiceError::new(400, "missing_tokenizer", "tokenizer file unavailable"))?;
    let before = file.metadata().map_err(|_| {
        ServiceError::new(400, "missing_tokenizer", "tokenizer metadata unavailable")
    })?;
    if !before.is_file() {
        return Err(ServiceError::new(
            400,
            "missing_tokenizer",
            "tokenizer path is not a regular file",
        ));
    }
    let extent = before.len();
    if extent != crate::tokenize::APPROVED_TOKENIZER_BYTES {
        return Err(ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer extent mismatch",
        ));
    }
    let capacity = usize::try_from(extent)
        .map_err(|_| ServiceError::new(400, "tokenizer_inventory", "tokenizer extent overflow"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        ServiceError::new(
            503,
            "tokenizer_inventory",
            "approved tokenizer allocation unavailable",
        )
    })?;
    bytes.resize(capacity, 0);
    file.read_exact(&mut bytes).map_err(|_| {
        ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer extent changed during read",
        )
    })?;
    let mut probe = [0u8; 1];
    if file
        .read(&mut probe)
        .map_err(|_| ServiceError::new(400, "tokenizer_inventory", "tokenizer EOF probe failed"))?
        != 0
    {
        return Err(ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer extends past its extent",
        ));
    }
    let after = file.metadata().map_err(|_| {
        ServiceError::new(
            400,
            "tokenizer_inventory",
            "tokenizer metadata changed during read",
        )
    })?;
    if file_stamp(&before)? != file_stamp(&after)? || after.len() != extent {
        return Err(ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer stamp changed during read",
        ));
    }
    Ok(bytes)
}
