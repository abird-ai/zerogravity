//! Atomic, optimistic document replacement. A short parent-directory lock
//! serializes our writers; an editor never holds the lock while open.
use super::*;
use std::{
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Write},
    sync::atomic::{AtomicU64, Ordering},
};

pub struct ConfigFile {
    path: PathBuf,
    original: Option<Vec<u8>>,
}
impl ConfigFile {
    pub fn open(path: PathBuf) -> Result<Self> {
        if !path.is_absolute() || path.file_name().is_none() {
            return Err(bad("configuration file requires an absolute path"));
        }
        let original = read_optional(&path)?;
        Ok(Self { path, original })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn exists(&self) -> bool {
        self.original.is_some()
    }
    pub fn bytes(&self) -> &[u8] {
        self.original.as_deref().unwrap_or(INITIAL.as_bytes())
    }

    pub fn require_existing(&self) -> Result<()> {
        if self.exists() {
            Ok(())
        } else {
            Err(Error::Unsupported(
                "selected configuration file does not exist".into(),
            ))
        }
    }
    pub fn change(&self, edit: Change<'_>) -> Result<()> {
        self.replace(&changed(&self.path, self.bytes(), edit)?)
    }
    pub fn create(&self, bytes: &[u8]) -> Result<()> {
        if self.exists() {
            return Err(Error::State("configuration already exists".into()));
        }
        self.replace(bytes)
    }
    /// Validate before creating directories or staging. Compare the exact
    /// original bytes under the cooperative lock to refuse stale clients.
    pub fn replace(&self, bytes: &[u8]) -> Result<()> {
        validate(&self.path, bytes)?;
        let parent = self.path.parent().expect("absolute file parent");
        fs::create_dir_all(parent)?;
        let lock = lock_parent(parent)?;
        self.require_current()?;
        if self.original.as_deref() == Some(bytes) {
            return Ok(());
        }
        let staged = Temporary::new(parent, bytes)?;
        if self.exists() {
            fs::rename(&staged.path, &self.path)?;
        } else {
            // Atomic no-clobber create, even if another writer ignores our lock.
            fs::hard_link(&staged.path, &self.path)?;
            fs::remove_file(&staged.path).map_err(|error| {
                Error::State(format!(
                    "configuration created but staging cleanup failed: {error}"
                ))
            })?;
        }
        sync_parent(&lock)?;
        Ok(())
    }

    pub fn remove(&self) -> Result<()> {
        self.require_existing()?;
        let lock = lock_parent(self.path.parent().expect("absolute file parent"))?;
        self.require_current()?;
        fs::remove_file(&self.path)?;
        sync_parent(&lock)
    }

    /// Caller owns editor invocation; all frontends use the same final
    /// replacement. Invalid, canceled and stale drafts leave old bytes intact.
    pub fn edit(&self, editor: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
        let staged = Temporary::new(&std::env::temp_dir(), self.bytes())?;
        editor(&staged.path)?;
        let candidate = read_optional(&staged.path)?
            .ok_or_else(|| bad("editor removed configuration draft"))?;
        self.replace(&candidate)
    }

    fn require_current(&self) -> Result<()> {
        if read_optional(&self.path)? != self.original {
            return Err(Error::State(
                "configuration changed since it was read; reload before saving".into(),
            ));
        }
        Ok(())
    }
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_file() => pipelines::read_document_bytes(path).map(Some),
        Ok(_) => Err(bad(
            "configuration management requires a regular file, not a symlink",
        )),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn lock_parent(parent: &Path) -> Result<File> {
    // Unix flock can lock an open directory; no extra persistent file.
    #[cfg(unix)]
    let file = File::open(parent)?;
    #[cfg(not(unix))]
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(parent.join(".config-lock"))?;
    file.lock()?;
    Ok(file)
}

fn sync_parent(lock: &File) -> Result<()> {
    #[cfg(unix)]
    lock.sync_all().map_err(|error| {
        Error::State(format!(
            "configuration publication occurred but directory durability is unknown: {error}"
        ))
    })?;
    #[cfg(not(unix))]
    let _ = lock;
    Ok(())
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temporary {
    path: PathBuf,
}
impl Temporary {
    fn new(parent: &Path, bytes: &[u8]) -> Result<Self> {
        for _ in 0..64 {
            let path = parent.join(format!(
                ".zg-config-{}-{}.yaml",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(mut file) => {
                    let staged = Self { path };
                    file.write_all(bytes)?;
                    file.sync_all()?;
                    return Ok(staged);
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(Error::State("configuration staging name exhausted".into()))
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
