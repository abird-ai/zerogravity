//! Read-only bounded selection metadata. This grants neither verified source
//! ownership nor model readiness; execution must open the exact snapshot again.
use super::*;
use std::path::{Path, PathBuf};

pub struct CacheMetadata {
    root: RootDir,
    path: PathBuf,
    limits: super::super::CacheLimits,
}
pub struct SnapshotMetadata {
    pub digest: String,
    pub manifest: Manifest,
}
impl CacheMetadata {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            root: RootDir::open(path, false)?,
            path: path.into(),
            limits: super::super::CacheLimits::default(),
        })
    }
    /// Read-only immutable selection shared by model-family preparation paths.
    /// No payload verification, request, cache lock or model authority is issued.
    pub fn select(
        &self,
        selector: &zerogravity_core::run_policy::SourceSelector,
        requested: Option<&str>,
    ) -> Result<SnapshotMetadata> {
        use zerogravity_core::run_policy::SourceSelector;
        let matches = |snapshot: &SnapshotMetadata| -> Result<bool> {
            let source = snapshot.manifest.source();
            match selector {
                SourceSelector::Catalog(name) => {
                    Ok(self.reference(name)?.digest == snapshot.digest)
                }
                SourceSelector::Name(name) => Ok(self
                    .named(name)?
                    .is_some_and(|selected| selected.digest == snapshot.digest)),
                SourceSelector::Snapshot(digest) => Ok(digest == &snapshot.digest),
                SourceSelector::ByName(name) => {
                    Ok(self.human_reference(name)?.digest == snapshot.digest)
                }
                SourceSelector::HuggingFace(value) | SourceSelector::GitHub(value) => {
                    let provider = if matches!(selector, SourceSelector::HuggingFace(_)) {
                        "huggingface"
                    } else {
                        "github"
                    };
                    Ok(
                        crate::cache::provenance::ProviderLocator::parse(provider, value)?
                            .matches(source),
                    )
                }
                SourceSelector::Http(url) => {
                    #[cfg(feature = "acquisition")]
                    {
                        let weights = snapshot
                            .manifest
                            .entries()
                            .iter()
                            .filter(|e| e.role() == FileRole::Weights)
                            .collect::<Vec<_>>();
                        if weights.len() != 1 {
                            return Ok(false);
                        }
                        let e = weights[0];
                        let spec = crate::acquisition::object::HttpsObjectSpec::cached_locator(
                            url,
                            e.path(),
                            e.role(),
                            e.bytes(),
                            e.digest(),
                        )?;
                        Ok(source == spec.source())
                    }
                    #[cfg(not(feature = "acquisition"))]
                    {
                        let _ = url;
                        Err(Error::Unsupported("HTTP source identity requires compiled acquisition contract parser; no request is made".into()))
                    }
                }
                SourceSelector::Local(_) => Err(Error::Malformed(
                    "local path is not a cached selector".into(),
                )),
            }
        };
        let snapshot = if let Some(d) = requested {
            let snapshot = self.snapshot(d)?;
            if !matches(&snapshot)? {
                return Err(Error::Malformed(
                    "explicit snapshot does not match requested provider/content identity".into(),
                ));
            }
            snapshot
        } else {
            match selector {
                SourceSelector::Catalog(name) => self.reference(name)?,
                SourceSelector::Name(name) => self.named(name)?.ok_or_else(|| {
                    Error::Unsupported(
                        "model name is not acquired; use model list or an explicit model pull"
                            .into(),
                    )
                })?,
                SourceSelector::Snapshot(digest) => self.snapshot(digest)?,
                SourceSelector::ByName(name) => self.human_reference(name)?,
                SourceSelector::Local(_) => {
                    return Err(Error::Malformed(
                        "local path is not a cached selector".into(),
                    ));
                }
                SourceSelector::Http(_) => {
                    return Err(Error::Malformed(
                    "HTTP locator requires an explicit snapshot binding an exact cached content contract".into(),
                ));
                }
                SourceSelector::HuggingFace(_) | SourceSelector::GitHub(_) => self.provider_optional(selector)?.ok_or_else(|| Error::Unsupported(
                    "provider source is not acquired; use models/pull then an explicit --snapshot or catalog reference".into()))?,
            }
        };
        Ok(snapshot)
    }
    /// Exact provider candidate lookup, with absence distinct from
    /// ambiguity and invalid metadata. Never performs a remote lookup.
    pub fn provider_optional(
        &self,
        selector: &zerogravity_core::run_policy::SourceSelector,
    ) -> Result<Option<SnapshotMetadata>> {
        use zerogravity_core::run_policy::SourceSelector;
        let (provider, value) = match selector {
            SourceSelector::HuggingFace(value) => ("huggingface", value),
            SourceSelector::GitHub(value) => ("github", value),
            _ => return Err(Error::Malformed("expected provider source selector".into())),
        };
        let locator = crate::cache::provenance::ProviderLocator::parse(provider, value)?;
        let mut candidates = self
            .snapshots()?
            .into_iter()
            .filter(|s| locator.matches(s.manifest.source()))
            .collect::<Vec<_>>();
        match candidates.len() {
            0 => Ok(None),
            1 => Ok(candidates.pop()),
            n => Err(Error::Unsupported(format!(
                "provider source matches {n} acquired snapshots; an explicit --snapshot or catalog reference is required"
            ))),
        }
    }

    pub fn root(&self) -> &Path {
        &self.path
    }
    pub(super) fn read(&self, path: &str, limit: usize) -> Result<Vec<u8>> {
        let mut file = self.root.open_file(path, OFlags::RDONLY, Mode::empty())?;
        let before = file.metadata()?;
        if before.len() > limit as u64 {
            return Err(Error::Limit("cache selection metadata bytes".into()));
        }
        let mut raw = Vec::with_capacity(before.len() as usize);
        (&mut file).take(limit as u64 + 1).read_to_end(&mut raw)?;
        let after = file.metadata()?;
        let stamp = |m: &std::fs::Metadata| {
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
        if raw.len() as u64 != before.len() || stamp(&before) != stamp(&after) {
            return Err(Error::State("cache selection metadata changed".into()));
        }
        Ok(raw)
    }
    pub(super) fn directory_absent(&self, path: &str) -> Result<bool> {
        match self.root.open_dir(path, false) {
            Ok(_) => Ok(false),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
            Err(error) => Err(error),
        }
    }

    /// Only an absent snapshot directory is missing. A dangling or corrupt
    /// manifest in a present directory must not authorize acquisition.
    pub fn snapshot_optional(&self, digest: &str) -> Result<Option<SnapshotMetadata>> {
        validate_digest(digest)?;
        match self.root.open_dir(&format!("snapshots/{digest}"), false) {
            Ok(_) => self.snapshot(digest).map(Some),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub fn snapshot(&self, digest: &str) -> Result<SnapshotMetadata> {
        validate_digest(digest)?;
        let manifest = Manifest::from_json(
            &self.read(
                &format!("snapshots/{digest}/manifest.json"),
                self.limits.max_manifest_bytes,
            )?,
            &self.limits,
        )?;
        if manifest.digest()? != digest {
            return Err(Error::Malformed("cache selection manifest identity".into()));
        }
        Ok(SnapshotMetadata {
            digest: digest.into(),
            manifest,
        })
    }
    pub fn reference(&self, name: &str) -> Result<SnapshotMetadata> {
        self.reference_optional(name)?.ok_or_else(|| Error::Unsupported(format!(
            "catalog reference {name} is not acquired; use models/pull then run the printed reference"
        )))
    }

    fn reference_optional(&self, name: &str) -> Result<Option<SnapshotMetadata>> {
        let raw = match self.read(&reference_key(name)?, CONTROL_LIMIT * 2) {
            Ok(raw) => raw,
            // Only missing reference records are absence, never broken bindings.
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let reference: Reference = serde_json::from_slice(&raw)
            .map_err(|_| Error::Malformed("invalid cache selection reference".into()))?;
        if reference.schema != "zerogravity.cache.reference.v1" || reference.name != name {
            return Err(Error::Malformed(
                "cache selection reference identity".into(),
            ));
        }
        self.snapshot(&reference.snapshot).map(Some)
    }

    /// A bare name may denote either existing durable claim. If both are
    /// present they must agree. No projection following, repair or provider
    /// lookup. Explicit catalog:/by-name: remain independently selectable.
    pub fn named(&self, name: &str) -> Result<Option<SnapshotMetadata>> {
        if !zerogravity_core::run_policy::SourceSelector::is_human_name(name) {
            return Err(Error::Malformed("invalid bare model name".into()));
        }
        let canonical = self.reference_optional(name)?;
        let human = self.human_reference_optional(name)?;
        match (canonical, human) {
            (Some(a), Some(b)) if a.digest != b.digest => Err(Error::Malformed(
                "ambiguous bare model name; use catalog:, by-name: or an exact snapshot digest"
                    .into(),
            )),
            (Some(a), _) | (_, Some(a)) => Ok(Some(a)),
            (None, None) => Ok(None),
        }
    }

    /// Bounded cold inventory of existing checkpoint records. No lock
    /// creation, prefix authentication, active/dead determination or repair.
    pub fn partials(&self) -> Result<Vec<PartialSummary>> {
        use super::partials::{PARTIAL_LIMIT, partial_record_path, summarize_record};
        let mut result = Vec::new();
        for node in self.root.walk_canonical_tree(MAX_NODES)? {
            if !node.is_directory && partial_record_path(&node.path) {
                if result.len() == self.limits.max_files {
                    return Err(Error::Limit("partial checkpoint record count".into()));
                }
                result.push(summarize_record(
                    &self.read(&node.path, PARTIAL_LIMIT)?,
                    &node.path,
                )?);
            }
        }
        result.sort_unstable_by(|a, b| a.key.cmp(&b.key).then(a.bytes.cmp(&b.bytes)));
        Ok(result)
    }

    pub fn snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        let mut result = Vec::new();
        let mut bytes = 0usize;
        for node in self.root.walk_canonical_tree(MAX_NODES)? {
            if let Some(d) = node.path.strip_prefix("snapshots/")
                && node.is_directory
                && !d.contains('/')
            {
                if result.len() == 64 {
                    return Err(Error::Limit(
                        "cache selection at most64 snapshots; specify --snapshot".into(),
                    ));
                }
                let snapshot = self.snapshot(d)?;
                bytes = bytes
                    .checked_add(snapshot.manifest.canonical_bytes()?.len())
                    .ok_or_else(|| Error::Limit("cache selection bytes".into()))?;
                if bytes > 16 * 1024 * 1024 {
                    return Err(Error::Limit("cache selection metadata total".into()));
                }
                result.push(snapshot);
            }
        }
        result.sort_by(|a, b| a.digest.cmp(&b.digest));
        Ok(result)
    }
    /// Only bounded source-geometry metadata may be read during routing. Raw
    /// weights and tokenizer parsing remain in the original preparation path.
    pub fn config(&self, snapshot: &SnapshotMetadata, path: &str) -> Result<Vec<u8>> {
        let entry = snapshot
            .manifest
            .entries()
            .iter()
            .find(|e| e.path() == path && e.role() == FileRole::Config)
            .ok_or_else(|| Error::Malformed("cache config role/path".into()))?;
        if entry.bytes() > 1024 * 1024 {
            return Err(Error::Limit("cache routing config1MiB".into()));
        }
        let raw = self.read(
            &format!("snapshots/{}/files/{path}", snapshot.digest),
            entry.bytes() as usize,
        )?;
        if raw.len() as u64 != entry.bytes() || digest(&raw) != entry.digest() {
            return Err(Error::Malformed("cache routing config digest".into()));
        }
        Ok(raw)
    }
    pub fn inspect_weights(
        &self,
        snapshot: &SnapshotMetadata,
        path: &str,
    ) -> Result<crate::artifact::ArtifactInfo> {
        let entry = snapshot
            .manifest
            .entries()
            .iter()
            .find(|e| e.path() == path && e.role() == FileRole::Weights)
            .ok_or_else(|| Error::Malformed("cache weights role/path".into()))?;
        let file = self.root.open_file(
            &format!("snapshots/{}/files/{path}", snapshot.digest),
            OFlags::RDONLY,
            Mode::empty(),
        )?;
        if file.metadata()?.len() != entry.bytes() {
            return Err(Error::Malformed("cache weights length changed".into()));
        }
        Ok(crate::artifact::Artifact::from_file(file, Path::new(path))?.info)
    }
    pub fn disk_limit_bytes(&self) -> Result<u64> {
        let policy: Policy = serde_json::from_slice(&self.read("policy.json", CONTROL_LIMIT)?)
            .map_err(|_| Error::Malformed("cache selection policy".into()))?;
        if policy.schema != "zerogravity.cache.policy.v2" || policy.disk_limit_bytes == 0 {
            return Err(Error::Malformed("cache selection policy identity".into()));
        }
        Ok(policy.disk_limit_bytes)
    }
}
