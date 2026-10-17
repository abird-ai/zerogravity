//! Bounded data-only snapshot metadata. No source identity or model readiness.
use super::paths::{validate_digest, validate_relative_path};
use crate::{Error, Result};
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const SCHEMA: &str = "zerogravity.cache.snapshot.v1";
const MAX_FILES: usize = 4096;
const MAX_MANIFEST: usize = 8 * 1024 * 1024;
const MAX_DEPTH: usize = 8;

/// Logical input bounds, independently subdivided by the store's disk policy.
#[derive(Clone, Copy, Debug)]
pub struct CacheLimits {
    pub max_manifest_bytes: usize,
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_snapshot_bytes: u64,
}
impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 1024 * 1024,
            max_files: MAX_FILES,
            max_file_bytes: 1024 * 1024 * 1024 * 1024,
            max_snapshot_bytes: 4 * 1024 * 1024 * 1024 * 1024,
        }
    }
}
impl CacheLimits {
    pub fn validate(&self) -> Result<()> {
        if self.max_manifest_bytes == 0
            || self.max_manifest_bytes > MAX_MANIFEST
            || self.max_files == 0
            || self.max_files > MAX_FILES
            || self.max_file_bytes == 0
            || self.max_snapshot_bytes == 0
        {
            return Err(Error::Limit("invalid cache manifest limits".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileRole {
    Weights,
    Config,
    Tokenizer,
    Index,
    License,
    Auxiliary,
}

/// A local content hash does not authenticate an upstream source. This records
/// whether the resolver supplied a raw SHA256 or observed bytes for the first
/// time. Older/local manifests conservatively report Unspecified.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestProvenance {
    Unspecified,
    TrustedSha256,
    FirstSeen,
}

/// Secret-free provenance labels; a provider resolver must qualify its own pin.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ManifestSource {
    provider: String,
    repository: String,
    revision: String,
}
impl ManifestSource {
    pub fn new(
        provider: impl Into<String>,
        repository: impl Into<String>,
        revision: impl Into<String>,
    ) -> Result<Self> {
        let source = Self {
            provider: provider.into(),
            repository: repository.into(),
            revision: revision.into(),
        };
        if source.provider.is_empty()
            || source.provider.len() > 32
            || !source
                .provider
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        {
            return Err(Error::Malformed("invalid cache provider label".into()));
        }
        for (label, limit) in [(&source.repository, 4096), (&source.revision, 256)] {
            if label.is_empty()
                || label.len() > limit
                || label.chars().any(char::is_control)
                || label.contains(['\\', '@', '?', '#'])
                || label.contains("://")
            {
                return Err(Error::Malformed(
                    "cache provenance requires bounded repository/revision labels, not credential URLs"
                        .into(),
                ));
            }
        }
        Ok(source)
    }
    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn repository(&self) -> &str {
        &self.repository
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
}

/// Every listed file belongs to the complete closure. Optional selection is a
/// plan/resolver concern before a final manifest is constructed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ManifestEntry {
    path: String,
    digest: String,
    bytes: u64,
    role: FileRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    digest_provenance: Option<DigestProvenance>,
}
impl ManifestEntry {
    pub fn new(
        path: impl Into<String>,
        digest: impl Into<String>,
        bytes: u64,
        role: FileRole,
    ) -> Result<Self> {
        let value = Self {
            path: path.into(),
            digest: digest.into(),
            bytes,
            role,
            digest_provenance: None,
        };
        validate_relative_path(&value.path)?;
        validate_digest(&value.digest)?;
        Ok(value)
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn with_digest_provenance(mut self, provenance: DigestProvenance) -> Self {
        self.digest_provenance = Some(provenance);
        self
    }
    pub fn digest_provenance(&self) -> DigestProvenance {
        self.digest_provenance
            .unwrap_or(DigestProvenance::Unspecified)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSource {
    provider: String,
    repository: String,
    revision: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    path: String,
    digest: String,
    // Native serde u64 parsing never passes through f64. Fractions/exponents
    // are not part of this explicit cache schema, including zero fractions.
    bytes: u64,
    role: FileRole,
    #[serde(default)]
    digest_provenance: Option<DigestProvenance>,
}
struct RawEntries(Vec<RawEntry>);
impl<'de> Deserialize<'de> for RawEntries {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct EntriesVisitor;
        impl<'de> Visitor<'de> for EntriesVisitor {
            type Value = RawEntries;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a bounded cache file array")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut array: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = array.next_element::<RawEntry>()? {
                    if entries.len() == MAX_FILES {
                        return Err(serde::de::Error::custom("too many cache manifest files"));
                    }
                    entries.push(entry);
                }
                Ok(RawEntries(entries))
            }
        }
        deserializer.deserialize_seq(EntriesVisitor)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    schema: String,
    source: RawSource,
    files: RawEntries,
}

/// Validated canonical data closure, distinct from an executable model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    source: ManifestSource,
    entries: Vec<ManifestEntry>,
    total_bytes: u64,
}
impl Manifest {
    pub fn new(
        source: ManifestSource,
        mut entries: Vec<ManifestEntry>,
        limits: &CacheLimits,
    ) -> Result<Self> {
        limits.validate()?;
        if entries.is_empty() || entries.len() > limits.max_files {
            return Err(Error::Limit("cache manifest file count".into()));
        }
        entries.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        let mut paths = BTreeSet::new();
        let mut lengths = BTreeMap::new();
        let mut total_bytes = 0u64;
        for entry in &entries {
            if !paths.insert(entry.path.as_str()) {
                return Err(Error::Malformed("duplicate cache logical path".into()));
            }
            if entry.bytes > limits.max_file_bytes {
                return Err(Error::Limit("cache file exceeds length limit".into()));
            }
            if lengths
                .insert(entry.digest.as_str(), entry.bytes)
                .is_some_and(|old| old != entry.bytes)
            {
                return Err(Error::Malformed(
                    "one digest has conflicting lengths".into(),
                ));
            }
            total_bytes = total_bytes
                .checked_add(entry.bytes)
                .filter(|n| *n <= limits.max_snapshot_bytes)
                .ok_or_else(|| Error::Limit("cache snapshot byte total".into()))?;
        }
        for entry in &entries {
            for (index, _) in entry.path.match_indices('/') {
                if paths.contains(&entry.path[..index]) {
                    return Err(Error::Malformed(
                        "cache file/directory path collision".into(),
                    ));
                }
            }
        }
        let manifest = Self {
            source,
            entries,
            total_bytes,
        };
        if manifest.canonical_bytes()?.len() > limits.max_manifest_bytes {
            return Err(Error::Limit("canonical cache manifest byte limit".into()));
        }
        Ok(manifest)
    }
    pub fn from_json(bytes: &[u8], limits: &CacheLimits) -> Result<Self> {
        limits.validate()?;
        if bytes.len() > limits.max_manifest_bytes {
            return Err(Error::Limit("cache manifest JSON byte limit".into()));
        }
        bounded_depth(bytes)?;
        let raw: RawManifest = serde_json::from_slice(bytes)
            .map_err(|e| Error::Malformed(format!("cache manifest: {e}")))?;
        if raw.schema != SCHEMA {
            return Err(Error::Unsupported("cache manifest schema version".into()));
        }
        let source = ManifestSource::new(
            raw.source.provider,
            raw.source.repository,
            raw.source.revision,
        )?;
        let entries = raw
            .files
            .0
            .into_iter()
            .map(|r| {
                let mut entry = ManifestEntry::new(r.path, r.digest, r.bytes, r.role)?;
                entry.digest_provenance = r.digest_provenance;
                Ok(entry)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(source, entries, limits)
    }
    pub fn entries(&self) -> &[ManifestEntry] {
        &self.entries
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
    /// Compact fixed-field-order v1 JSON, sorted paths, one final newline.
    /// Only this metadata is encoded; source/config/tokenizer bytes are untouched.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        #[derive(Serialize)]
        struct Wire<'a> {
            schema: &'static str,
            source: &'a ManifestSource,
            files: &'a [ManifestEntry],
        }
        let mut bytes = serde_json::to_vec(&Wire {
            schema: SCHEMA,
            source: &self.source,
            files: &self.entries,
        })
        .map_err(|e| Error::Malformed(format!("cache manifest encoding: {e}")))?;
        bytes.push(b'\n');
        Ok(bytes)
    }
    pub fn digest(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(self.canonical_bytes()?)))
    }
}

fn bounded_depth(bytes: &[u8]) -> Result<()> {
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for &byte in bytes {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    depth += 1;
                    if depth > MAX_DEPTH {
                        return Err(Error::Limit("cache manifest JSON depth".into()));
                    }
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    Ok(()) // The actual serde parser validates syntax/string termination.
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(path: &str, bytes: u64) -> ManifestEntry {
        ManifestEntry::new(path, "0".repeat(64), bytes, FileRole::Auxiliary).unwrap()
    }
    fn source() -> ManifestSource {
        ManifestSource::new("local", "tiny-fixture", "v1").unwrap()
    }
    #[test]
    fn canonical_order_roundtrip_and_unicode_remain_explicit() {
        let a = Manifest::new(
            source(),
            vec![entry("β/model", 3), entry("A", 3)],
            &CacheLimits::default(),
        )
        .unwrap();
        let b = Manifest::new(
            source(),
            vec![entry("A", 3), entry("β/model", 3)],
            &CacheLimits::default(),
        )
        .unwrap();
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
        assert_eq!(
            Manifest::from_json(&a.canonical_bytes().unwrap(), &CacheLimits::default()).unwrap(),
            a
        );
        assert_eq!(a.total_bytes(), 6);
        assert!(a.canonical_bytes().unwrap().ends_with(b"\n"));
    }
    #[test]
    fn collisions_lengths_count_and_overflow_refuse() {
        let limits = CacheLimits::default();
        for files in [
            vec![entry("a", 3), entry("a", 3)],
            vec![entry("a", 3), entry("a-", 3), entry("a/x", 3)],
            vec![entry("a", 3), entry("b", 4)],
        ] {
            assert!(Manifest::new(source(), files, &limits).is_err());
        }
        let broad = CacheLimits {
            max_file_bytes: u64::MAX,
            max_snapshot_bytes: u64::MAX,
            ..limits
        };
        assert!(
            Manifest::new(
                source(),
                vec![entry("a", u64::MAX), entry("b", u64::MAX)],
                &broad
            )
            .is_err()
        );
        assert!(
            Manifest::new(
                source(),
                vec![entry("a", 3)],
                &CacheLimits {
                    max_files: 0,
                    ..limits
                }
            )
            .is_err()
        );
        assert!(
            Manifest::new(
                source(),
                vec![entry("a", 3)],
                &CacheLimits {
                    max_manifest_bytes: 1,
                    ..limits
                }
            )
            .is_err()
        );
    }
    #[test]
    fn duplicate_unknown_fields_and_noninteger_lengths_are_not_trusted() {
        let good = Manifest::new(source(), vec![entry("a", 3)], &CacheLimits::default()).unwrap();
        let text = String::from_utf8(good.canonical_bytes().unwrap()).unwrap();
        for invalid in [
            text.replace("\"schema\":", "\"schema\":\"duplicate\",\"schema\":"),
            text.replace("\"provider\":", "\"provider\":\"local\",\"provider\":"),
            text.replace("\"bytes\":3", "\"bytes\":3,\"bytes\":3"),
            text.replace("\"bytes\":3", "\"bytes\":3,\"unknown\":true"),
            text.replace("\"bytes\":3", "\"bytes\":3.0"),
            text.replace("\"bytes\":3", "\"bytes\":3e0"),
            text.replace("\"bytes\":3", "\"bytes\":18446744073709551616"),
        ] {
            assert!(
                Manifest::from_json(invalid.as_bytes(), &CacheLimits::default()).is_err(),
                "{invalid}"
            );
        }
        assert!(bounded_depth(b"[[[[[[[[[]]]]]]]]]").is_err());
        assert!(ManifestSource::new("local", "https://user:secret@host/model", "v1").is_err());
    }
}
