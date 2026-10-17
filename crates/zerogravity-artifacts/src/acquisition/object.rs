//! One explicitly authorized HTTPS object. The supplied size and raw digest
//! bind data, not remote membership, model closure or executable readiness.
use super::{PullLimits, ResolvedFile, ResolvedPullPlan, http};
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{
    CacheStore, DigestProvenance, FileRole, Manifest, ManifestSource, SnapshotLease,
    TransferAdmission,
};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::fmt;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use url::Url;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

const MAX_URL_BYTES: usize = 16 * 1024;
const MAX_ORIGIN_BYTES: usize = 4096;
pub(super) const PREPARE_CONTROL_BYTES: u64 = 1024 * 1024;

/// A pure, secret-free expected origin and content contract. The download URL
/// is supplied only to prepare and remains inside the existing redacted plan.
#[derive(Clone)]
pub struct HttpsObjectSpec {
    origin: String,
    path: String,
    role: FileRole,
    bytes: u64,
    sha256: String,
    source: ManifestSource,
}

impl HttpsObjectSpec {
    pub fn new(
        origin: impl AsRef<str>,
        path: impl AsRef<str>,
        role: FileRole,
        bytes: u64,
        sha256: impl AsRef<str>,
    ) -> Result<Self> {
        let origin = origin.as_ref();
        if origin.len() > MAX_ORIGIN_BYTES {
            return Err(Error::Limit("HTTPS object origin limit".into()));
        }
        let parsed = parse_object_url(origin)?;
        let authority_and_tail = origin.split_once("://").expect("validated URL syntax").1;
        let tail = authority_and_tail
            .find(['/', '?', '#'])
            .map_or("", |position| &authority_and_tail[position..]);
        if !matches!(tail, "" | "/") || parsed.path() != "/" || parsed.query().is_some() {
            return Err(Error::Malformed(
                "HTTPS object requires a bare origin".into(),
            ));
        }
        let origin = parsed.origin().ascii_serialization();
        if origin.len() > MAX_ORIGIN_BYTES {
            return Err(Error::Limit("HTTPS object normalized origin limit".into()));
        }
        let path = path.as_ref();
        let sha256 = sha256.as_ref();
        validate_relative_path(path)?;
        validate_digest(sha256)?;
        let path = path.to_owned();
        let sha256 = sha256.to_owned();
        let origin_digest = hash(
            &serde_json::to_vec(&("zerogravity.https-object.origin.v1", &origin))
                .map_err(|_| Error::Malformed("HTTPS object origin encoding failed".into()))?,
        );
        // The fixed serde tuple is unambiguous and explicitly versioned. The
        // URL path/query are transient locators and never become this identity.
        let contract_digest = hash(
            &serde_json::to_vec(&(
                "zerogravity.https-object.contract.v1",
                &origin,
                &path,
                role,
                bytes,
                &sha256,
            ))
            .map_err(|_| Error::Malformed("HTTPS object contract encoding failed".into()))?,
        );
        let source = ManifestSource::new(
            "https-object",
            format!("origin-{origin_digest}"),
            contract_digest,
        )?;
        Ok(Self {
            origin,
            path,
            role,
            bytes,
            sha256,
            source,
        })
    }

    /// Normalized origin for explicit credential scoping. No download path,
    /// query, userinfo or fragment is present.
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }

    fn validate_limits(&self, limits: &PullLimits) -> Result<()> {
        limits.validate()?;
        if self.bytes > limits.cache.max_file_bytes || self.bytes > limits.cache.max_snapshot_bytes
        {
            return Err(Error::Limit(
                "HTTPS object exceeds caller byte bounds".into(),
            ));
        }
        Ok(())
    }

    /// Local plan construction only: no client, HEAD, metadata or payload
    /// request occurs. Size is caller-authorized rather than remotely observed.
    /// The operation-local control lease ends with preparation, as for resolve;
    /// returned plan allocations and all TLS/OS memory are not lifetime-accounted.
    pub fn prepare(
        &self,
        url: &str,
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<ResolvedPullPlan> {
        self.validate_limits(limits)?;
        check_cancel(cancel)?;
        let _control = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: PREPARE_CONTROL_BYTES,
            }],
        )?;
        let parsed = parse_object_url(url)?;
        if parsed.origin().ascii_serialization() != self.origin {
            return Err(Error::Malformed(
                "HTTPS object URL has a different origin".into(),
            ));
        }
        check_cancel(cancel)?;
        let file = ResolvedFile::new(
            &self.path,
            parsed.as_str(),
            self.role,
            self.bytes,
            Some(self.sha256.clone()),
            format!("https-object:{}", self.source.revision()),
        )?;
        let plan = ResolvedPullPlan::new(self.source.clone(), vec![file], limits)?;
        check_cancel(cancel)?;
        Ok(plan)
    }

    /// Pure offline opening: no URL, transport, credential or admission input.
    /// A canonical digest selects a snapshot; other values resolve one existing
    /// ref once. Original managed descriptors and full raw verification apply.
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        self.validate_limits(limits)?;
        check_cancel(cancel)?;
        open_cached_object(
            store,
            snapshot,
            limits,
            ExpectedObject {
                source: &self.source,
                path: &self.path,
                role: self.role,
                bytes: self.bytes,
                digest: ExpectedDigest::TrustedSha256(&self.sha256),
            },
            cancel,
        )
    }
}

pub(super) enum ExpectedDigest<'a> {
    TrustedSha256(&'a str),
    FirstSeen,
}
pub(super) struct ExpectedObject<'a> {
    pub source: &'a ManifestSource,
    pub path: &'a str,
    pub role: FileRole,
    pub bytes: u64,
    pub digest: ExpectedDigest<'a>,
}
/// The original managed single-object full hash, exact caller metadata and LFS
/// policy. A FirstSeen contract has no fabricated expected raw SHA or provenance.
/// Native callers validate a snapshot digest first; HTTPS retains ref support.
pub(super) fn open_cached_object(
    store: &CacheStore,
    snapshot: &str,
    limits: &PullLimits,
    expected: ExpectedObject<'_>,
    cancel: &AtomicBool,
) -> Result<SnapshotLease> {
    check_cancel(cancel)?;
    let lease = if validate_digest(snapshot).is_ok() {
        store.open_snapshot_cancellable(snapshot, cancel)?
    } else {
        store.resolve_ref_cancellable(snapshot, cancel)?
    };
    let manifest = lease.manifest();
    if manifest.source() != expected.source || manifest.entries().len() != 1 {
        return Err(Error::Malformed(
            "cached object source/closure mismatch".into(),
        ));
    }
    let entry = &manifest.entries()[0];
    let digest_matches = match expected.digest {
        ExpectedDigest::TrustedSha256(sha) => {
            entry.digest() == sha && entry.digest_provenance() == DigestProvenance::TrustedSha256
        }
        ExpectedDigest::FirstSeen => entry.digest_provenance() == DigestProvenance::FirstSeen,
    };
    if entry.path() != expected.path
        || entry.role() != expected.role
        || entry.bytes() != expected.bytes
        || !digest_matches
    {
        return Err(Error::Malformed(
            "cached object content contract mismatch".into(),
        ));
    }
    Manifest::from_json(&manifest.canonical_bytes()?, &limits.cache)?;
    let mut prefix = [0u8; 128];
    let count = expected.bytes.min(prefix.len() as u64) as usize;
    lease
        .file(expected.path)?
        .read_exact_at(&mut prefix[..count], 0)?;
    check_cancel(cancel)?;
    super::executor::reject_pointer(&prefix[..count])?;
    Ok(lease)
}

impl fmt::Debug for HttpsObjectSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpsObjectSpec")
            .field("source", &self.source)
            .field("path", &self.path)
            .field("role", &self.role)
            .field("bytes", &self.bytes)
            .field("sha256", &self.sha256)
            .finish()
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("HTTPS object preparation canceled".into()))
    } else {
        Ok(())
    }
}

fn parse_object_url(value: &str) -> Result<Url> {
    if value.len() > MAX_URL_BYTES {
        return Err(Error::Limit("HTTPS object URL limit".into()));
    }
    if value.trim() != value || value.chars().any(char::is_control) || value.contains('\\') {
        return Err(Error::Malformed("invalid HTTPS object URL syntax".into()));
    }
    let (_, remainder) = value
        .split_once("://")
        .ok_or_else(|| Error::Malformed("invalid HTTPS object URL syntax".into()))?;
    let authority = remainder.split(['/', '?', '#']).next().unwrap_or_default();
    // Even an empty explicit userinfo marker refuses. url::Url otherwise drops
    // that spelling, so inspecting parsed username alone would miss it.
    if authority.contains('@') {
        return Err(Error::Malformed("HTTPS object userinfo refused".into()));
    }
    let diagnostic_http = cfg!(feature = "diagnostic-acquisition")
        && value
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http:"));
    http::parse_url(value, diagnostic_http)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheOptions, CommitOutcome, ManifestEntry};
    use std::fs;
    use std::os::unix::fs::DirBuilderExt;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

    const DATA: &[u8] = b"explicit HTTPS object data";
    const PATH: &str = "nested/object.bin";

    fn memory(bytes: u64) -> TransferAdmission {
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("test-host", "direct-https-object").unwrap();
        arbiter.register(resource.clone(), bytes).unwrap();
        TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("tiny-local-fixture", "direct-object").unwrap(),
        }
    }

    fn released(admission: &TransferAdmission) {
        let usage = admission.arbiter.snapshot(&admission.resource).unwrap();
        assert_eq!(usage.reserved, 0);
        assert_eq!(usage.reservations, 0);
    }

    fn spec() -> HttpsObjectSpec {
        HttpsObjectSpec::new(
            "https://example.test",
            PATH,
            FileRole::Auxiliary,
            DATA.len() as u64,
            hash(DATA),
        )
        .unwrap()
    }

    #[test]
    fn canonical_origin_and_contract_bind_all_expected_fields() {
        let plain = spec();
        let normalized = HttpsObjectSpec::new(
            "HTTPS://EXAMPLE.TEST:443/",
            PATH,
            FileRole::Auxiliary,
            DATA.len() as u64,
            hash(DATA),
        )
        .unwrap();
        assert_eq!(plain.origin(), "https://example.test");
        assert_eq!(plain.source(), normalized.source());
        let idna = HttpsObjectSpec::new(
            "https://bücher.example:443",
            PATH,
            FileRole::Auxiliary,
            DATA.len() as u64,
            hash(DATA),
        )
        .unwrap();
        assert_eq!(idna.origin(), "https://xn--bcher-kva.example");
        for altered in [
            HttpsObjectSpec::new(
                "https://other.test",
                PATH,
                FileRole::Auxiliary,
                DATA.len() as u64,
                hash(DATA),
            )
            .unwrap(),
            HttpsObjectSpec::new(
                "https://example.test:444",
                PATH,
                FileRole::Auxiliary,
                DATA.len() as u64,
                hash(DATA),
            )
            .unwrap(),
            HttpsObjectSpec::new(
                plain.origin(),
                "other.bin",
                FileRole::Auxiliary,
                DATA.len() as u64,
                hash(DATA),
            )
            .unwrap(),
            HttpsObjectSpec::new(
                plain.origin(),
                PATH,
                FileRole::Weights,
                DATA.len() as u64,
                hash(DATA),
            )
            .unwrap(),
            HttpsObjectSpec::new(
                plain.origin(),
                PATH,
                FileRole::Auxiliary,
                DATA.len() as u64 + 1,
                hash(DATA),
            )
            .unwrap(),
            HttpsObjectSpec::new(
                plain.origin(),
                PATH,
                FileRole::Auxiliary,
                DATA.len() as u64,
                hash(b"other"),
            )
            .unwrap(),
        ] {
            assert_ne!(plain.source().revision(), altered.source().revision());
        }
        assert_eq!(plain.source().provider(), "https-object");
        assert!(plain.source().repository().starts_with("origin-"));
        assert_eq!(plain.source().revision().len(), 64);
        assert!(!format!("{plain:?}").contains("https://"));
    }

    #[test]
    fn changing_signed_locators_preserves_contract_and_never_enters_debug_or_source() {
        let spec = spec();
        let admission = memory(128 * 1024 * 1024);
        let cancel = AtomicBool::new(false);
        let first = spec
            .prepare(
                "https://example.test/SECRET_PATH?signature=SECRET_QUERY",
                &PullLimits::default(),
                &admission,
                &cancel,
            )
            .unwrap();
        let second = spec
            .prepare(
                "HTTPS://EXAMPLE.TEST:443/OTHER_PATH?signature=ROTATED_SECRET",
                &PullLimits::default(),
                &admission,
                &cancel,
            )
            .unwrap();
        assert_eq!(first.source(), second.source());
        assert_eq!(first.files()[0].identity(), second.files()[0].identity());
        assert_eq!(first.total_bytes(), Some(DATA.len() as u64));
        assert_eq!(first.files()[0].sha256(), Some(spec.sha256()));
        let visible = format!(
            "{first:?}{second:?}{spec:?}{}",
            serde_json::to_string(first.source()).unwrap()
        );
        for secret in [
            "SECRET_PATH",
            "SECRET_QUERY",
            "OTHER_PATH",
            "ROTATED_SECRET",
            "https://",
        ] {
            assert!(!visible.contains(secret), "visible secret: {secret}");
        }
        released(&admission);
    }

    #[test]
    fn malformed_origins_urls_paths_and_checksums_refuse_with_sanitized_errors() {
        for origin in [
            "https://example.test/path",
            "https://example.test?SECRET_QUERY",
            "https://example.test#SECRET_FRAGMENT",
            "https://user:SECRET_PASSWORD@example.test",
            "https://@example.test",
            "https://example.test/../path",
            "https://example.test/segment/..",
            "https://example.test/./",
            "https://example.test/%2e",
            "https://example.test:999999",
            "https://",
            "https://example.test\n",
            " https://example.test",
            "https:\\example.test",
        ] {
            let error =
                HttpsObjectSpec::new(origin, PATH, FileRole::Auxiliary, 1, hash(b"x")).unwrap_err();
            assert!(!error.to_string().contains("SECRET"));
        }
        assert!(
            HttpsObjectSpec::new(
                "https://example.test",
                "../escape",
                FileRole::Auxiliary,
                0,
                hash(b"")
            )
            .is_err()
        );
        assert!(
            HttpsObjectSpec::new(
                "https://example.test",
                PATH,
                FileRole::Auxiliary,
                0,
                "A".repeat(64)
            )
            .is_err()
        );
        let spec = spec();
        let admission = memory(128 * 1024 * 1024);
        for url in [
            "https://other.test/?SECRET_QUERY",
            "https://user:SECRET_PASSWORD@example.test/object",
            "https://@example.test/object",
            "https://example.test/object#SECRET_FRAGMENT",
            "https://example.test/\nSECRET",
            " https://example.test/object",
            "file:///SECRET_PATH",
        ] {
            let error = spec
                .prepare(
                    url,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false),
                )
                .unwrap_err();
            assert!(!error.to_string().contains("SECRET"));
            released(&admission);
        }
    }

    #[test]
    fn prepare_admission_cancel_and_exact_bounds_precede_url_work_and_release() {
        let spec = spec();
        let denied = memory(PREPARE_CONTROL_BYTES - 1);
        assert!(matches!(
            spec.prepare(
                "invalidSECRET",
                &PullLimits::default(),
                &denied,
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        released(&denied);
        let admission = memory(128 * 1024 * 1024);
        assert!(matches!(
            spec.prepare(
                "invalidSECRET",
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(true)
            ),
            Err(Error::State(_))
        ));
        let limits = PullLimits {
            cache: crate::cache::CacheLimits {
                max_file_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            spec.prepare(
                "invalidSECRET",
                &limits,
                &admission,
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        let limits = PullLimits {
            cache: crate::cache::CacheLimits {
                max_manifest_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            spec.prepare(
                "https://example.test/object",
                &limits,
                &admission,
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        let huge = HttpsObjectSpec::new(spec.origin(), PATH, spec.role(), u64::MAX, spec.sha256())
            .unwrap();
        assert!(
            huge.prepare(
                "https://example.test/object",
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        released(&admission);
    }

    #[test]
    fn zero_is_exact_and_diagnostic_http_remains_literal_loopback_only() {
        let empty = HttpsObjectSpec::new(
            "https://example.test",
            "empty.bin",
            FileRole::Auxiliary,
            0,
            hash(b""),
        )
        .unwrap();
        let admission = memory(128 * 1024 * 1024);
        let plan = empty
            .prepare(
                "https://example.test/empty",
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(plan.files()[0].bytes(), Some(0));
        assert_eq!(plan.total_bytes(), Some(0));
        let local = HttpsObjectSpec::new(
            "http://127.0.0.1:12345",
            PATH,
            FileRole::Auxiliary,
            DATA.len() as u64,
            hash(DATA),
        );
        assert_eq!(local.is_ok(), cfg!(feature = "diagnostic-acquisition"));
        assert!(
            HttpsObjectSpec::new(
                "http://example.test",
                PATH,
                FileRole::Auxiliary,
                DATA.len() as u64,
                hash(DATA)
            )
            .is_err()
        );
        released(&admission);
    }

    struct Fixture {
        root: PathBuf,
        store: CacheStore,
        admission: TransferAdmission,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let root = std::env::temp_dir().join(format!(
                "zerogravity-object-api-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let store =
                CacheStore::open(CacheOptions::new(Some(root.clone()), 8 * 1024 * 1024)).unwrap();
            Self {
                root,
                store,
                admission: memory(128 * 1024 * 1024),
            }
        }
        fn publish(
            &self,
            source: &ManifestSource,
            files: &[(&str, FileRole, &[u8], DigestProvenance)],
            reference: Option<&str>,
        ) -> String {
            let entries = files
                .iter()
                .map(|(path, role, data, provenance)| {
                    ManifestEntry::new(*path, hash(data), data.len() as u64, *role)
                        .unwrap()
                        .with_digest_provenance(*provenance)
                })
                .collect();
            let manifest = Manifest::new(source.clone(), entries, &Default::default()).unwrap();
            let mut tx = self
                .store
                .begin(manifest, &self.admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            for (path, _, data, _) in files {
                let mut reader = *data;
                tx.import_object(path, &mut reader).unwrap();
            }
            match tx.commit(reference).unwrap() {
                CommitOutcome::Durable(snapshot) => snapshot.digest().to_owned(),
                other => panic!("unexpected uncertain fixture: {other:?}"),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn offline_original_snapshot_and_ref_match_full_exact_contract_without_url() {
        let f = Fixture::new();
        let spec = spec();
        let snapshot = f.publish(
            spec.source(),
            &[(PATH, spec.role(), DATA, DigestProvenance::TrustedSha256)],
            Some("object"),
        );
        for selector in [snapshot.as_str(), "object"] {
            let lease = spec
                .open_cached(
                    &f.store,
                    selector,
                    &PullLimits::default(),
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(lease.manifest().digest().unwrap(), snapshot);
            assert_eq!(lease.manifest().entries()[0].digest(), spec.sha256());
        }
        assert!(
            spec.open_cached(
                &f.store,
                &snapshot,
                &PullLimits::default(),
                &AtomicBool::new(true)
            )
            .is_err()
        );
        let limits = PullLimits {
            cache: crate::cache::CacheLimits {
                max_file_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            spec.open_cached(&f.store, &snapshot, &limits, &AtomicBool::new(false)),
            Err(Error::Limit(_))
        ));
        released(&f.admission);
    }

    #[test]
    fn offline_rejects_wrong_source_metadata_provenance_extra_files_and_corruption() {
        let f = Fixture::new();
        let spec = spec();
        let trust = DigestProvenance::TrustedSha256;
        let wrong_source =
            ManifestSource::new("https-object", "other-origin", spec.source().revision()).unwrap();
        let cases = [
            f.publish(&wrong_source, &[(PATH, spec.role(), DATA, trust)], None),
            f.publish(
                spec.source(),
                &[("other.bin", spec.role(), DATA, trust)],
                None,
            ),
            f.publish(
                spec.source(),
                &[(PATH, FileRole::Config, DATA, trust)],
                None,
            ),
            f.publish(
                spec.source(),
                &[(PATH, spec.role(), b"different", trust)],
                None,
            ),
            f.publish(
                spec.source(),
                &[(PATH, spec.role(), DATA, DigestProvenance::FirstSeen)],
                None,
            ),
            f.publish(
                spec.source(),
                &[(PATH, spec.role(), DATA, DigestProvenance::Unspecified)],
                None,
            ),
            f.publish(
                spec.source(),
                &[
                    (PATH, spec.role(), DATA, trust),
                    ("extra.bin", spec.role(), b"extra", trust),
                ],
                None,
            ),
        ];
        for snapshot in cases {
            assert!(
                spec.open_cached(
                    &f.store,
                    &snapshot,
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
        }
        let snapshot = f.publish(spec.source(), &[(PATH, spec.role(), DATA, trust)], None);
        let limits = PullLimits {
            cache: crate::cache::CacheLimits {
                max_manifest_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            spec.open_cached(&f.store, &snapshot, &limits, &AtomicBool::new(false)),
            Err(Error::Limit(_))
        ));
        fs::write(
            f.root.join(format!("snapshots/{snapshot}/files/{PATH}")),
            vec![b'x'; DATA.len()],
        )
        .unwrap();
        assert!(
            spec.open_cached(
                &f.store,
                &snapshot,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        released(&f.admission);
    }

    #[test]
    fn offline_pointer_policy_matches_existing_executor_even_for_manually_trusted_data() {
        let f = Fixture::new();
        let pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\n";
        let spec = HttpsObjectSpec::new(
            "https://example.test",
            PATH,
            FileRole::Auxiliary,
            pointer.len() as u64,
            hash(pointer),
        )
        .unwrap();
        let snapshot = f.publish(
            spec.source(),
            &[(PATH, spec.role(), pointer, DigestProvenance::TrustedSha256)],
            None,
        );
        assert!(matches!(
            spec.open_cached(
                &f.store,
                &snapshot,
                &PullLimits::default(),
                &AtomicBool::new(false)
            ),
            Err(Error::Unsupported(_))
        ));
        released(&f.admission);
    }
}
