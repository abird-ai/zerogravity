//! Generic CacheStore seeds distinguish byte integrity from acquisition eligibility.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::{Fixture, admission, digest, durable};
use std::io::Cursor;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::Error;
use zerogravity_artifacts::acquisition::{
    ComponentKind, ComponentPullRequest, ComponentRequest, Provider, PullLimits, PullRequest,
    Selection, open_components, validate_cached_selection,
};
use zerogravity_artifacts::cache::{
    CacheStore, FileRole, Manifest, ManifestEntry, ManifestSource, TransferAdmission,
};

const PIN: &str = "1111111111111111111111111111111111111111";
const CONTROL_PIN: &str = "2222222222222222222222222222222222222222";
const POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1\noid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\nsize 123\n";
type Entry<'a> = (&'a str, FileRole, &'a [u8]);

fn request(provider: Provider, repository: &str, revision: &str, path: &str) -> PullRequest {
    PullRequest {
        provider,
        repository: repository.into(),
        revision: if revision.starts_with("release-") {
            "fixture-tag"
        } else {
            revision
        }
        .into(),
        selection: if revision.starts_with("release-") {
            Selection::ReleaseAsset { name: path.into() }
        } else {
            Selection::Files {
                paths: vec![path.into()],
                subfolder: None,
            }
        },
    }
}
fn seed(
    store: &CacheStore,
    memory: &TransferAdmission,
    provider: Provider,
    repository: &str,
    revision: &str,
    entries: &[Entry<'_>],
) -> String {
    let manifest = Manifest::new(
        ManifestSource::new(provider.label(), repository, revision).unwrap(),
        entries
            .iter()
            .map(|(path, role, bytes)| {
                ManifestEntry::new(*path, digest(bytes), bytes.len() as u64, *role).unwrap()
            })
            .collect(),
        &Default::default(),
    )
    .unwrap();
    let mut tx = store
        .begin(manifest, memory, Arc::new(AtomicBool::new(false)))
        .unwrap();
    for (path, _, bytes) in entries {
        tx.import_object(path, &mut Cursor::new(*bytes)).unwrap();
    }
    durable(tx.commit(None).unwrap())
}
struct Sources {
    _fixture: Fixture,
    store: CacheStore,
    memory: TransferAdmission,
}
impl Sources {
    fn new() -> Self {
        let fixture = Fixture::new();
        let store = fixture.open();
        Self {
            _fixture: fixture,
            store,
            memory: admission(128 * 1024 * 1024),
        }
    }
    fn weights(
        &self,
        provider: Provider,
        revision: &str,
        selected: &str,
        entries: &[Entry<'_>],
    ) -> (ComponentPullRequest, Vec<(String, String)>) {
        let mut members = vec![
            ComponentRequest::new(
                "weights",
                ComponentKind::Weights,
                request(provider, "converted/weights", revision, selected),
            )
            .unwrap(),
        ];
        let weights = seed(
            &self.store,
            &self.memory,
            provider,
            "converted/weights",
            revision,
            entries,
        );
        let mut bindings = vec![("weights".into(), weights)];
        for (label, kind, path, role, bytes) in [
            (
                "config",
                ComponentKind::Config,
                "config.json",
                FileRole::Config,
                b"{}".as_slice(),
            ),
            (
                "tokenizer",
                ComponentKind::Tokenizer,
                "tokenizer.json",
                FileRole::Tokenizer,
                b"{}".as_slice(),
            ),
        ] {
            members.push(
                ComponentRequest::new(
                    label,
                    kind,
                    request(Provider::HuggingFace, "author/original", CONTROL_PIN, path),
                )
                .unwrap(),
            );
            bindings.push((
                label.into(),
                seed(
                    &self.store,
                    &self.memory,
                    Provider::HuggingFace,
                    "author/original",
                    CONTROL_PIN,
                    &[(path, role, bytes)],
                ),
            ));
        }
        (
            ComponentPullRequest::new(members, &PullLimits::default()).unwrap(),
            bindings,
        )
    }
}
fn index(first: &str, second: &str) -> Vec<u8> {
    // Tensor-payload total, deliberately unrelated to index/shard file lengths.
    serde_json::to_vec(&serde_json::json!({
        "metadata":{"total_size":2},"weight_map":{"a":first,"b":second}
    }))
    .unwrap()
}

#[test]
fn correct_hash_generic_pointer_bytes_remain_inspectable_but_repository_offline_refuses() {
    let sources = Sources::new();
    for (provider, revision) in [
        (Provider::HuggingFace, PIN),
        (Provider::GitHub, PIN),
        (Provider::GitHub, "release-71-asset-93"),
    ] {
        for role in [FileRole::Weights, FileRole::Auxiliary] {
            let snapshot = seed(
                &sources.store,
                &sources.memory,
                provider,
                "owner/repo",
                revision,
                &[("model.gguf", role, POINTER)],
            );
            // Ordinary A1 byte access is still legal; no parser/model eligibility follows.
            let lease = sources.store.open_snapshot(&snapshot).unwrap();
            assert_eq!(lease.read_verified("model.gguf", 1024).unwrap(), POINTER);
            let error = validate_cached_selection(
                &lease,
                &request(provider, "owner/repo", revision, "model.gguf"),
            )
            .unwrap_err();
            assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        }
    }
}

#[test]
fn pointer_weights_cannot_supply_an_offline_component_set() {
    let sources = Sources::new();
    for (provider, revision) in [
        (Provider::HuggingFace, PIN),
        (Provider::GitHub, PIN),
        (Provider::GitHub, "release-71-asset-93"),
    ] {
        let (request, bindings) = sources.weights(
            provider,
            revision,
            "model.gguf",
            &[("model.gguf", FileRole::Weights, POINTER)],
        );
        // The weights file, real roles, independent control sources and all bytes are valid A1 data.
        assert!(matches!(
            open_components(&sources.store, &request, &bindings, &PullLimits::default()),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn offline_component_weights_require_all_split_gguf_members() {
    let sources = Sources::new();
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let (request, bindings) = sources.weights(
            provider,
            PIN,
            "model-00001-of-00002.gguf",
            &[("model-00001-of-00002.gguf", FileRole::Weights, b"one")],
        );
        assert!(matches!(
            open_components(&sources.store, &request, &bindings, &PullLimits::default()),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn offline_component_weights_require_index_members_and_source_index_for_named_shards() {
    let sources = Sources::new();
    let json = index(
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    );
    for entries in [
        vec![
            (
                "nested/model.safetensors.index.json",
                FileRole::Index,
                json.as_slice(),
            ),
            (
                "nested/model-00001-of-00002.safetensors",
                FileRole::Weights,
                b"one".as_slice(),
            ),
        ],
        vec![
            (
                "nested/model-00001-of-00002.safetensors",
                FileRole::Weights,
                b"one".as_slice(),
            ),
            (
                "nested/model-00002-of-00002.safetensors",
                FileRole::Weights,
                b"two".as_slice(),
            ),
        ],
    ] {
        let selected = entries[0].0;
        let (request, bindings) = sources.weights(Provider::HuggingFace, PIN, selected, &entries);
        assert!(matches!(
            open_components(&sources.store, &request, &bindings, &PullLimits::default()),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn complete_split_and_nested_index_sets_keep_separate_control_sources() {
    let sources = Sources::new();
    let json = index(
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    );
    for entries in [
        vec![
            (
                "model-00001-of-00002.gguf",
                FileRole::Weights,
                b"one".as_slice(),
            ),
            (
                "model-00002-of-00002.gguf",
                FileRole::Weights,
                b"two".as_slice(),
            ),
        ],
        vec![
            (
                "nested/model.safetensors.index.json",
                FileRole::Index,
                json.as_slice(),
            ),
            (
                "nested/model-00001-of-00002.safetensors",
                FileRole::Weights,
                b"one".as_slice(),
            ),
            (
                "nested/model-00002-of-00002.safetensors",
                FileRole::Weights,
                b"two".as_slice(),
            ),
        ],
    ] {
        let (request, bindings) = sources.weights(Provider::GitHub, PIN, entries[0].0, &entries);
        let set =
            open_components(&sources.store, &request, &bindings, &PullLimits::default()).unwrap();
        assert_eq!(set.components().len(), 3);
        assert_eq!(
            set.component("weights")
                .unwrap()
                .lease()
                .manifest()
                .entries()
                .len(),
            entries.len()
        );
        assert_eq!(
            set.component("config")
                .unwrap()
                .lease()
                .manifest()
                .source()
                .repository(),
            "author/original"
        );
        assert_eq!(
            set.component("weights")
                .unwrap()
                .lease()
                .manifest()
                .source()
                .repository(),
            "converted/weights"
        );
    }
}

#[test]
fn offline_index_reads_share_one_cumulative_metadata_and_request_budget() {
    let sources = Sources::new();
    let a = index("a.safetensors", "a.safetensors");
    let b = index("b.safetensors", "b.safetensors");
    let (request, bindings) = sources.weights(
        Provider::HuggingFace,
        PIN,
        "a.safetensors.index.json",
        &[
            ("a.safetensors.index.json", FileRole::Index, &a),
            ("b.safetensors.index.json", FileRole::Index, &b),
            ("a.safetensors", FileRole::Weights, b"a"),
            ("b.safetensors", FileRole::Weights, b"b"),
        ],
    );
    // Each index fits individually, but the cumulative bytes do not. With ample
    // budget, this whole-snapshot multi-variant set still refuses as ambiguous.
    let bytes = PullLimits {
        max_metadata_bytes: a.len() + b.len() - 1,
        ..PullLimits::default()
    };
    assert!(a.len() < bytes.max_metadata_bytes && b.len() < bytes.max_metadata_bytes);
    assert!(matches!(
        open_components(&sources.store, &request, &bindings, &bytes),
        Err(Error::Limit(_))
    ));
    let requests = PullLimits {
        max_pages: 1,
        ..PullLimits::default()
    };
    assert!(matches!(
        open_components(&sources.store, &request, &bindings, &requests),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        open_components(&sources.store, &request, &bindings, &PullLimits::default()),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn ordinary_initial_selection_remains_sparse_role_agnostic_and_data_only() {
    let sources = Sources::new();
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let snapshot = seed(
            &sources.store,
            &sources.memory,
            provider,
            "owner/repo",
            PIN,
            &[
                ("weights/q4.gguf", FileRole::Auxiliary, b"abc"),
                (
                    "weights/config.json",
                    FileRole::Weights,
                    b"not parsed as config",
                ),
            ],
        );
        let lease = sources.store.open_snapshot(&snapshot).unwrap();
        validate_cached_selection(
            &lease,
            &request(provider, "owner/repo", PIN, "weights/q4.gguf"),
        )
        .unwrap();
        assert_eq!(lease.manifest().entries().len(), 2); // Missing tokenizer is deliberate.
    }
}

#[test]
fn canonical_different_release_assets_remain_coherent_offline_components() {
    let sources = Sources::new();
    let (mut request, mut bindings) = sources.weights(
        Provider::GitHub,
        "release-71-asset-93",
        "model.gguf",
        &[("model.gguf", FileRole::Weights, b"data")],
    );
    let mut members = request.components().to_vec();
    for (position, label, kind, path, role, revision) in [
        (
            1,
            "config",
            ComponentKind::Config,
            "config.json",
            FileRole::Config,
            "release-82-asset-101",
        ),
        (
            2,
            "tokenizer",
            ComponentKind::Tokenizer,
            "tokenizer.json",
            FileRole::Tokenizer,
            "release-82-asset-102",
        ),
    ] {
        members[position] = ComponentRequest::new(
            label,
            kind,
            self::request(Provider::GitHub, "author/original", revision, path),
        )
        .unwrap();
        bindings[position].1 = seed(
            &sources.store,
            &sources.memory,
            Provider::GitHub,
            "author/original",
            revision,
            &[(path, role, b"{}")],
        );
    }
    request = ComponentPullRequest::new(members, &PullLimits::default()).unwrap();
    let set = open_components(&sources.store, &request, &bindings, &PullLimits::default()).unwrap();
    assert_eq!(set.components().len(), 3);
    assert_ne!(
        set.component("config")
            .unwrap()
            .lease()
            .manifest()
            .source()
            .revision(),
        set.component("tokenizer")
            .unwrap()
            .lease()
            .manifest()
            .source()
            .revision()
    );
}
