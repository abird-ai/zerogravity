//! Explicit interactive-client acquisition intent. No terminal or secondary runtime.
use super::*;
use zerogravity::{
    Runtime,
    cache::{CacheMetadata, provenance::ProviderLocator},
    run_policy::SourceSelector,
};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Intent {
    Files(PullRequest),
    Quant(acquisition::GgufQuantRequest),
    #[cfg(feature = "model-qwen35")]
    Recipe(String),
}

impl Intent {
    pub(crate) fn subject(&self) -> crate::control_host::progress::Subject {
        use crate::control_host::progress::Subject;
        match self {
            #[cfg(feature = "model-qwen35")]
            Self::Recipe(target) => Subject::Catalog(target.clone()),
            Self::Files(request) => Subject::Repository {
                provider: request.provider.label().into(),
                repository: request.repository.clone(),
                revision: request.revision.clone(),
            },
            Self::Quant(request) => Subject::Repository {
                provider: request.source().provider().label().into(),
                repository: request.source().repository().into(),
                revision: request.source().revision().into(),
            },
        }
    }

    /// An interactive pull is explicit and immutable. No branch/tag guesses, raw
    /// URLs, shell flags, credentials, or auto-selected quantization.
    pub(crate) fn parse(source: &str, selection: &str) -> Result<Self> {
        let (provider, value) = match SourceSelector::parse(source.trim())? {
            SourceSelector::HuggingFace(value) => (Provider::HuggingFace, value),
            SourceSelector::GitHub(value) => (Provider::GitHub, value),
            #[cfg(feature = "model-qwen35")]
            SourceSelector::Name(name) if selection.is_empty() => {
                zerogravity::models::QwenPullRecipe::for_target(&name)?;
                return Ok(Self::Recipe(name));
            }
            _ => {
                return Err(Error::Malformed(
                    "Use an approved catalog target or hf:/gh:OWNER/REPO@COMMIT".into(),
                ));
            }
        };
        let locator = ProviderLocator::parse(provider.label(), &value)?;
        let revision = locator
            .revision
            .filter(|revision| {
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            })
            .ok_or_else(|| {
                Error::Malformed("Interactive pull requires a full lowercase commit SHA".into())
            })?;
        if let Some(quant) = selection.trim().strip_prefix("quant:") {
            return Ok(Self::Quant(acquisition::GgufQuantRequest::new(
                acquisition::CandidateRequest::new(provider, &locator.repository, &revision, None)?,
                quant,
            )?));
        }
        let paths = selection
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let request = PullRequest {
            provider,
            repository: locator.repository,
            revision,
            selection: Selection::Files {
                paths,
                subfolder: None,
            },
        };
        request.validate(&PullLimits::default())?;
        Ok(Self::Files(request))
    }

    pub(crate) fn confirmation(&self) -> Result<String> {
        let selection = match self {
            Self::Files(request) => {
                let Selection::Files { paths, .. } = &request.selection else {
                    unreachable!()
                };
                format!(
                    "{}:{}@{}\n{} exact files:\n{}",
                    request.provider.label(),
                    request.repository,
                    request.revision,
                    paths.len(),
                    paths.join("\n")
                )
            }
            Self::Quant(request) => format!(
                "{}:{}@{}\nGGUF quant: {}",
                request.source().provider().label(),
                request.source().repository(),
                request.source().revision(),
                request.quant()
            ),
            #[cfg(feature = "model-qwen35")]
            Self::Recipe(target) => {
                let recipe = zerogravity::models::QwenPullRecipe::for_target(target)?;
                format!(
                    "{}@{}\nApproved {target}: {} bytes\nExact snapshot {}",
                    recipe.repository(),
                    recipe.revision(),
                    recipe.bytes(),
                    recipe.snapshot()
                )
            }
        };
        Ok(format!(
            "Pull into the selected cache? Existing disk policy (new cache: 16 GiB). No automatic load.\n{selection}"
        ))
    }

    pub(crate) fn execute(
        &self,
        runtime: &Runtime,
        cancel: &Arc<AtomicBool>,
        offline: bool,
    ) -> Result<String> {
        use std::sync::atomic::Ordering;
        if offline {
            return Err(Error::Unsupported(
                "Offline mode prohibits acquisition".into(),
            ));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::State("acquisition canceled".into()));
        }
        let root = zerogravity::cache::paths::resolve_root(
            None,
            std::env::var_os("XDG_CACHE_HOME").as_deref(),
            std::env::var_os("HOME").as_deref(),
        )?;
        let disk = match CacheMetadata::open(&root) {
            Ok(metadata) => metadata.disk_limit_bytes()?,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => DISK_DEFAULT,
            Err(error) => return Err(error),
        };
        let control = runtime.control();
        let receipt = match self {
            #[cfg(feature = "model-qwen35")]
            Self::Recipe(target) => {
                let recipe = zerogravity::models::QwenPullRecipe::for_target(target)?;
                if recipe.bytes() > disk {
                    return Err(Error::Limit("approved source exceeds existing cache disk policy; acquire explicitly with sufficient disk limit".into()));
                }
                let request = recipe.request();
                let mut bounds = PullLimits::default();
                bounds.cache.max_snapshot_bytes = recipe.bytes();
                let hub =
                    runtime
                        .artifacts()
                        .hub(HubSelection::Files(&request), bounds, None, None)?;
                let client = hub_client(Provider::HuggingFace, None)?;
                let admission = admission(MEMORY_DEFAULT)?;
                let operation = control
                    .begin_download(cancel.clone())?
                    .expect_snapshot(recipe.snapshot())?;
                transfer_recipe(operation, &hub, &client, &admission, &recipe, &root, disk)?
            }
            selection => {
                let request = match selection {
                    Self::Files(request) => HubSelection::Files(request),
                    Self::Quant(request) => HubSelection::Quant(request),
                    #[cfg(feature = "model-qwen35")]
                    Self::Recipe(_) => unreachable!(),
                };
                let bounds = PullLimits::default();
                let hub = runtime.artifacts().hub(request, bounds, None, None)?;
                let client = hub_client(request.provider(), None)?;
                let admission = admission(MEMORY_DEFAULT)?;
                let operation = control.begin_download(cancel.clone())?;
                transfer_hub(
                    operation,
                    &hub,
                    &client,
                    &admission,
                    || {
                        let mut cache = CacheOptions::new(Some(root), disk);
                        cache.limits = bounds.cache;
                        CacheStore::open(cache)
                    },
                    |_| Ok(()),
                )?
            }
        };
        match receipt.outcome() {
            CommitOutcome::Durable(snapshot) => Ok(format!(
                "Durable snapshot: {}{}",
                snapshot.digest(),
                if snapshot.by_name_warning().is_some() {
                    " · by-name projection needs refresh"
                } else {
                    ""
                },
            )),
            CommitOutcome::DurabilityUnknown { .. } => Err(Error::State(
                "Cache publication durability unknown; inspect before retrying".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_intent_is_pinned_explicit_and_cold() {
        let source = "hf:org/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let files = Intent::parse(source, "model.gguf\nconfig.json\ntokenizer.json").unwrap();
        assert!(
            matches!(&files, Intent::Files(request) if matches!(&request.selection, Selection::Files { paths, .. } if paths.len() == 3))
        );
        assert!(Intent::parse(source, "quant:Q4_K_M").is_ok());
        for (source, selection) in [
            ("hf:org/model", "model.gguf"),
            ("hf:org/model@main", "model.gguf"),
            (source, "../escape"),
            (source, ""),
            ("https://example.invalid/file", "model.gguf"),
        ] {
            assert!(Intent::parse(source, selection).is_err(), "{source}");
        }
        let runtime = Runtime::builder().build().unwrap();
        assert!(matches!(
            files.execute(&runtime, &Arc::new(AtomicBool::new(false)), true),
            Err(Error::Unsupported(_))
        ));
        assert!(runtime.control().download().is_none());
        runtime.shutdown().unwrap();
    }
}
