//! Cold observations through the original cache/device authorities. No leases,
//! execution eligibility, polling loop or alternate inventory store.
#[cfg(all(feature = "cache", target_os = "linux"))]
use zerogravity::{
    Result,
    cache::PartialSummary,
    control::{CachedModelSummary, ControlPlane},
};

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn cache_root() -> Result<std::path::PathBuf> {
    zerogravity::cache::paths::resolve_root(
        None,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) struct CacheObservation {
    pub cached: Vec<CachedModelSummary>,
    pub partial: Result<Vec<PartialSummary>>,
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn cache(control: ControlPlane<'_>) -> Result<CacheObservation> {
    let root = cache_root()?;
    Ok(CacheObservation {
        cached: control.cached_models(&root)?,
        // A bad checkpoint must not hide the verified canonical metadata.
        partial: control.cached_partials(&root),
    })
}

// A manifest may describe local or arbitrary provider labels. Only bounded,
// conventional remote repository/revision labels cross the browser boundary.
#[cfg(all(feature = "cache", target_os = "linux"))]
fn remote(provider: &str) -> bool {
    matches!(
        provider,
        "huggingface"
            | "huggingface-bundle"
            | "github"
            | "github-bundle"
            | "modelscope"
            | "local-hf-cache"
    )
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 240
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn public_repository<'a>(provider: &str, repository: &'a str) -> &'a str {
    if remote(provider)
        && repository.len() <= 512
        && repository
            .split_once('/')
            .is_some_and(|(owner, name)| label(owner) && label(name))
    {
        repository
    } else {
        "Repository not exposed"
    }
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn public_revision<'a>(provider: &str, revision: &'a str) -> &'a str {
    if remote(provider) && label(revision) {
        revision
    } else {
        "Revision not exposed"
    }
}

/// Both clients invalidate the same cold owner observations. A terminal
/// notification identifies its own operation, even after slot replacement.
pub(crate) fn models_changed(kind: zerogravity::control::ControlEventKind) -> bool {
    use zerogravity::control::{ControlEventKind as E, ReclaimStatus};
    match kind {
        E::CacheAliasFinished { success } => success,
        E::CacheReclamationFinished { outcome } => {
            matches!(outcome, ReclaimStatus::Durable | ReclaimStatus::Incomplete)
        }
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        E::DownloadChanged { status, .. } => status.terminal(),
        _ => false,
    }
}

pub(crate) fn devices_changed(kind: zerogravity::control::ControlEventKind) -> bool {
    use zerogravity::control::ControlEventKind as E;
    matches!(
        kind,
        E::ModelLoadFinished { .. } | E::ModelUnloadFinished { .. }
    )
}
