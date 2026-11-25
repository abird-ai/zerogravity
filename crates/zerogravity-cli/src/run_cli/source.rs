//! Exact cold source selection before choosing the original execution adapter.
use super::execute::Mode;
use super::*;

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(super) fn resolve(
    request: &mut Request,
    inventory: &devices::Inventory,
    mode: Mode,
) -> Result<()> {
    let Entrypoint::Model(id) = &request.entrypoint else {
        return Ok(());
    };
    let i = request
        .graph
        .models
        .iter()
        .position(|model| &model.id == id)
        .ok_or_else(|| bad("direct entrypoint model absent"))?;
    if request.graph.models[i].native_artifacts.is_some() {
        if mode.pull {
            return Err(bad(
                "--pull cannot acquire finite native diagnostic artifacts",
            ));
        }
        return Ok(());
    }
    if let Some(missing) = select_model(&mut request.graph.models[i])? {
        let digest = acquire_missing(request, inventory, mode.offer, mode.pull, &missing, None)?;
        bind(&mut request.graph.models[i], digest)?;
    }
    Ok(())
}

/// Cold unresolved catalog source. No credentials, transfer admission or
/// cache initialization occurs until every serving route has been checked.
#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) struct MissingSource {
    pub target: String,
    root: std::path::PathBuf,
    disk: Option<u64>,
}
#[cfg(all(feature = "cache", target_os = "linux"))]
impl MissingSource {
    #[cfg(all(
        any(feature = "serve", feature = "tui"),
        feature = "model-qwen35",
        feature = "text"
    ))]
    pub fn expected_snapshot(&self) -> Result<&'static str> {
        #[cfg(feature = "model-qwen35")]
        if let Some(digest) = zerogravity::models::qwen_source_snapshot(&self.target) {
            return Ok(digest);
        }
        Err(self.not_cached())
    }

    pub(crate) fn not_cached(&self) -> Error {
        let name = &self.target;
        Error::Unsupported(format!(
            "model {name} is not cached; use model list or explicit model pull{}",
            if zerogravity::targets::find(name).is_some() {
                "; --pull requires a supported exact catalog recipe"
            } else {
                ""
            }
        ))
    }
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn select_model(
    model: &mut zerogravity::pipeline::ModelInstance,
) -> Result<Option<MissingSource>> {
    use zerogravity::cache::{CacheMetadata, paths};
    let Some(source) = model.source.as_deref() else {
        return Ok(None);
    };
    let selector = SourceSelector::parse(source)?;
    if matches!(selector, SourceSelector::Local(_)) {
        return Ok(None);
    }
    match &selector {
        SourceSelector::HuggingFace(value) => {
            zerogravity::cache::provenance::ProviderLocator::parse("huggingface", value)?;
        }
        SourceSelector::GitHub(value) => {
            zerogravity::cache::provenance::ProviderLocator::parse("github", value)?;
        }
        SourceSelector::Http(url) => {
            #[cfg(feature = "acquisition")]
            zerogravity::acquisition::object::HttpsObjectSpec::validate_cached_locator(url)?;
            #[cfg(not(feature = "acquisition"))]
            {
                let _ = url;
                return Err(Error::Unsupported(
                    "HTTP source form requires compiled acquisition contract parser; no request is performed".into(),
                ));
            }
        }
        _ => {}
    }
    let root = paths::resolve_root(
        None,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    let metadata = match CacheMetadata::open(&root) {
        Ok(metadata) => Some(metadata),
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let digest = if let SourceSelector::Name(name) = &selector {
        let selected = metadata
            .as_ref()
            .map(|metadata| {
                zerogravity::models::find_named_cached(metadata, name, model.snapshot.as_deref())
            })
            .transpose()?
            .flatten();
        if let Some(selected) = selected {
            selected.digest
        } else {
            if model.snapshot.is_some() {
                return Err(Error::Unsupported(
                    "explicitly pinned model is not cached; acquire its exact snapshot separately"
                        .into(),
                ));
            }
            if !model.target.is_empty() && model.target != *name {
                return Err(bad(
                    "declared model target differs from requested missing catalog name",
                ));
            }
            return Ok(Some(MissingSource {
                target: name.clone(),
                root,
                disk: metadata
                    .as_ref()
                    .map(CacheMetadata::disk_limit_bytes)
                    .transpose()?,
            }));
        }
    } else if matches!(
        selector,
        SourceSelector::HuggingFace(_) | SourceSelector::GitHub(_)
    ) && model.snapshot.is_none()
    {
        let selected = metadata
            .as_ref()
            .map(|metadata| metadata.provider_optional(&selector))
            .transpose()?
            .flatten();
        if let Some(selected) = selected {
            selected.digest
        } else {
            #[cfg(feature = "model-qwen35")]
            if let SourceSelector::HuggingFace(locator) = &selector
                && let Some(target) = zerogravity::models::qwen_target_for_locator(locator)?
            {
                if !model.target.is_empty() && model.target != target {
                    return Err(bad(
                        "declared model target differs from requested exact catalog source",
                    ));
                }
                return Ok(Some(MissingSource {
                    target: target.into(),
                    root,
                    disk: metadata
                        .as_ref()
                        .map(CacheMetadata::disk_limit_bytes)
                        .transpose()?,
                }));
            }
            return Err(Error::Unsupported("provider source is not acquired; use models/pull then an explicit --snapshot or catalog reference".into()));
        }
    } else {
        let metadata = metadata.ok_or_else(|| {
            Error::Unsupported(
                "model source is not cached; use model pull with explicit source selection".into(),
            )
        })?;
        zerogravity::models::select_cached(&metadata, &selector, model.snapshot.as_deref())?.digest
    };
    bind(model, digest)?;
    Ok(None)
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn bind(model: &mut zerogravity::pipeline::ModelInstance, digest: String) -> Result<()> {
    #[cfg(feature = "model-qwen35")]
    if let Some(target) = zerogravity::models::qwen_target_for_snapshot(&digest) {
        if !model.target.is_empty() && model.target != target {
            return Err(bad(
                "declared model target differs from exact approved source",
            ));
        }
        model.target = target.into();
    }
    crate::output::emit!(Verbose, "Selected immutable snapshot: {digest}");
    model.source = Some(format!("sha256:{digest}"));
    model.snapshot = Some(digest);
    Ok(())
}

#[cfg(not(all(feature = "cache", target_os = "linux")))]
pub(super) fn resolve(_: &mut Request, _: &devices::Inventory, mode: Mode) -> Result<()> {
    let _ = mode.offer;
    if mode.pull {
        return Err(Error::Unsupported(
            "--pull requires the optional Linux acquisition feature".into(),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn acquire_missing(
    request: &Request,
    inventory: &devices::Inventory,
    offer: bool,
    pull: bool,
    missing: &MissingSource,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<String> {
    if request.diagnostics.dry_run || !(offer || pull) {
        return Err(missing.not_cached());
    }
    #[cfg(all(
        feature = "acquisition",
        feature = "model-qwen35",
        feature = "text",
        any(feature = "cuda", feature = "hip")
    ))]
    {
        let name = &missing.target;
        let recipe = zerogravity::models::QwenPullRecipe::for_target(name)?;
        let backend = super::qwen_source::select_backend(request, inventory, recipe.target())?
            .ok_or_else(|| Error::Unsupported("approved original HF source has no eligible compiled GPU route; choose an explicitly acquired GGUF for CPU".into()))?;
        // Preserve selected device/profile and source-geometry refusal BEFORE
        // any network intent. The original loader remains final admission.
        super::qwen_source::preflight_acquisition(request, inventory, backend)?;
        let disk = missing.disk.unwrap_or(crate::pull::default_disk_limit());
        if recipe.bytes() > disk {
            return Err(Error::Limit("approved source exceeds existing cache disk policy; acquire explicitly with sufficient disk limit".into()));
        }
        crate::output::emit!(
            Status,
            "Model {name} not cached · {} @ {} · {:.2} GiB · {:?}",
            recipe.repository(),
            recipe.revision(),
            recipe.bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            backend
        );
        if !pull && !super::input::confirm_download()? {
            return Err(Error::Unsupported(
                "download declined; no acquisition performed".into(),
            ));
        }
        crate::pull::acquire_for_run(&recipe, &missing.root, disk, cancel)?;
        Ok(recipe.snapshot().into())
    }
    #[cfg(not(all(
        feature = "acquisition",
        feature = "model-qwen35",
        feature = "text",
        any(feature = "cuda", feature = "hip")
    )))]
    {
        let _ = (inventory, &missing.root, missing.disk, cancel);
        if pull {
            Err(Error::Unsupported("automatic catalog acquisition requires the optional acquisition and qualified source-bound GPU features; use explicit model pull".into()))
        } else {
            Err(missing.not_cached())
        }
    }
}
