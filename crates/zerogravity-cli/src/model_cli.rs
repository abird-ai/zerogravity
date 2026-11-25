//! Normal model commands; the original SDK/cache owners retain all authority.
use crate::args::{Format, ModelRead};
use zerogravity::{Error, Result, run_policy::SourceSelector};

pub(super) fn search(args: &[String]) -> Result<()> {
    let query = match args {
        [] => None,
        [query] => Some(query.to_ascii_lowercase()),
        _ => {
            return Err(Error::Malformed(
                "model search accepts at most one query".into(),
            ));
        }
    };
    println!("Known model families (intent only; not cache or execution status):");
    for model in zerogravity::targets::MODEL_TARGETS {
        if query.as_ref().is_none_or(|q| {
            model.slug.to_ascii_lowercase().contains(q)
                || model
                    .repository
                    .is_some_and(|r| r.to_ascii_lowercase().contains(q))
        }) {
            println!(
                "{}  {}  {}",
                model.slug,
                model.task.label(),
                model.source_status.label()
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub(super) fn pull(args: &[String]) -> Result<()> {
    use zerogravity::cache::provenance::ProviderLocator;
    let first = args.first().ok_or_else(|| {
        Error::Malformed(
            "model pull requires an explicit provider source and selection; see model pull --help"
                .into(),
        )
    })?;
    if matches!(
        first.as_str(),
        "hf" | "github"
            | "candidates"
            | "xet"
            | "modelscope"
            | "azure"
            | "s3"
            | "gcs"
            | "oci"
            | "object"
            | "components"
    ) {
        return crate::pull::run(args);
    }
    let (provider, value) = match SourceSelector::parse(first)? {
        SourceSelector::HuggingFace(value) => ("huggingface", value),
        SourceSelector::GitHub(value) => ("github", value),
        _ => return Err(Error::Unsupported("model pull requires an explicit hf: or gh: locator and file/quant selection; no implicit acquisition".into())),
    };
    let locator = ProviderLocator::parse(provider, &value)?;
    let mut canonical = vec![
        if provider == "huggingface" {
            "hf".into()
        } else {
            "github".into()
        },
        locator.repository,
    ];
    if let Some(revision) = locator.revision {
        canonical.extend(["--revision".into(), revision]);
    }
    canonical.extend_from_slice(&args[1..]);
    crate::pull::run(&canonical)
}

#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
pub(super) fn pull(_: &[String]) -> Result<()> {
    Err(Error::Unsupported(
        "model pull requires the optional acquisition feature on Linux".into(),
    ))
}

struct Query {
    selector: Option<SourceSelector>,
    cache: Option<std::path::PathBuf>,
    snapshot: Option<String>,
    dry_run: bool,
    partial: bool,
}

fn parse(operation: ModelRead, args: &[String], yes: bool) -> Result<Query> {
    let invalid = || Error::Malformed("invalid model arguments; use model COMMAND --help".into());
    let mut query = Query {
        selector: None,
        cache: None,
        snapshot: None,
        dry_run: false,
        partial: false,
    };
    let mut rest = args.iter();
    while let Some(raw) = rest.next() {
        match raw.as_str() {
            "--cache" if query.cache.is_none() => {
                let path = std::path::PathBuf::from(rest.next().ok_or_else(invalid)?);
                if !path.is_absolute() {
                    return Err(invalid());
                }
                query.cache = Some(path);
            }
            "--snapshot" if query.snapshot.is_none() && operation != ModelRead::List => {
                let value = rest.next().ok_or_else(invalid)?;
                if !SourceSelector::is_snapshot_digest(value) {
                    return Err(invalid());
                }
                query.snapshot = Some(value.clone());
            }
            "--partial" if operation == ModelRead::List && !query.partial => query.partial = true,
            "--dry-run" if operation == ModelRead::Remove && !query.dry_run => query.dry_run = true,
            raw if !raw.starts_with('-')
                && operation != ModelRead::List
                && query.selector.is_none() =>
            {
                query.selector = Some(SourceSelector::parse(raw)?);
            }
            _ => return Err(invalid()),
        }
    }
    if operation != ModelRead::List && query.selector.is_none() {
        return Err(invalid());
    }
    if operation == ModelRead::Remove && query.dry_run == yes {
        return Err(Error::Malformed(
            "model remove requires exactly one of --dry-run or --yes".into(),
        ));
    }
    Ok(query)
}

pub(super) fn read(operation: ModelRead, args: &[String], format: Format, yes: bool) -> Result<()> {
    let query = parse(operation, args, yes)?;
    if let Some(SourceSelector::Local(path)) = &query.selector {
        if operation != ModelRead::Inspect || query.snapshot.is_some() || query.cache.is_some() {
            return Err(Error::Unsupported("local paths support artifact inspection only; cache operations require an acquired reference".into()));
        }
        return crate::inspect_with_format(path, format);
    }
    cached(operation, query, format)
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn cached(operation: ModelRead, query: Query, format: Format) -> Result<()> {
    use std::sync::atomic::AtomicBool;
    use zerogravity::{Runtime, cache::ReclaimMode};
    let root = zerogravity::cache::paths::resolve_root(
        query.cache.as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    let runtime = Runtime::builder().build()?;
    let control = runtime.control();
    if operation == ModelRead::List {
        if query.partial {
            let partials = control.cached_partials(&root)?;
            if format == Format::Json {
                return crate::output::json(&partials);
            }
            println!("Checkpoint records (may include active attempts; resume must revalidate):");
            for partial in partials {
                println!("{}  {}/{} bytes", partial.key, partial.bytes, partial.total);
            }
            return Ok(());
        }
        let snapshots = control.cached_models(&root)?;
        if format == Format::Json {
            return crate::output::json(&snapshots);
        }
        if snapshots.is_empty() {
            println!("No acquired snapshots.");
            return Ok(());
        }
        println!("Cached snapshot metadata (payloads and execution not verified):");
        for snapshot in snapshots {
            println!(
                "{}  {}:{}@{}  {} bytes",
                snapshot.snapshot,
                snapshot.provider,
                snapshot.repository,
                snapshot.revision,
                snapshot.bytes,
            );
        }
        return Ok(());
    }
    let selector = query.selector.as_ref().expect("validated selector");
    let pin = query.snapshot.as_deref();
    let cancel = AtomicBool::new(false);
    match operation {
        ModelRead::Inspect | ModelRead::Verify => {
            let detail = if operation == ModelRead::Inspect {
                control.inspect_cached(&root, selector, pin)?
            } else {
                control.verify_cached(&root, selector, pin, &cancel)?
            };
            describe(
                &detail.snapshot,
                &detail.manifest,
                detail.payload_verified,
                format,
            )
        }
        ModelRead::Remove => {
            let selected = control.inspect_cached(&root, selector, pin)?;
            let outcome = control.reclaim_cached(
                &root,
                &selected.snapshot,
                if query.dry_run {
                    ReclaimMode::DryRun
                } else {
                    ReclaimMode::Apply
                },
                &cancel,
            )?;
            crate::cache_manage::describe_reclaim(outcome)
        }
        ModelRead::List => unreachable!(),
    }
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn describe(
    digest: &str,
    manifest: &zerogravity::cache::Manifest,
    verified: bool,
    format: Format,
) -> Result<()> {
    if format == Format::Json {
        let document: serde_json::Value = serde_json::from_slice(&manifest.canonical_bytes()?)
            .map_err(|_| Error::State("cache manifest encoding".into()))?;
        return crate::output::json(&serde_json::json!({
            "snapshot": digest, "bytes": manifest.total_bytes(), "manifest": document,
            "payload_verified": verified, "execution_qualified": false,
        }));
    }
    let source = manifest.source();
    println!(
        "Snapshot: {digest}\nSource: {}:{}@{}\nFiles: {}\nBytes: {}",
        source.provider(),
        source.repository(),
        source.revision(),
        manifest.entries().len(),
        manifest.total_bytes()
    );
    println!(
        "{}",
        if verified {
            "Payloads verified; no remote freshness or execution claim."
        } else {
            "Manifest identity checked; payloads and execution not verified."
        }
    );
    Ok(())
}

#[cfg(not(all(feature = "cache", target_os = "linux")))]
fn cached(_: ModelRead, _: Query, _: Format) -> Result<()> {
    Err(Error::Unsupported(
        "cached model commands require Linux and the cache feature".into(),
    ))
}
