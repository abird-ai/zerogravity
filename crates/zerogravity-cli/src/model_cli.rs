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

/// Browse and manage the selected document's discovery sources. No second
/// catalog, cache or credential store; remote queries are explicit.
pub(super) fn discovery(
    operation: &str,
    args: &[String],
    selection: Option<(&str, &str)>,
    offline: bool,
    format: Format,
) -> Result<()> {
    use std::sync::atomic::AtomicBool;
    use zerogravity::discovery::{self as d, Kind, Source};
    let bad =
        || Error::Malformed("invalid discovery command; see model sources/browse --help".into());
    let file =
        zerogravity::configuration::ConfigFile::open(crate::config_cli::selected_path(selection)?)?;
    if selection.is_some() {
        file.require_existing()?;
    }
    let doc = zerogravity::pipelines::parse_document(file.bytes())?;
    let mut sources = d::sources(&doc);
    if operation == "sources" {
        match args {
            [] => {}
            [op] if op == "list" => {}
            [op, name, kind, scope @ ..]
                if matches!(op.as_str(), "add" | "edit") && scope.len() <= 1 =>
            {
                let kind = match kind.as_str() {
                    "hf" | "hugging_face" => Kind::HuggingFace,
                    "github" | "gh" => Kind::Github,
                    "local" => Kind::Local,
                    _ => return Err(bad()),
                };
                let existing = sources.iter().position(|source| &source.name == name);
                if (op == "add") == existing.is_some() {
                    return Err(Error::State(
                        "source already exists or does not exist".into(),
                    ));
                }
                let source = Source {
                    name: name.clone(),
                    kind,
                    scope: scope.first().cloned(),
                    enabled: existing.is_none_or(|i| sources[i].enabled),
                };
                if let Some(i) = existing {
                    sources[i] = source;
                } else {
                    sources.push(source);
                }
                d::replace(&file, &sources)?;
            }
            [op, name] if matches!(op.as_str(), "remove" | "enable" | "disable") => {
                let i = sources
                    .iter()
                    .position(|source| &source.name == name)
                    .ok_or_else(bad)?;
                if op == "remove" {
                    sources.remove(i);
                } else {
                    sources[i].enabled = op == "enable";
                }
                d::replace(&file, &sources)?;
            }
            _ => return Err(bad()),
        }
        if format == Format::Json {
            return crate::output::json(&sources);
        }
        for s in sources {
            println!(
                "{}  {}  {}{}",
                s.name,
                s.kind.label(),
                s.scope.as_deref().unwrap_or("default"),
                if s.enabled { "" } else { " · disabled" }
            );
        }
        return Ok(());
    }
    let (name, rest) = if operation == "search" {
        let pos = args.iter().position(|a| a == "--source").ok_or_else(bad)?;
        if pos + 2 != args.len() {
            return Err(bad());
        }
        (&args[pos + 1], &args[..pos])
    } else {
        args.split_first().ok_or_else(bad)?
    };
    let mut query = "";
    let mut page = 0;
    let mut cursor: Option<&str> = None;
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        if arg == "--page" {
            page = rest.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
        } else if arg == "--cursor" {
            cursor = Some(rest.next().ok_or_else(bad)?.as_str());
        } else if query.is_empty() && !arg.starts_with('-') {
            query = arg;
        } else {
            return Err(bad());
        }
    }
    let cancel = AtomicBool::new(false);
    let Some(source) = sources.iter().find(|s| &s.name == name && s.enabled) else {
        if operation != "browse" || !query.is_empty() || page != 0 || cursor.is_some() {
            return Err(bad());
        }
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        {
            let repository = d::files(name, offline, &cancel)?;
            if format == Format::Json {
                return crate::output::json(&repository);
            }
            println!("{}", repository.locator);
            for f in repository.files {
                println!(
                    "{}  {}  {}",
                    f.path,
                    f.bytes.map_or("unknown".into(), |n| n.to_string()),
                    f.role
                );
            }
            return Ok(());
        }
        #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
        return Err(Error::Unsupported(
            "remote repository discovery requires acquisition".into(),
        ));
    };
    let result = if source.kind == Kind::Local {
        d::local(source, query, page, &cancel)?
    } else if offline {
        d::Page {
            entries: d::curated(source)
                .into_iter()
                .filter(|e| e.name.to_lowercase().contains(&query.to_lowercase()))
                .collect(),
            more: false,
            next_cursor: None,
        }
    } else {
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        {
            d::browse(source, query, page, cursor, false, &cancel)?
        }
        #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
        {
            return Err(Error::Unsupported(
                "remote discovery requires acquisition; use --offline for curated metadata".into(),
            ));
        }
    };
    if format == Format::Json {
        return crate::output::json(&result);
    }
    for e in result.entries {
        println!(
            "{}  {}{}",
            e.name,
            e.locator,
            if e.payload_hold {
                " · payload hold"
            } else {
                ""
            }
        );
    }
    if result.more {
        if let Some(cursor) = result.next_cursor {
            println!("More results: --page {} --cursor {}", page + 1, cursor);
        } else {
            println!("More results: use --page {}", page + 1);
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
    let normalized = zerogravity::discovery::normalize_locator(first);
    let (provider, value) = match SourceSelector::parse(&normalized)? {
        SourceSelector::HuggingFace(value) => ("huggingface", value),
        SourceSelector::GitHub(value) => ("github", value),
        _ => return Err(Error::Unsupported("model pull requires an explicit hf: or gh: locator and file/quant selection; no implicit acquisition".into())),
    };
    let locator = ProviderLocator::parse(provider, &value)?;
    if provider == "huggingface"
        && zerogravity::targets::MODEL_TARGETS.iter().any(|t| {
            t.payload_hold
                && t.repository
                    .is_some_and(|repo| repo.eq_ignore_ascii_case(&locator.repository))
        })
    {
        return Err(Error::Unsupported(
            "Repository is on an explicit model payload hold".into(),
        ));
    }
    #[cfg(feature = "model-qwen35")]
    if args.len() == 1
        && provider == "huggingface"
        && let Some(target) = zerogravity::targets::MODEL_TARGETS
            .iter()
            .find(|t| t.repository == Some(locator.repository.as_str()))
        && let Ok(recipe) = zerogravity::models::QwenPullRecipe::for_target(target.slug)
        && locator
            .revision
            .as_deref()
            .is_none_or(|revision| revision == recipe.revision())
    {
        let intent = crate::pull::native::Intent::parse(target.slug, "")?;
        let runtime = zerogravity::Runtime::builder().build()?;
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let _signals = crate::signals::CancellationSignals::register(cancel.clone())?;
        crate::output::emit!(Status, "{}", intent.confirmation()?);
        let result = intent.execute(&runtime, &cancel, false);
        let retired = runtime.shutdown();
        println!("{}", result?);
        return retired;
    }
    if args.len() == 1 {
        return Err(Error::Malformed("Select repository files or quantization: model browse REPOSITORY_URL, then model pull SOURCE --file PATH or --quant TAG".into()));
    }
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
