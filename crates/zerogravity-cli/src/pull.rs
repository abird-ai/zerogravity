//! Explicit acquisition frontend. Ordinary inspect/run never call this module.
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::acquisition::{self, Provider, PullLimits, PullRequest, Selection};
use zerogravity::cache::{
    CacheOptions, CacheStore, CommitOutcome, SnapshotLease, TransferAdmission,
};
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity::{Error, Result};

const DISK_DEFAULT: u64 = 16 * 1024 * 1024 * 1024;
const MEMORY_DEFAULT: u64 = 128 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Help,
    List,
    Inspect(String),
    Verify(String),
    Fetch(PullRequest),
}
#[derive(Debug)]
struct Options {
    action: Action,
    root: Option<PathBuf>,
    disk: u64,
    memory: u64,
    max_file: Option<u64>,
    max_snapshot: Option<u64>,
    max_files: Option<usize>,
    metadata: Option<usize>,
    pages: Option<usize>,
    timeout: Option<u64>,
    reference: Option<String>,
    dry_run: bool,
    offline: bool,
}
fn invalid() -> Error {
    // Do not echo unknown argv: a mistaken token option must stay out of logs.
    Error::Malformed("invalid pull arguments; use pull --help (tokens are environment-only)".into())
}
fn parse(args: &[String]) -> Result<Options> {
    if args.len() > 8192 || args.iter().any(|s| s.len() > 4096) {
        return Err(Error::Limit("pull arguments exceed bounds".into()));
    }
    let first = args.first().ok_or_else(invalid)?;
    if matches!(first.as_str(), "help" | "--help" | "-h") {
        return if args.len() == 1 {
            Ok(Options {
                action: Action::Help,
                root: None,
                disk: DISK_DEFAULT,
                memory: MEMORY_DEFAULT,
                max_file: None,
                max_snapshot: None,
                max_files: None,
                metadata: None,
                pages: None,
                timeout: None,
                reference: None,
                dry_run: false,
                offline: false,
            })
        } else {
            Err(invalid())
        };
    }
    let mut rest = args[1..].iter();
    let provider = match first.as_str() {
        "hf" => Some(Provider::HuggingFace),
        "github" => Some(Provider::GitHub),
        "list" | "inspect" | "verify" => None,
        _ => return Err(invalid()),
    };
    let positional = if first == "list" {
        None
    } else {
        Some(rest.next().ok_or_else(invalid)?.clone())
    };
    let mut result = Options {
        action: Action::List,
        root: None,
        disk: DISK_DEFAULT,
        memory: MEMORY_DEFAULT,
        max_file: None,
        max_snapshot: None,
        max_files: None,
        metadata: None,
        pages: None,
        timeout: None,
        reference: None,
        dry_run: false,
        offline: false,
    };
    let mut revision = None;
    let mut release = None;
    let mut asset = None;
    let mut subfolder = None;
    let mut files = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(flag) = rest.next() {
        if flag != "--file" && !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--offline" => result.offline = true,
            "--dry-run" => result.dry_run = true,
            "--cache"
            | "--ref"
            | "--revision"
            | "--release"
            | "--asset"
            | "--file"
            | "--subfolder"
            | "--disk-limit-bytes"
            | "--memory-limit-bytes"
            | "--max-file-bytes"
            | "--max-snapshot-bytes"
            | "--max-files"
            | "--max-metadata-bytes"
            | "--max-pages"
            | "--timeout-seconds" => {
                let value = rest.next().ok_or_else(invalid)?;
                if value.is_empty() {
                    return Err(invalid());
                }
                let number = || {
                    value
                        .parse::<u64>()
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or_else(invalid)
                };
                let count = || usize::try_from(number()?).map_err(|_| invalid());
                match flag.as_str() {
                    "--cache" => {
                        let path = PathBuf::from(value);
                        if !path.is_absolute() {
                            return Err(invalid());
                        }
                        result.root = Some(path);
                    }
                    "--ref" => {
                        if value.chars().any(char::is_control) {
                            return Err(invalid());
                        }
                        result.reference = Some(value.clone());
                    }
                    "--revision" => revision = Some(value.clone()),
                    "--release" => release = Some(value.clone()),
                    "--asset" => asset = Some(value.clone()),
                    "--file" => files.push(value.clone()),
                    "--subfolder" => subfolder = Some(value.clone()),
                    "--disk-limit-bytes" => result.disk = number()?,
                    "--memory-limit-bytes" => result.memory = number()?,
                    "--max-file-bytes" => result.max_file = Some(number()?),
                    "--max-snapshot-bytes" => result.max_snapshot = Some(number()?),
                    "--max-files" => result.max_files = Some(count()?),
                    "--max-metadata-bytes" => result.metadata = Some(count()?),
                    "--max-pages" => result.pages = Some(count()?),
                    "--timeout-seconds" => {
                        let seconds = number()?;
                        if seconds > 86400 {
                            return Err(invalid());
                        }
                        result.timeout = Some(seconds);
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(invalid()),
        }
    }
    if let Some(provider) = provider {
        let (revision, selection) = match (revision, release, asset) {
            (Some(revision), None, None) if !files.is_empty() => (
                revision,
                Selection::Files {
                    paths: files,
                    subfolder,
                },
            ),
            (None, Some(tag), Some(name))
                if provider == Provider::GitHub && files.is_empty() && subfolder.is_none() =>
            {
                (tag, Selection::ReleaseAsset { name })
            }
            _ => return Err(invalid()),
        };
        let request = PullRequest {
            provider,
            repository: positional.ok_or_else(invalid)?,
            revision,
            selection,
        };
        request.validate(&limits(&result)?)?;
        result.action = Action::Fetch(request);
    } else {
        if revision.is_some()
            || release.is_some()
            || asset.is_some()
            || subfolder.is_some()
            || !files.is_empty()
            || result.reference.is_some()
            || result.dry_run
        {
            return Err(invalid());
        }
        result.action = match first.as_str() {
            "list" => Action::List,
            "inspect" => Action::Inspect(positional.ok_or_else(invalid)?),
            "verify" => Action::Verify(positional.ok_or_else(invalid)?),
            _ => unreachable!(),
        };
    }
    Ok(result)
}
fn limits(options: &Options) -> Result<PullLimits> {
    let mut limits = PullLimits::default();
    if let Some(n) = options.max_file {
        limits.cache.max_file_bytes = n;
    }
    if let Some(n) = options.max_snapshot {
        limits.cache.max_snapshot_bytes = n;
    }
    if let Some(n) = options.max_files {
        limits.cache.max_files = n;
    }
    if let Some(n) = options.metadata {
        limits.max_metadata_bytes = n;
    }
    if let Some(n) = options.pages {
        limits.max_pages = n;
    }
    limits.validate()?;
    Ok(limits)
}
fn open(options: &Options) -> Result<CacheStore> {
    let mut cache = CacheOptions::new(options.root.clone(), options.disk);
    cache.limits = limits(options)?.cache;
    CacheStore::open(cache)
}
fn cached(store: &CacheStore, name: &str) -> Result<SnapshotLease> {
    if name.len() == 64
        && name
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        store.open_snapshot(name)
    } else {
        store.resolve_ref(name)
    }
}
fn describe(lease: &SnapshotLease) -> Result<()> {
    use std::io::Write;
    std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
    eprintln!("Verified cached data; no model execution support or remote freshness inferred.");
    Ok(())
}
fn validate_cached(lease: &SnapshotLease, request: &PullRequest) -> Result<()> {
    let source = lease.manifest().source();
    if source.provider() != request.provider.label() || source.repository() != request.repository {
        return Err(Error::Malformed(
            "cached reference has a different provider/repository".into(),
        ));
    }
    let locator_ok = match &request.selection {
        Selection::Files { .. } => {
            source.revision().len() == 40
                && source
                    .revision()
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        Selection::ReleaseAsset { .. } => source
            .revision()
            .strip_prefix("release-")
            .and_then(|rest| rest.split_once("-asset-"))
            .is_some_and(|(release, asset)| {
                [release, asset].iter().all(|part| {
                    part.parse::<u64>()
                        .ok()
                        .is_some_and(|n| n > 0 && n.to_string() == *part)
                })
            }),
    };
    if !locator_ok {
        return Err(Error::Malformed(
            "cached reference has a different or unpinned locator kind".into(),
        ));
    }
    if matches!(request.selection, Selection::Files { .. })
        && request.revision.len() == 40
        && request.revision.bytes().all(|b| b.is_ascii_hexdigit())
        && !source.revision().eq_ignore_ascii_case(&request.revision)
    {
        return Err(Error::Malformed(
            "cached reference has a different pinned revision".into(),
        ));
    }
    let paths: Vec<String> = match &request.selection {
        Selection::Files { paths, subfolder } => paths
            .iter()
            .map(|path| match subfolder {
                Some(folder) => format!("{folder}/{path}"),
                None => path.clone(),
            })
            .collect(),
        Selection::ReleaseAsset { name } => vec![name.clone()],
    };
    if paths.iter().any(|path| {
        !lease
            .manifest()
            .entries()
            .iter()
            .any(|entry| entry.path() == path)
    }) {
        return Err(Error::Unsupported(
            "cached snapshot lacks required selected files".into(),
        ));
    }
    Ok(())
}
fn admission(memory: u64) -> Result<TransferAdmission> {
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "artifact-transfer-staging")?;
    // This synchronous CLI owns the fixed domain for its process lifetime.
    // It is not a global host-RAM, OS or cross-process reservation.
    arbiter.register(resource.clone(), memory)?;
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("artifact-acquisition", "cli")?,
    })
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let options = parse(args)?;
    match &options.action {
        Action::Help => {
            help();
            Ok(())
        }
        Action::List => {
            for digest in open(&options)?.list_snapshots()? {
                println!("{digest}");
            }
            Ok(())
        }
        Action::Inspect(name) | Action::Verify(name) => describe(&cached(&open(&options)?, name)?),
        Action::Fetch(request) => {
            let reference = options.reference.clone().unwrap_or_else(|| {
                format!(
                    "{}/{}/{}",
                    request.provider.label(),
                    request.repository,
                    request.revision
                )
            });
            // Offline is terminal before client, credentials, metadata, DNS or
            // transfer-admission construction. Local validation is still real.
            if options.offline {
                let lease = cached(&open(&options)?, &reference)?;
                validate_cached(&lease, request)?;
                return describe(&lease);
            }
            let (origin, token) = match request.provider {
                Provider::HuggingFace => ("https://huggingface.co", "HF_TOKEN"),
                Provider::GitHub => ("https://api.github.com", "GITHUB_TOKEN"),
            };
            let mut http_options = HttpOptions::default();
            if let Some(seconds) = options.timeout {
                http_options.total_timeout = std::time::Duration::from_secs(seconds);
            }
            let mut client =
                HttpsClient::from_env(origin, request.provider.label(), Some(token), http_options)?;
            if request.provider == Provider::GitHub {
                client = client.with_credential_origin("https://raw.githubusercontent.com")?;
            }
            let admission = admission(options.memory)?;
            let cancel = Arc::new(AtomicBool::new(false));
            let plan =
                acquisition::resolve(&client, request, &limits(&options)?, &admission, &cancel)?;
            if options.dry_run {
                println!(
                    "Provider: {}\nRepository: {}\nPinned revision: {}\nSelected bytes: {}",
                    plan.source().provider(),
                    plan.source().repository(),
                    plan.source().revision(),
                    plan.total_bytes()
                );
                for file in plan.files() {
                    println!(
                        "{} {} {:?}",
                        file.path(),
                        file.bytes(),
                        file.digest_provenance()
                    );
                }
                println!(
                    "DRY RUN: metadata only; no model-weight transfer/cache publication or inference qualification."
                );
                return Ok(());
            }
            let receipt = acquisition::execute(
                &open(&options)?,
                &plan,
                &client,
                &admission,
                &cancel,
                Some(&reference),
            )?;
            match receipt.outcome() {
                CommitOutcome::Durable(snapshot) => {
                    println!("Snapshot: {}\nReference: {reference}\nVerified data only; no inference eligibility implied.", snapshot.digest());
                    use std::io::Write;
                    std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;
                    eprintln!("Verified during acquisition; data only, not immutable against arbitrary same-UID writers or executable model readiness.");
                    Ok(())
                }
                CommitOutcome::DurabilityUnknown { .. } => Err(Error::State("complete cache binding is visible but durability is unknown; inspect before retrying".into())),
            }
        }
    }
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull hf REPO --revision REV --file PATH [--file PATH ...]
    zerogravity pull github OWNER/REPO --revision REV --file PATH
    zerogravity pull github OWNER/REPO --release TAG --asset NAME
    zerogravity pull list
    zerogravity pull inspect DIGEST_OR_REF
    zerogravity pull verify DIGEST_OR_REF

OPTIONS:
    --cache ABS_PATH           Override XDG cache root
                              Default: $XDG_CACHE_HOME/abird/zerogravity/models
                              (fallback $HOME/.cache/abird/zerogravity/models)
    --ref NAME                 Local reference (default provider/repo/requested-revision)
    --subfolder PATH           Paths are relative to this repository subfolder
    --dry-run                  Resolve metadata; no model-weight transfer/publication
    --offline                  Verified local cache only; no token/client/provider construction
    --disk-limit-bytes N       Cooperative cache-owned disk policy (default16GiB)
    --memory-limit-bytes N     This process transfer-staging cap (default128MiB, not all RAM)
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N
    --timeout-seconds N        Per HTTP operation total deadline (1..86400, default120)

Credentials: HF_TOKEN or GITHUB_TOKEN environment only; no token command-line option.
Select an exact weight/index or auxiliary path; same-repository required model
closure is added, and missing sidecars or ambiguous variants refuse. First-seen
SHA256 is weaker than a trusted provider digest. No eviction, A3 loader or new inference
format support; persistent cross-invocation resume is not implemented."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }
    #[test]
    fn exact_selection_and_release_are_explicit() {
        assert!(matches!(
            parse(&args(&[
                "github",
                "a/b",
                "--revision",
                "main",
                "--file",
                "x",
                "--file",
                "y"
            ]))
            .unwrap()
            .action,
            Action::Fetch(_)
        ));
        assert!(matches!(
            parse(&args(&[
                "github",
                "a/b",
                "--release",
                "v1",
                "--asset",
                "tiny.gguf"
            ]))
            .unwrap()
            .action,
            Action::Fetch(PullRequest {
                selection: Selection::ReleaseAsset { .. },
                ..
            })
        ));
        for parts in [
            vec!["hf", "a/b"],
            vec!["hf", "a/b", "--release", "v1", "--asset", "x"],
            vec!["github", "a/b", "--revision", "main", "--file", "../x"],
            vec!["list", "--revision", "main"],
            vec!["github", "a/b", "--token", "secret-value"],
        ] {
            let error = parse(&args(&parts)).unwrap_err().to_string();
            assert!(!error.contains("secret-value"));
        }
    }
    #[test]
    fn limits_duplicate_flags_and_relative_cache_refuse() {
        for parts in [
            vec!["list", "--cache", "relative"],
            vec!["list", "--max-pages", "0"],
            vec!["list", "--timeout-seconds", "0"],
            vec!["list", "--timeout-seconds", "86401"],
            vec!["list", "--offline", "--offline"],
            vec!["list", "--max-files", "999999"],
        ] {
            assert!(parse(&args(&parts)).and_then(|o| limits(&o)).is_err());
        }
    }
}
