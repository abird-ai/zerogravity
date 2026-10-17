//! Explicit offline local-cache import; no transport/client/token construction.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::cache::import_hf::{HfImportEntry, HfImportRequest};
use zerogravity::cache::{CacheLimits, CacheOptions, CommitOutcome, FileRole, TransferAdmission};
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity::{Error, Result};
const MEMORY_DEFAULT: u64 = 128 * 1024 * 1024;
struct Options {
    request: HfImportRequest,
    cache: CacheOptions,
    memory: u64,
    apply: bool,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid cache import-hf arguments; use cache import-hf --help (no credentials required)"
            .into(),
    )
}
fn absolute(value: &str) -> Result<PathBuf> {
    if !Path::new(value).is_absolute() || value.chars().any(char::is_control) {
        return Err(invalid());
    }
    Ok(value.into())
}
fn parse(args: &[String]) -> Result<Options> {
    if args.len() > 16_448
        || args.iter().any(|arg| arg.len() > 8192)
        || args
            .iter()
            .try_fold(0usize, |n, a| n.checked_add(a.len()))
            .is_none_or(|n| n > 8 * 1024 * 1024)
    {
        return Err(invalid());
    }
    let mut source = None;
    let mut repository = None;
    let mut commit = None;
    let mut root = None;
    let mut apply = None;
    let mut disk = super::DEFAULT_DISK_LIMIT;
    let mut memory = MEMORY_DEFAULT;
    let mut limits = CacheLimits::default();
    let mut files = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        if flag != "--file" && flag != "--sha256" && !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        if matches!(flag.as_str(), "--dry-run" | "--apply") {
            if apply.is_some() {
                return Err(invalid());
            }
            apply = Some(flag == "--apply");
            continue;
        }
        let value = rest.next().ok_or_else(invalid)?;
        let number = || {
            value
                .parse::<u64>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(invalid)
        };
        match flag.as_str() {
            "--source" => source = Some(absolute(value)?),
            "--cache" => root = Some(absolute(value)?),
            "--repository" => repository = Some(value.clone()),
            "--commit" => commit = Some(value.clone()),
            "--file" => {
                let (role, path) = value.split_once(':').ok_or_else(invalid)?;
                let role = match role {
                    "weights" => FileRole::Weights,
                    "config" => FileRole::Config,
                    "tokenizer" => FileRole::Tokenizer,
                    "index" => FileRole::Index,
                    "license" => FileRole::License,
                    "auxiliary" => FileRole::Auxiliary,
                    _ => return Err(invalid()),
                };
                zerogravity::cache::paths::validate_relative_path(path).map_err(|_| invalid())?;
                if files.len() >= 4096 || files.insert(path.to_string(), role).is_some() {
                    return Err(invalid());
                }
            }
            "--sha256" => {
                let (path, sha) = value.rsplit_once('=').ok_or_else(invalid)?;
                zerogravity::cache::paths::validate_relative_path(path).map_err(|_| invalid())?;
                zerogravity::cache::paths::validate_digest(sha).map_err(|_| invalid())?;
                if hashes.len() >= 4096
                    || hashes.insert(path.to_string(), sha.to_string()).is_some()
                {
                    return Err(invalid());
                }
            }
            "--disk-limit-bytes" => disk = number()?,
            "--memory-limit-bytes" => memory = number()?,
            "--max-file-bytes" => limits.max_file_bytes = number()?,
            "--max-snapshot-bytes" => limits.max_snapshot_bytes = number()?,
            "--max-files" => {
                limits.max_files = usize::try_from(number()?).map_err(|_| invalid())?
            }
            "--max-manifest-bytes" => {
                limits.max_manifest_bytes = usize::try_from(number()?).map_err(|_| invalid())?
            }
            _ => return Err(invalid()),
        }
    }
    limits.validate()?;
    if files.is_empty()
        || files.len() > limits.max_files
        || hashes.keys().any(|path| !files.contains_key(path))
    {
        return Err(invalid());
    }
    let entries = files
        .into_iter()
        .map(|(path, role)| {
            let hash = hashes.remove(&path);
            HfImportEntry::new(path, role, hash)
        })
        .collect::<Result<Vec<_>>>()?;
    let request = HfImportRequest::new(
        source.ok_or_else(invalid)?,
        repository.ok_or_else(invalid)?,
        commit.ok_or_else(invalid)?,
        entries,
    )?;
    let mut cache = CacheOptions::new(Some(root.ok_or_else(invalid)?), disk);
    cache.limits = limits;
    Ok(Options {
        request,
        cache,
        memory,
        apply: apply.ok_or_else(invalid)?,
    })
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        help();
        return Ok(());
    }
    let options = parse(args)?;
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "artifact-transfer-staging")?;
    arbiter.register(resource.clone(), options.memory)?;
    let admission = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("local-hf-cache-import", "cli")?,
    };
    let plan = options.request.prepare(
        &options.cache.limits,
        &admission,
        Arc::new(AtomicBool::new(false)),
    )?;
    plan.check_destination(&options.cache)?;
    println!(
        "Local source repository: {}\nDeclared commit: {}\nSelected bytes: {}",
        plan.source().repository(),
        plan.source().revision(),
        plan.total_bytes()
    );
    if !options.apply {
        for entry in plan.entries() {
            println!(
                "{} {} {:?} {}",
                entry.path(),
                entry.bytes(),
                entry.role(),
                if entry.expected_sha256().is_some() {
                    "caller SHA256 supplied; not yet verified"
                } else {
                    "first-seen; not yet hashed"
                }
            );
        }
        println!(
            "DRY RUN: local metadata only; no payload copy or destination publication. Local cache names do not prove remote origin or model readiness."
        );
        return Ok(());
    }
    let receipt = plan.apply(options.cache)?;
    for entry in receipt.entries() {
        println!(
            "{} {} {} {:?} copied={}",
            entry.path(),
            entry.bytes(),
            entry.digest(),
            entry.digest_provenance(),
            entry.copied_bytes()
        );
    }
    match receipt.into_outcome() {
        CommitOutcome::Durable(snapshot) => println!(
            "Snapshot: {}\nImported verified local data only; no remote-origin or model-readiness claim.",
            snapshot.digest()
        ),
        CommitOutcome::DurabilityUnknown {
            snapshot, cause, ..
        } => {
            eprintln!(
                "Snapshot {} may be installed but durability is unknown; inspect before retrying.",
                snapshot.digest()
            );
            return Err(cause);
        }
    }
    Ok(())
}
fn help() {
    println!(
        "USAGE:
    zerogravity cache import-hf --source ABS_REPO_CACHE --repository owner/repo \\
        --commit EXACT_COMMIT --file ROLE:PATH [--file ROLE:PATH ...] \\
        [--sha256 PATH=DIGEST ...] --dry-run|--apply --cache ABS_EXISTING_DEST

Import selected files from one local Hugging Face repository-cache snapshot.
The source basename must be models--owner--repo for the declared owner/repo.
EXACT_COMMIT is 40 lowercase hexadecimal characters; no refs/aliases/globs.
ROLE: weights, config, tokenizer, index, license or auxiliary. Paths are exact.
--sha256 supplies a caller-trusted raw digest; blob names and Git OIDs do not.

OPTIONS:
    --dry-run                  Read bounded metadata; no payload copy/publication
    --apply                    Copy/hash through the existing admitted cache publisher
    --cache ABS_EXISTING_DEST  Required existing 0700 destination; never creates a root
    --disk-limit-bytes N       Cooperative destination policy (default16GiB)
    --memory-limit-bytes N     Process transfer/control staging (default128MiB, not all RAM)
    --max-file-bytes N --max-snapshot-bytes N --max-files N --max-manifest-bytes N

Offline cache-only operation: no network, credentials or online signal handlers.
Source and destination must be disjoint; aliases/overlap or uncertain mount identity refuse.
The source is never locked, chmodded, linked into the destination or deleted.
Untrusted source bytes are first-seen unless an explicit caller SHA256 verifies.
A declared local commit is not authenticated remote membership; import neither
executes repository code nor grants model/format compatibility or inference support.
No reference is changed. Source mutation, mismatch or incomplete publication refuses."
    );
}
