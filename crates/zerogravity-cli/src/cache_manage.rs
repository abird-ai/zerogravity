//! Explicit offline exact-target cache maintenance; no transport or inference.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use zerogravity::cache::{CacheOptions, CacheStore, ReclaimMode, ReclaimOutcome, ReclaimReport};
use zerogravity::{Error, Result};

const DEFAULT_DISK_LIMIT: u64 = 16 * 1024 * 1024 * 1024;

struct Options {
    digest: String,
    root: PathBuf,
    mode: ReclaimMode,
    disk: u64,
}

fn invalid() -> Error {
    Error::Malformed("invalid cache arguments; use cache --help".into())
}

fn parse(args: &[String]) -> Result<Options> {
    if args.len() < 3 || args.len() > 16 || args.iter().any(|arg| arg.len() > 4096) {
        return Err(invalid());
    }
    if args[0] != "remove" {
        return Err(invalid());
    }
    zerogravity::cache::paths::validate_digest(&args[1]).map_err(|_| invalid())?;
    let mut seen = BTreeSet::new();
    let mut root = None;
    let mut mode = None;
    let mut disk = DEFAULT_DISK_LIMIT;
    let mut rest = args[2..].iter();
    while let Some(flag) = rest.next() {
        if !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--dry-run" | "--apply" => {
                if mode.is_some() {
                    return Err(invalid());
                }
                mode = Some(if flag == "--dry-run" {
                    ReclaimMode::DryRun
                } else {
                    ReclaimMode::Apply
                });
            }
            "--cache" => {
                let value = rest.next().ok_or_else(invalid)?;
                if !Path::new(value).is_absolute() || value.chars().any(char::is_control) {
                    return Err(invalid());
                }
                root = Some(PathBuf::from(value));
            }
            "--disk-limit-bytes" => {
                disk = rest
                    .next()
                    .ok_or_else(invalid)?
                    .parse::<u64>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(invalid)?;
            }
            _ => return Err(invalid()),
        }
    }
    Ok(Options {
        digest: args[1].clone(),
        root: root.ok_or_else(invalid)?,
        mode: mode.ok_or_else(invalid)?,
        disk,
    })
}

fn describe(report: &ReclaimReport) {
    println!(
        "Snapshot: {}\nRetirement began or resumed: {}\nCandidate named entries: {} files, {} directories\nRemoved named entries: {} files, {} directories\nRetained shared blobs: {}\nCharged cache bytes before: {}",
        report.snapshot(),
        report.retired(),
        report.candidate_files(),
        report.candidate_directories(),
        report.removed_files(),
        report.removed_directories(),
        report.retained_blobs(),
        report.used_before_bytes(),
    );
    match report.used_after_bytes() {
        Some(after) => {
            println!("Charged cache bytes after: {after}");
            if after <= report.used_before_bytes() {
                println!(
                    "Observed charged-byte reduction: {}",
                    report.used_before_bytes() - after
                );
            } else {
                println!(
                    "Observed charged-byte increase: {}",
                    after - report.used_before_bytes()
                );
            }
        }
        None => println!("Charged cache bytes after: unknown"),
    }
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if (args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help"))
        || (args.len() == 2
            && args[0] == "remove"
            && matches!(args[1].as_str(), "--help" | "-h" | "help"))
    {
        help();
        return Ok(());
    }
    let options = parse(args)?;
    let store = CacheStore::open_existing(CacheOptions::new(Some(options.root), options.disk))?;
    // Local maintenance has no credential/client/provider or online signal setup,
    // even in a binary that also has acquisition enabled.
    match store.reclaim_snapshot(&options.digest, options.mode, &AtomicBool::new(false))? {
        ReclaimOutcome::DryRun(report) => {
            describe(&report);
            println!("DRY RUN: observation only; apply revalidates. No target entries removed.");
            Ok(())
        }
        ReclaimOutcome::Durable(report) => {
            describe(&report);
            println!(
                "Removal completed with directory syncs; measured cache charge is not a filesystem-free-space guarantee."
            );
            Ok(())
        }
        ReclaimOutcome::Busy => Err(Error::State(
            "cache removal is busy; retry after active users close".into(),
        )),
        ReclaimOutcome::Referenced { references } => Err(Error::State(format!(
            "snapshot is retained by {references} references; no reference was removed"
        ))),
        ReclaimOutcome::Incomplete { report, cause } => {
            describe(&report);
            eprintln!(
                "INCOMPLETE: target may be retired; retry the same digest after inspection. Completed changes were not rolled back."
            );
            Err(cause)
        }
    }
}

fn help() {
    println!(
        "USAGE:
    zerogravity cache remove DIGEST --dry-run|--apply --cache ABS_PATH

OPTIONS:
    --dry-run                 Observe exact-target eligibility; no target removal
    --apply                   Revalidate and remove the exact unreferenced snapshot
    --cache ABS_PATH          Required existing cache root; never creates a root
    --disk-limit-bytes N      Existing cooperative disk policy (default16GiB)

Linux cache-only operation: no network, credentials, provider or signal handlers.
Only fresh protected-v2 roots support removal; legacy roots are not migrated.
DIGEST is an exact lowercase SHA256 snapshot identity, not a reference or path.
Referenced or busy targets refuse. Shared blobs remain for other snapshots.
An interrupted removal can leave a named, charged retired target; retry only the
same digest. No automatic eviction, reference deletion, orphan sweep or force-live
cleanup. Concurrent loaders must use run-cache or managed snapshot handles;
raw reads need external quiescence. Counts describe named entries and observed
cache-owned charge, not physical bytes freed or filesystem free space.
Dry-run is not a reservation."
    );
}
