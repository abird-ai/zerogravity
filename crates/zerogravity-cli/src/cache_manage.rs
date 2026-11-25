//! Explicit offline exact-target cache maintenance; no transport or inference.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use zerogravity::cache::{
    ByNameKind, CacheOptions, CacheStore, PublishedSnapshot, ReclaimMode, ReclaimOutcome,
    ReclaimReport,
};
use zerogravity::{Error, Result};

#[path = "cache_import.rs"]
mod import;

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
    if let Some(warning) = report.by_name_warning() {
        crate::output::emit!(
            Warning,
            "canonical cache maintenance completed but by-name refresh needs repair: {warning}"
        );
    }
}

fn by_name_store(root: PathBuf, disk: u64) -> Result<CacheStore> {
    CacheStore::open_existing(CacheOptions::new(Some(root), disk))
}

fn parse_by_name_common(args: &[String]) -> Result<(PathBuf, u64, bool)> {
    let mut root = None;
    let mut disk = DEFAULT_DISK_LIMIT;
    let mut replace = false;
    let mut seen = BTreeSet::new();
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        if !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
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
            "--replace" => replace = true,
            _ => return Err(invalid()),
        }
    }
    Ok((root.ok_or_else(invalid)?, disk, replace))
}

fn print_by_name_entries(entries: &[zerogravity::cache::ByNameEntry]) {
    for entry in entries {
        let kind = match entry.kind() {
            ByNameKind::Automatic => "auto",
            ByNameKind::Alias => "alias",
        };
        println!("{}  {}  {}", entry.name(), entry.snapshot(), kind);
    }
}

fn print_by_name_update(update: &zerogravity::cache::ByNameUpdate) {
    if let Some(snapshot) = update.snapshot() {
        println!(
            "by-name/{} -> ../snapshots/{}{}",
            update.name(),
            snapshot,
            if update.changed() { "" } else { " (unchanged)" }
        );
    } else {
        println!(
            "by-name/{} {}",
            update.name(),
            if update.changed() {
                "removed"
            } else {
                "was not an explicit alias"
            }
        );
    }
    if let Some(warning) = update.projection_warning() {
        crate::output::emit!(
            Warning,
            "human alias metadata is saved but by-name projection needs refresh: {warning}"
        );
    }
}

pub(super) fn report_publication(snapshot: &PublishedSnapshot) {
    for name in snapshot.by_name() {
        crate::output::emit!(Status, "By-name: by-name/{name}");
    }
    if let Some(warning) = snapshot.by_name_warning() {
        crate::output::emit!(
            Warning,
            "canonical snapshot is durable but by-name projection needs refresh: {warning}"
        );
    }
}

fn run_by_name(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        help();
        return Ok(());
    }
    if args.is_empty() || args.len() > 12 || args.iter().any(|arg| arg.len() > 4096) {
        return Err(invalid());
    }

    match args[0].as_str() {
        "list" => {
            let (root, disk, replace) = parse_by_name_common(&args[1..])?;
            if replace {
                return Err(invalid());
            }
            let entries = by_name_store(root, disk)?.list_by_name()?;
            print_by_name_entries(&entries);
            Ok(())
        }
        "refresh" | "--refresh" => {
            let (root, disk, replace) = parse_by_name_common(&args[1..])?;
            if replace {
                return Err(invalid());
            }
            let entries = by_name_store(root, disk)?.refresh_by_name()?;
            println!(
                "Refreshed {} human-only by-name entries from canonical snapshots and human alias metadata.",
                entries.len()
            );
            print_by_name_entries(&entries);
            Ok(())
        }
        "reset" => {
            if args[1..]
                .iter()
                .filter(|arg| arg.as_str() == "--discard-aliases")
                .count()
                != 1
            {
                return Err(invalid());
            }
            let filtered = args[1..]
                .iter()
                .filter(|arg| arg.as_str() != "--discard-aliases")
                .cloned()
                .collect::<Vec<_>>();
            let (root, disk, replace) = parse_by_name_common(&filtered)?;
            if replace {
                return Err(invalid());
            }
            let entries = by_name_store(root, disk)?.reset_by_name()?;
            println!(
                "Reset human-only by-name state; explicit aliases were discarded. Rebuilt {} automatic entries.",
                entries.len()
            );
            print_by_name_entries(&entries);
            Ok(())
        }
        "set" => {
            if args.len() < 4 {
                return Err(invalid());
            }
            let name = &args[1];
            let snapshot = &args[2];
            zerogravity::cache::paths::validate_digest(snapshot).map_err(|_| invalid())?;
            let (root, disk, replace) = parse_by_name_common(&args[3..])?;
            let update = by_name_store(root, disk)?.set_by_name_alias(name, snapshot, replace)?;
            print_by_name_update(&update);
            Ok(())
        }
        "remove" => {
            if args.len() < 3 {
                return Err(invalid());
            }
            let name = &args[1];
            let (root, disk, replace) = parse_by_name_common(&args[2..])?;
            if replace {
                return Err(invalid());
            }
            let update = by_name_store(root, disk)?.remove_by_name_alias(name)?;
            print_by_name_update(&update);
            Ok(())
        }
        _ => Err(invalid()),
    }
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if args.first().is_some_and(|arg| arg == "import-hf") {
        return import::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "by-name") {
        return run_by_name(&args[1..]);
    }
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
            crate::output::emit!(
                Warning,
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
    zerogravity cache by-name list --cache ABS_PATH
    zerogravity cache by-name refresh --cache ABS_PATH
    zerogravity cache by-name reset --discard-aliases --cache ABS_PATH
    zerogravity cache by-name set NAME DIGEST [--replace] --cache ABS_PATH
    zerogravity cache by-name remove NAME --cache ABS_PATH
    zerogravity cache import-hf --help

OPTIONS:
    --dry-run                 Observe exact-target eligibility; no target removal
    --apply                   Revalidate and remove the exact unreferenced snapshot
    --cache ABS_PATH          Required existing cache root; never creates a root
    --disk-limit-bytes N      Existing cooperative disk policy (default16GiB)
    by-name list              List intended human names without model loading
    by-name refresh           Reconcile snapshots and atomically rebuild symlinks
    by-name reset             With --discard-aliases, rebuild human state from
                              canonical snapshots after state loss/corruption
    by-name set NAME DIGEST   Create an explicit human alias; --replace is required
                              to retarget an existing explicit alias
    by-name remove NAME       Remove only an explicit human alias

by-name is one flat lowercase namespace. Each immutable snapshot gets one stable
model-first automatic name (for example qwen3.5-0.8b-qwen-hf-2fc06364).
Explicit aliases are stored separately from runtime refs. Automatic collisions
preserve the existing primary name and suffix only the newcomer. Reset is
deliberately destructive only to explicit human aliases and requires
--discard-aliases. Every symlink
targets ../snapshots/DIGEST. Runtime selection, provenance, quota and reclamation
authority never read by-name or its human metadata.

Linux cache-only operation: no network, credentials, provider or signal handlers.
import-hf explicitly copies selected local HF snapshot data; see its own help.
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
