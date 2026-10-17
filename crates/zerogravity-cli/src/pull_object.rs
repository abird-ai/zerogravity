//! Explicit content-bound HTTPS input; signed object URLs are never argv/output.
use super::{Action, Options, ProgressDisplay, admission, limits, signals};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition;
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::acquisition::object::HttpsObjectSpec;
use zerogravity::cache::{CacheOptions, CacheStore, CommitOutcome, FileRole};
use zerogravity::{Error, Result};
const URL_LIMIT: usize = 16 * 1024;
struct ObjectOptions {
    spec: HttpsObjectSpec,
    url_file: Option<String>,
    snapshot: Option<String>,
    token_env: Option<String>,
    common: Options,
}
fn invalid() -> Error {
    Error::Malformed("invalid pull object arguments; use pull object --help (object URLs and token values are never argv)".into())
}
fn parse(args: &[String]) -> Result<ObjectOptions> {
    if args.len() > 64 || args.iter().any(|arg| arg.len() > 4096) {
        return Err(invalid());
    }
    let mut common = Options {
        action: Action::Help,
        root: None,
        disk: super::DISK_DEFAULT,
        memory: super::MEMORY_DEFAULT,
        max_file: None,
        unknown_size_cap: None,
        max_snapshot: None,
        max_files: None,
        metadata: None,
        pages: None,
        timeout: None,
        reference: None,
        dry_run: false,
        offline: false,
    };
    let mut origin = None;
    let mut path = None;
    let mut role = None;
    let mut bytes = None;
    let mut sha = None;
    let mut url_file = None;
    let mut snapshot = None;
    let mut token_env = None;
    let mut seen = BTreeSet::new();
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        if !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--dry-run" => {
                common.dry_run = true;
                continue;
            }
            "--offline" => {
                common.offline = true;
                continue;
            }
            _ => {}
        }
        let value = rest.next().ok_or_else(invalid)?;
        let positive = || {
            value
                .parse::<u64>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(invalid)
        };
        match flag.as_str() {
            "--origin" => origin = Some(value.clone()),
            "--file" => path = Some(value.clone()),
            "--sha256" => sha = Some(value.clone()),
            "--bytes" => bytes = Some(value.parse::<u64>().map_err(|_| invalid())?),
            "--role" => {
                role = Some(match value.as_str() {
                    "weights" => FileRole::Weights,
                    "config" => FileRole::Config,
                    "tokenizer" => FileRole::Tokenizer,
                    "index" => FileRole::Index,
                    "license" => FileRole::License,
                    "auxiliary" => FileRole::Auxiliary,
                    _ => return Err(invalid()),
                })
            }
            "--url-file" => {
                if value != "-"
                    && (!Path::new(value).is_absolute() || value.chars().any(char::is_control))
                {
                    return Err(invalid());
                }
                url_file = Some(value.clone());
            }
            "--snapshot" => {
                if value.is_empty() || value.chars().any(char::is_control) {
                    return Err(invalid());
                }
                snapshot = Some(value.clone());
            }
            "--token-env" => {
                if value.is_empty()
                    || value.len() > 128
                    || !value
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
                {
                    return Err(invalid());
                }
                token_env = Some(value.clone());
            }
            "--cache" => {
                if !Path::new(value).is_absolute() || value.chars().any(char::is_control) {
                    return Err(invalid());
                }
                common.root = Some(value.into());
            }
            "--disk-limit-bytes" => common.disk = positive()?,
            "--memory-limit-bytes" => common.memory = positive()?,
            "--max-file-bytes" => common.max_file = Some(positive()?),
            "--max-snapshot-bytes" => common.max_snapshot = Some(positive()?),
            "--timeout-seconds" => {
                let n = positive()?;
                if n > 86400 {
                    return Err(invalid());
                }
                common.timeout = Some(n);
            }
            _ => return Err(invalid()),
        }
    }
    if (common.offline && (common.dry_run || snapshot.is_none()))
        || (!common.offline && (snapshot.is_some() || url_file.is_none()))
    {
        return Err(invalid());
    }
    let spec = HttpsObjectSpec::new(
        origin.ok_or_else(invalid)?,
        path.ok_or_else(invalid)?,
        role.ok_or_else(invalid)?,
        bytes.ok_or_else(invalid)?,
        sha.ok_or_else(invalid)?,
    )?;
    let bounds = limits(&common)?;
    if spec.bytes() > bounds.cache.max_file_bytes || spec.bytes() > bounds.cache.max_snapshot_bytes
    {
        return Err(Error::Limit(
            "object contract exceeds configured limits".into(),
        ));
    }
    Ok(ObjectOptions {
        spec,
        url_file,
        snapshot,
        token_env,
        common,
    })
}
fn read_url(input: &str) -> Result<String> {
    read_secret_input(input, URL_LIMIT)
}
pub(super) fn read_secret_input(input: &str, limit: usize) -> Result<String> {
    if limit == 0 || limit > URL_LIMIT {
        return Err(Error::Limit("invalid secret input bound".into()));
    }
    let mut bytes = Vec::with_capacity(limit + 1);
    if input == "-" {
        std::io::stdin()
            .lock()
            .take((limit + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::State("secret input could not be read".into()))?;
    } else {
        let fd = rustix::fs::open(
            input,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map_err(|_| Error::State("secret file could not be opened".into()))?;
        let file = File::from(fd);
        let metadata = file
            .metadata()
            .map_err(|_| Error::State("secret file could not be inspected".into()))?;
        if !metadata.is_file() || metadata.len() > limit as u64 {
            return Err(Error::Limit(
                "secret input must be a bounded regular file".into(),
            ));
        }
        file.take((limit + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::State("secret input could not be read".into()))?;
    }
    if bytes.len() > limit {
        return Err(Error::Limit("secret input exceeds its bound".into()));
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
    }
    let url =
        String::from_utf8(bytes).map_err(|_| Error::Malformed("invalid secret input".into()))?;
    if url.is_empty() || url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(Error::Malformed("invalid secret input".into()));
    }
    Ok(url)
}
fn cache_options(common: &Options) -> Result<CacheOptions> {
    let mut result = CacheOptions::new(common.root.clone(), common.disk);
    result.limits = limits(common)?.cache;
    Ok(result)
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        help();
        return Ok(());
    }
    let options = parse(args)?;
    let bounds = limits(&options.common)?;
    let cancel = Arc::new(AtomicBool::new(false));
    if options.common.offline {
        // No URL-file/stdin metadata, credential lookup/client or signal setup.
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = options.spec.open_cached(
            &store,
            options.snapshot.as_deref().ok_or_else(invalid)?,
            &bounds,
            &cancel,
        )?;
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached object contract; no URL read, remote freshness or model-readiness claim."
        );
        return Ok(());
    }
    let url = read_url(options.url_file.as_deref().ok_or_else(invalid)?)?;
    let admission = admission(options.common.memory)?;
    let plan = options.spec.prepare(&url, &bounds, &admission, &cancel)?;
    if options.common.dry_run {
        println!(
            "Provider: {}\nOrigin identity: {}\nContract: {}\nFile: {}\nRole: {:?}\nExact bytes: {}\nSHA256: {}\nDRY RUN: validated caller-supplied contract only; no remote metadata, payload transfer or cache publication.",
            plan.source().provider(),
            plan.source().repository(),
            plan.source().revision(),
            options.spec.path(),
            options.spec.role(),
            options.spec.bytes(),
            options.spec.sha256()
        );
        return Ok(());
    }
    let mut http = HttpOptions::default();
    if let Some(seconds) = options.common.timeout {
        http.total_timeout = std::time::Duration::from_secs(seconds);
    }
    let client = HttpsClient::from_env(
        options.spec.origin(),
        "https-object",
        options.token_env.as_deref(),
        http,
    )?;
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let store = CacheStore::open(cache_options(&options.common)?)?;
    let mut display = ProgressDisplay::default();
    let receipt = acquisition::execute_with_progress(
        &store,
        &plan,
        &client,
        &admission,
        &cancel,
        None,
        &mut |event| {
            let _ = display.report(event, &mut std::io::stderr());
        },
    )?;
    match receipt.outcome() {
        CommitOutcome::Durable(snapshot) => {
            println!(
                "Snapshot: {}\nVerified caller-digest object data only; no repository closure or inference eligibility implied.",
                snapshot.digest()
            );
            std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;
            Ok(())
        }
        CommitOutcome::DurabilityUnknown { .. } => Err(Error::State(
            "complete object binding is visible but durability is unknown; inspect before retrying"
                .into(),
        )),
    }
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull object --origin https://host --file LOGICAL --role ROLE \\
        --bytes N --sha256 DIGEST --url-file ABS_PATH|- [--dry-run]
    zerogravity pull object --origin https://host --file LOGICAL --role ROLE \\
        --bytes N --sha256 DIGEST --offline --snapshot DIGEST_OR_REF

The origin is a strict bare HTTPS origin (no path/query/userinfo); it is identity,
not the object URL. Object URLs and token values are never command-line arguments.
--url-file is an explicitly selected regular file or stdin, bounded16KiB, with at
most one final LF/CRLF. No FIFO/leaf-symlink/device input. Explicit stdin can block.
ROLE: weights/config/tokenizer/index/license/auxiliary. Exact bytes may be zero.
The caller supplies an authorized raw SHA256; ETags/Git IDs/URLs do not supply it.

OPTIONS:
    --cache ABS_PATH --disk-limit-bytes N --memory-limit-bytes N
    --max-file-bytes N --max-snapshot-bytes N --timeout-seconds N (1..86400)
    --token-env NAME            Optional origin-scoped credential environment name
    --dry-run                   Validate caller contract/URL only; no network or cache writes
    --offline --snapshot ID     Verify exact cached contract before URL/token/client access

Online defaults to apply and publishes no mutable reference. Offline may omit
URL input entirely and never fetches; existing cached references are valid selectors.
Signed URL path/query and credentials are not printed or stored in cache metadata.
Source identities derive from canonical origin and the exact content contract.
Uses existing HTTPS/redirect/deadline/resume/cancel/progress and admitted publisher.
No source authenticity beyond TLS/caller SHA, repository closure, license, model
format or execution support is granted. SIGINT/SIGTERM cancellation is cooperative;
blocked HTTP reads can take their configured whole-request deadline."
    );
}
