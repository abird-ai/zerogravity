//! Native ModelScope explicit-commit acquisition through the shared cold owner.
use super::{Action, Options, ProgressDisplay, admission, limits, signals};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::acquisition::{self, modelscope::ModelScopeRequest};
use zerogravity::cache::{CacheOptions, CacheStore, CommitOutcome};
use zerogravity::{Error, Result};
struct ModelScopeOptions {
    request: ModelScopeRequest,
    token_env: Option<String>,
    snapshot: Option<String>,
    common: Options,
}
fn invalid() -> Error {
    Error::Malformed("invalid pull modelscope arguments; use pull modelscope --help (tokens are environment-only)".into())
}
fn parse(args: &[String]) -> Result<ModelScopeOptions> {
    if args.len() > 8192 || args.iter().any(|s| s.len() > 4096) {
        return Err(invalid());
    }
    let repository = args.first().ok_or_else(invalid)?;
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
    let (mut commit, mut subfolder, mut token_env, mut snapshot) = (None, None, None, None);
    let mut official_token = false;
    let mut paths = Vec::new();
    let mut seen = BTreeSet::new();
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        if flag != "--file" && !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--offline" => {
                common.offline = true;
                continue;
            }
            "--dry-run" => {
                common.dry_run = true;
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
        let count = || usize::try_from(positive()?).map_err(|_| invalid());
        match flag.as_str() {
            "--commit" => commit = Some(value.clone()),
            "--file" => paths.push(value.clone()),
            "--subfolder" => subfolder = Some(value.clone()),
            "--profile" => {
                official_token = match value.as_str() {
                    "anonymous" => false,
                    "official-token" => true,
                    _ => return Err(invalid()),
                }
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
            "--snapshot" => snapshot = Some(value.clone()),
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
            "--max-files" => common.max_files = Some(count()?),
            "--max-metadata-bytes" => common.metadata = Some(count()?),
            "--max-pages" => common.pages = Some(count()?),
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
    if official_token != token_env.is_some()
        || (common.offline && (common.dry_run || snapshot.is_none()))
        || (!common.offline && snapshot.is_some())
    {
        return Err(invalid());
    }
    let request = ModelScopeRequest::new(
        "https://modelscope.cn",
        repository,
        &commit.ok_or_else(invalid)?,
        paths,
        subfolder,
    )?;
    if let Some(snapshot) = &snapshot
        && (snapshot.len() != 64
            || !snapshot
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    {
        return Err(invalid());
    }
    limits(&common)?;
    Ok(ModelScopeOptions {
        request,
        token_env,
        snapshot,
        common,
    })
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
    // No credential environment, client, provider, signal handler or admission
    // is touched by the terminal offline branch. Misses never acquire data.
    if options.common.offline {
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = options.request.open_cached(
            &store,
            options.snapshot.as_deref().ok_or_else(invalid)?,
            &bounds,
            &cancel,
        )?;
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached ModelScope commit/data only; no remote freshness, account authorization or inference claim."
        );
        return Ok(());
    }
    let admission = admission(options.common.memory)?;
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let mut http = HttpOptions::default();
    if let Some(seconds) = options.common.timeout {
        http.total_timeout = std::time::Duration::from_secs(seconds);
    }
    let client = if let Some(name) = &options.token_env {
        let token = std::env::var(name).map_err(|_| {
            Error::State("ModelScope official-token environment value unavailable".into())
        })?;
        HttpsClient::with_modelscope_token(options.request.origin(), &token, http)?
    } else {
        HttpsClient::new(options.request.origin(), http)?
    };
    let plan = options
        .request
        .resolve(&client, &bounds, &admission, &cancel)?;
    if options.common.dry_run {
        println!(
            "Provider: modelscope\nRepository: {}\nExact commit: {}",
            options.request.repository(),
            options.request.commit()
        );
        for file in plan.files() {
            println!(
                "File: {}\nExact bytes: {}\nRaw digest provenance: {:?}",
                file.path(),
                super::selected_bytes(file.bytes(), file.maximum_bytes()),
                file.digest_provenance()
            );
        }
        println!(
            "DRY RUN: complete bounded tree and index metadata only; no model-weight transfer or cache publication. Explicit commit trusts the official endpoint; no local Git/Merkle or inference qualification."
        );
        return Ok(());
    }
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
            println!("Snapshot: {}\nVerified during acquisition; pinned repository data only, no model readiness or license approval.", snapshot.digest());
            std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?; Ok(())
        }
        CommitOutcome::DurabilityUnknown { .. } => Err(Error::State("complete ModelScope data is visible but durability is unknown; inspect before retrying".into())),
    }
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull modelscope OWNER/REPO --commit FULL_LOWERCASE_40HEX \\
        --file PATH [--file PATH ...] [--subfolder PATH] [--dry-run]
    zerogravity pull modelscope [same request] --offline --snapshot DIGEST

OPTIONS:
    --profile anonymous|official-token  Default anonymous; no ambient SDK credentials
    --token-env NAME             Required only for official-token (Bearer + m_session_id)
    --dry-run                    Complete bounded tree and index metadata only
    --offline --snapshot DIGEST  Verified cached data before credentials/client/signals
    --cache ABS_PATH             XDG default abird/zerogravity/models or explicit root
    --disk-limit-bytes N         Cooperative cache policy (default16GiB)
    --memory-limit-bytes N       Process transfer-staging cap (default128MiB, not all RAM)
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N --timeout-seconds N

Only https://modelscope.cn model repositories and explicit commits are supported.
Branches/tags, mirror endpoints, capped bodies and mutable refs are unsupported.
Missing selected file/index sizes use bounded body-free HEAD at the same full
commit. Only exact HTTP200 identity lengths are accepted; missing/error/redirect
metadata refuses without a GET fallback. Metadata and payload redirects refuse. A directory with 3000
or more entries refuses because the official shallow protocol silently truncates;
request and byte limits cover both complete tree walking and index metadata.
Missing/ambiguous selections and required model closure refuse. Row SHA256 is a
trusted expected raw digest when present; otherwise bytes are FirstSeen. An
explicit commit trusts the endpoint selector, not locally verified Git objects.
Official tokens must already exist; no SDK state, login, refresh or cookie jar.
Offline data proves no remote freshness/private access or inference support."
    );
}
