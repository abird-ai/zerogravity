//! Metadata-only file choices, using the existing admitted provider HTTP owner.
use super::{Action, Options, admission, limits, signals};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::acquisition::{
    CandidateCoverage, CandidateRequest, Provider, RepositoryCandidates, discover_candidates,
    inspect_cached_candidates,
};
use zerogravity::cache::{CacheOptions, CacheStore};
use zerogravity::{Error, Result};

struct Arguments {
    request: CandidateRequest,
    common: Options,
    snapshot: Option<String>,
}
fn invalid() -> Error {
    Error::Malformed("invalid pull candidates arguments; use pull candidates --help (tokens are environment-only)".into())
}
fn parse(args: &[String]) -> Result<Arguments> {
    if args.len() < 2 || args.len() > 8192 || args.iter().any(|arg| arg.len() > 4096) {
        return Err(invalid());
    }
    let provider = match args[0].as_str() {
        "hf" => Provider::HuggingFace,
        "github" => Provider::GitHub,
        _ => return Err(invalid()),
    };
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
    let (mut revision, mut subfolder, mut snapshot) = (None, None, None);
    let mut seen = BTreeSet::new();
    let mut rest = args[2..].iter();
    while let Some(flag) = rest.next() {
        if !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        if flag == "--offline" {
            common.offline = true;
            continue;
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
            "--revision" => revision = Some(value.as_str()),
            "--subfolder" => subfolder = Some(value.as_str()),
            "--snapshot" => {
                if value.is_empty() || value.chars().any(char::is_control) {
                    return Err(invalid());
                }
                snapshot = Some(value.clone());
            }
            "--cache" => {
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    return Err(invalid());
                }
                common.root = Some(path);
            }
            "--disk-limit-bytes" => common.disk = positive()?,
            "--memory-limit-bytes" => common.memory = positive()?,
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
    if common.offline != snapshot.is_some()
        || !common.offline && (common.root.is_some() || seen.contains("--disk-limit-bytes"))
    {
        return Err(invalid());
    }
    let request =
        CandidateRequest::new(provider, &args[1], revision.ok_or_else(invalid)?, subfolder)?;
    limits(&common)?;
    Ok(Arguments {
        request,
        common,
        snapshot,
    })
}
fn display(result: &RepositoryCandidates) -> Result<()> {
    let mut out = std::io::stdout().lock();
    let emit = |out: &mut dyn Write, value: &serde_json::Value| {
        serde_json::to_writer(out, value)
            .map_err(|_| Error::State("candidate output failed".into()))
    };
    out.write_all(b"{\"source\":")?;
    emit(
        &mut out,
        &serde_json::json!({"provider":result.source().provider(),
        "repository":result.source().repository(),"revision":result.source().revision()}),
    )?;
    out.write_all(b",\"coverage\":")?;
    emit(
        &mut out,
        &serde_json::json!(match result.coverage() {
            CandidateCoverage::PinnedRepositoryTree => "pinned-remote-tree",
            CandidateCoverage::VerifiedCacheSubset => "verified-cached-subset",
        }),
    )?;
    out.write_all(b",\"files\":[")?;
    let size_basis = match (result.coverage(), result.source().provider()) {
        (CandidateCoverage::VerifiedCacheSubset, _) => "verified-retained-content",
        (_, "github") => "stored-git-blob",
        _ => "hf-tree-or-lfs-metadata",
    };
    for (index, file) in result.files().iter().enumerate() {
        if index > 0 {
            out.write_all(b",")?;
        }
        emit(
            &mut out,
            &serde_json::json!({"path":file.path(),"bytes":file.bytes(),
            "role_hint":file.role(),"expected_sha256":file.sha256(),"size_basis":size_basis}),
        )?;
    }
    out.write_all(b"],\"discovery_only\":true}\n")?;
    eprintln!(
        "File discovery only; paths are repository-relative. No payload/HEAD/index transfer or cache publication; no quant, model-family completeness or inference qualification. Cached coverage is only retained data, not remote freshness or current access."
    );
    Ok(())
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "help" | "--help" | "-h") {
        help();
        return Ok(());
    }
    let args = parse(args)?;
    let limits = limits(&args.common)?;
    if args.common.offline {
        let mut cache = CacheOptions::new(args.common.root.clone(), args.common.disk);
        cache.limits = limits.cache;
        let store = CacheStore::open_existing(cache)?;
        let lease = super::cached(&store, args.snapshot.as_deref().ok_or_else(invalid)?)?;
        return display(&inspect_cached_candidates(
            &lease,
            &args.request,
            &limits,
            &AtomicBool::new(false),
        )?);
    }
    let (origin, token) = match args.request.provider() {
        Provider::HuggingFace => ("https://huggingface.co", "HF_TOKEN"),
        Provider::GitHub => ("https://api.github.com", "GITHUB_TOKEN"),
    };
    let mut options = HttpOptions::default();
    if let Some(seconds) = args.common.timeout {
        options.total_timeout = std::time::Duration::from_secs(seconds);
    }
    let client = HttpsClient::from_env(
        origin,
        args.request.provider().label(),
        Some(token),
        options,
    )?;
    let admission = admission(args.common.memory)?;
    let cancel = Arc::new(AtomicBool::new(false));
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    display(&discover_candidates(
        &client,
        &args.request,
        &limits,
        &admission,
        &cancel,
    )?)
}
fn help() {
    println!("USAGE:
    zerogravity pull candidates hf|github OWNER/REPO --revision REV [--subfolder PATH]
    zerogravity pull candidates hf|github OWNER/REPO --revision REV --offline --snapshot DIGEST_OR_REF [--cache ABS]

Metadata-only file discovery, including multiple GGUF choices. Paths retain their
repository-relative bytes. Use the printed exact commit and --file PATH in the
existing pull command; no quant/default/fallback or split/model completeness is chosen.

OPTIONS:
    --subfolder PATH           Component-boundary repository filter, recursive within it
    --max-files N --max-metadata-bytes N --max-pages N
    --memory-limit-bytes N     Metadata staging cap (default128MiB, not all RAM)
    --timeout-seconds N        Original finite per-HTTP-operation deadline (1..86400)
    --offline --snapshot ID    Full verified retained subset before credentials/client/admission/signals
    --cache ABS --disk-limit-bytes N   Offline existing root/policy only

Online uses existing origin-scoped HF_TOKEN/GITHUB_TOKEN environment inputs only.
No token argv, provider endpoints, --file/--quant/exclude, body caps or mutable refs.
No HEAD, index or model-body transfer and no cache publication. Unknown bytes stay
unknown; listed sizes are metadata, not a payload reservation or transfer-cap test.
GitHub sizes describe stored Git blobs, possibly pointers, not resolved model bytes;
Git object IDs are never raw SHA256. HF sizes may be tree or LFS metadata.
Acquisition independently checks actual bytes and refuses unqualified pointers.
Actual pull still enforces its original file/snapshot caps and complete required closure.
Offline output is a verified-cached-subset, not a complete remote list, native-protocol
proof, current access/freshness, license acceptance or inference qualification.");
}
