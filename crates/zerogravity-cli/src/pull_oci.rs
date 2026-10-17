//! Explicit raw OCI GGUF-layer acquisition; no container runtime or extraction.
use super::{Action, Options, ProgressDisplay, admission, limits, signals};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition;
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::acquisition::oci::{
    GgufLayer, OciProfile, OciReference, OciRequest, docker_hub_client,
};
use zerogravity::cache::{CacheOptions, CacheStore, CommitOutcome};
use zerogravity::{Error, Result};

struct OciOptions {
    request: OciRequest,
    token_env: Option<String>,
    snapshot: Option<String>,
    common: Options,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid pull oci arguments; use pull oci --help (tokens are environment-only)".into(),
    )
}
fn parse(args: &[String]) -> Result<OciOptions> {
    if args.len() > 64 || args.iter().any(|value| value.len() > 4096) {
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
    let mut profile = OciProfile::Anonymous;
    let mut repository = None;
    let mut reference = None;
    let mut file = None;
    let mut layer = GgufLayer::Unique;
    let mut token_env = None;
    let mut snapshot = None;
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
                .filter(|value| *value > 0)
                .ok_or_else(invalid)
        };
        let count = || usize::try_from(positive()?).map_err(|_| invalid());
        match flag.as_str() {
            "--origin" => origin = Some(value.clone()),
            "--profile" => {
                profile = match value.as_str() {
                    "anonymous" => OciProfile::Anonymous,
                    "bearer" => OciProfile::Bearer,
                    "dockerhub" => OciProfile::DockerHubAnonymous,
                    _ => return Err(invalid()),
                }
            }
            "--repository" => repository = Some(value.clone()),
            "--tag" | "--manifest-sha256" => {
                if reference.is_some() {
                    return Err(invalid());
                }
                reference = Some(if flag == "--tag" {
                    OciReference::Tag(value.clone())
                } else {
                    OciReference::ManifestSha256(value.clone())
                });
            }
            "--file" => {
                if !value.ends_with(".gguf") {
                    return Err(invalid());
                }
                file = Some(value.clone());
            }
            "--layer-sha256" => layer = GgufLayer::Sha256(value.clone()),
            "--token-env" => {
                if value.is_empty()
                    || value.len() > 128
                    || !value
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                {
                    return Err(invalid());
                }
                token_env = Some(value.clone());
            }
            "--snapshot" => {
                if value.is_empty() || value.chars().any(char::is_control) {
                    return Err(invalid());
                }
                snapshot = Some(value.clone());
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
            "--max-metadata-bytes" => common.metadata = Some(count()?),
            "--max-pages" => common.pages = Some(count()?),
            "--timeout-seconds" => {
                let seconds = positive()?;
                if seconds > 86400 {
                    return Err(invalid());
                }
                common.timeout = Some(seconds);
            }
            _ => return Err(invalid()),
        }
    }
    if (common.offline && (common.dry_run || snapshot.is_none()))
        || (!common.offline && snapshot.is_some())
        || (profile == OciProfile::Bearer) != token_env.is_some()
    {
        return Err(invalid());
    }
    let origin = origin
        .or_else(|| {
            (profile == OciProfile::DockerHubAnonymous)
                .then(|| "https://registry-1.docker.io".to_owned())
        })
        .ok_or_else(invalid)?;
    let request = OciRequest::new(
        &origin,
        profile,
        &repository.ok_or_else(invalid)?,
        reference.ok_or_else(invalid)?,
        &file.ok_or_else(invalid)?,
        layer,
    )?;
    limits(&common)?;
    Ok(OciOptions {
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
    // Terminal offline path before credentials, broker/client, admission or signals.
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
            "Verified cached OCI request/data; no independent remote descriptor membership, tag freshness, signature or inference claim."
        );
        return Ok(());
    }
    let admission = admission(options.common.memory)?;
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let mut http = HttpOptions::default();
    if let Some(seconds) = options.common.timeout {
        http.total_timeout = std::time::Duration::from_secs(seconds);
    }
    let client = match options.request.profile() {
        OciProfile::Anonymous => HttpsClient::new(options.request.origin(), http)?,
        OciProfile::Bearer => {
            let token = std::env::var(options.token_env.as_deref().ok_or_else(invalid)?)
                .map_err(|_| Error::State("OCI bearer environment value unavailable".into()))?;
            HttpsClient::with_bearer(options.request.origin(), &token, http)?
        }
        OciProfile::DockerHubAnonymous => {
            let broker = HttpsClient::new("https://auth.docker.io", http.clone())?;
            docker_hub_client(
                &options.request,
                &broker,
                http,
                &bounds,
                &admission,
                &cancel,
            )?
        }
    };
    let plan = options
        .request
        .resolve(&client, &bounds, &admission, &cancel)?;
    if options.common.dry_run {
        println!(
            "Provider: {}\nRequest identity: {}\nObserved manifest raw SHA256: {}\nManifest selection: {}",
            plan.source().provider(),
            plan.source().repository(),
            plan.source().revision(),
            if matches!(options.request.reference(), OciReference::ManifestSha256(_)) {
                "verified against caller-supplied manifest digest"
            } else {
                "tag resolved once; no future freshness guarantee"
            }
        );
        for file in plan.files() {
            println!(
                "File: {}\nExact bytes: {}\nLayer SHA256: {} (from hashed manifest descriptor)",
                file.path(),
                super::selected_bytes(file.bytes(), file.maximum_bytes()),
                file.sha256().unwrap_or("unavailable")
            );
        }
        println!(
            "DRY RUN: manifest/broker metadata only; no layer payload transfer, cache publication or inference qualification."
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
            println!(
                "Snapshot: {}\nVerified raw GGUF layer data only; no container execution, repository closure or inference eligibility implied.",
                snapshot.digest()
            );
            std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;
            Ok(())
        }
        CommitOutcome::DurabilityUnknown { .. } => Err(Error::State(
            "complete OCI data is visible but durability is unknown; inspect before retrying"
                .into(),
        )),
    }
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull oci --origin https://registry --repository REPO \\
        (--tag TAG | --manifest-sha256 RAW_SHA256) --file LOGICAL.gguf
    zerogravity pull oci --profile dockerhub --repository OWNER/REPO \\
        --tag TAG --file LOGICAL.gguf [--dry-run]
    zerogravity pull oci [same request flags] --offline --snapshot DIGEST_OR_REF

OPTIONS:
    --profile anonymous|bearer|dockerhub  Default anonymous; no ambient credentials
    --origin HTTPS_ORIGIN        Explicit bare registry origin; Docker Hub fixed default
    --layer-sha256 RAW_SHA256     Select one recognized raw GGUF layer (default unique)
    --token-env NAME             Required only for bearer; token value never argv
    --dry-run                    Manifest/broker metadata only; no payload/cache publication
    --offline --snapshot ID      Full verified cached data before credentials/client/broker
    --cache ABS_PATH             Existing pull XDG default, or explicit cache root
    --disk-limit-bytes N         Cooperative disk policy (default16GiB)
    --memory-limit-bytes N       Process transfer-staging cap (default128MiB, not all RAM)
    --max-file-bytes N --max-snapshot-bytes N
    --max-metadata-bytes N --max-pages N --timeout-seconds N

Manifest and layer digests use raw lowercase SHA256 without a sha256: flag prefix.
A tag is resolved once; cached reuse proves the declared request/data, not current
remote membership or freshness. No mutable reference is published. Docker Hub
uses only its fixed anonymous public-pull broker; bearer tokens are scoped to the
explicit registry by the original HTTP owner. No challenge-realm discovery,
login, token refresh, OCI signature verification or arbitrary registry parity.
Only recognized raw GGUF layers are selectable. OCI indexes/platform lists,
compressed/config/index layer selections and ambiguous GGUF selections refuse;
nothing is executed or extracted. Data alone grants no model
closure, licensing decision or inference support."
    );
}
