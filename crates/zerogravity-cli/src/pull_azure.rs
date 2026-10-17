//! Explicit native Azure BlockBlob frontend over the original transfer owner.
use super::{Action, Options, ProgressDisplay, admission, limits, signals};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition;
use zerogravity::acquisition::azure::{
    AzureConditionalPin, AzureConditionalRequest, AzureContentSpec, AzureObjectPin,
    AzureObjectRequest, AzurePrefixRequest, AzureRevision,
};
use zerogravity::acquisition::http::{AzureSharedKey, HttpOptions, HttpsClient};
use zerogravity::cache::{CacheOptions, CacheStore, CommitOutcome, FileRole};
use zerogravity::{Error, Result};
enum Selection {
    Conditional(AzureConditionalRequest),
    ConditionalOffline {
        pin: Box<AzureConditionalPin>,
        snapshot: String,
    },
    Content {
        spec: Box<AzureContentSpec>,
        snapshot: Option<String>,
    },
    Object(AzureObjectRequest),
    Prefix(AzurePrefixRequest),
    Offline {
        pin: Box<AzureObjectPin>,
        snapshot: String,
    },
}
enum AuthInput {
    Anonymous,
    Bearer(String),
    SharedKey(String),
    SasFile(String),
}
enum SelectedPin {
    Version(AzureObjectPin),
    Conditional(AzureConditionalPin),
}
struct AzureOptions {
    selection: Selection,
    auth: AuthInput,
    common: Options,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid pull azure arguments; use pull azure --help (credential values are never argv)"
            .into(),
    )
}
fn env_name(value: &str) -> Result<String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(invalid());
    }
    Ok(value.to_owned())
}
fn parse(args: &[String]) -> Result<AzureOptions> {
    if args.len() > 80 || args.iter().any(|value| value.len() > 4096) {
        return Err(invalid());
    }
    let kind = args.first().ok_or_else(invalid)?;
    if !matches!(kind.as_str(), "object" | "prefix") {
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
    let mut container = None;
    let mut blob = None;
    let mut prefix = None;
    let mut path = None;
    let mut role = None;
    let mut version = None;
    let mut snapshot = None;
    let mut cached = None;
    let mut bytes = None;
    let mut sha = None;
    let mut auth = None;
    let mut content_contract = false;
    let mut conditional_read = false;
    let mut etag = None;
    let mut seen = BTreeSet::new();
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        if !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--conditional-read" => {
                conditional_read = true;
                continue;
            }
            "--content-contract" => {
                content_contract = true;
                continue;
            }
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
        let count = || usize::try_from(positive()?).map_err(|_| invalid());
        match flag.as_str() {
            "--origin" => origin = Some(value.clone()),
            "--container" => container = Some(value.clone()),
            "--blob" => blob = Some(value.clone()),
            "--prefix" => prefix = Some(value.clone()),
            "--file" => path = Some(value.clone()),
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
            "--version-id" => version = Some(value.clone()),
            "--snapshot" => snapshot = Some(value.clone()),
            "--sha256" => sha = Some(value.clone()),
            "--etag" => etag = Some(value.clone()),
            "--bytes" => bytes = Some(value.parse::<u64>().map_err(|_| invalid())?),
            "--cached-snapshot" => {
                if value.len() != 64
                    || !value
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(invalid());
                }
                cached = Some(value.clone());
            }
            "--token-env" | "--shared-key-env" | "--sas-file" => {
                if auth.is_some() {
                    return Err(invalid());
                }
                auth = Some(match flag.as_str() {
                    "--token-env" => AuthInput::Bearer(env_name(value)?),
                    "--shared-key-env" => AuthInput::SharedKey(env_name(value)?),
                    _ => {
                        if value != "-"
                            && (!Path::new(value).is_absolute()
                                || value.chars().any(char::is_control))
                        {
                            return Err(invalid());
                        }
                        AuthInput::SasFile(value.clone())
                    }
                });
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
    if common.offline && common.dry_run
        || !common.offline && (cached.is_some() || (bytes.is_some() && !content_contract))
    {
        return Err(invalid());
    }
    if conditional_read && content_contract
        || etag.is_some() && (!conditional_read || !common.offline)
    {
        return Err(invalid());
    }
    let bounds = limits(&common)?;
    let origin = origin.ok_or_else(invalid)?;
    let container = container.ok_or_else(invalid)?;
    let selection = if kind == "prefix" {
        if common.offline {
            return Err(Error::Unsupported("offline Azure prefix is unavailable; select one explicit version/snapshot and cached snapshot digest".into()));
        }
        if conditional_read
            || content_contract
            || blob.is_some()
            || path.is_some()
            || role.is_some()
            || version.is_some()
            || snapshot.is_some()
            || sha.is_some()
        {
            return Err(invalid());
        }
        Selection::Prefix(AzurePrefixRequest::new(
            &origin,
            &container,
            &prefix.ok_or_else(invalid)?,
        )?)
    } else {
        if prefix.is_some() {
            return Err(invalid());
        }
        if conditional_read {
            if version.is_some() || snapshot.is_some() || sha.is_some() {
                return Err(invalid());
            }
            let request = AzureConditionalRequest::new(
                &origin,
                &container,
                &blob.ok_or_else(invalid)?,
                &path.ok_or_else(invalid)?,
                role.ok_or_else(invalid)?,
            )?;
            if common.offline {
                let pin = AzureConditionalPin::new(
                    &request,
                    bytes.ok_or_else(invalid)?,
                    etag.as_deref().ok_or_else(invalid)?,
                )?;
                if pin.bytes() > bounds.cache.max_file_bytes
                    || pin.bytes() > bounds.cache.max_snapshot_bytes
                {
                    return Err(Error::Limit(
                        "Azure conditional pin exceeds byte bounds".into(),
                    ));
                }
                Selection::ConditionalOffline {
                    pin: Box::new(pin),
                    snapshot: cached.ok_or_else(invalid)?,
                }
            } else {
                Selection::Conditional(request)
            }
        } else if content_contract {
            if version.is_some() || snapshot.is_some() {
                return Err(invalid());
            }
            let spec = AzureContentSpec::new(
                &origin,
                &container,
                &blob.ok_or_else(invalid)?,
                &path.ok_or_else(invalid)?,
                role.ok_or_else(invalid)?,
                bytes.ok_or_else(invalid)?,
                sha.as_deref().ok_or_else(invalid)?,
            )?;
            if spec.bytes() > bounds.cache.max_file_bytes
                || spec.bytes() > bounds.cache.max_snapshot_bytes
            {
                return Err(Error::Limit(
                    "Azure content contract exceeds caller byte bounds".into(),
                ));
            }
            if common.offline && cached.is_none() {
                return Err(invalid());
            }
            Selection::Content {
                spec: Box::new(spec),
                snapshot: cached,
            }
        } else {
            let revision = match (version, snapshot) {
                (Some(v), None) => AzureRevision::version(&v)?,
                (None, Some(s)) => AzureRevision::snapshot(&s)?,
                _ => return Err(invalid()),
            };
            let request = AzureObjectRequest::new(
                &origin,
                &container,
                &blob.ok_or_else(invalid)?,
                &path.ok_or_else(invalid)?,
                role.ok_or_else(invalid)?,
                revision,
                sha.as_deref(),
            )?;
            if common.offline {
                let pin = AzureObjectPin::new(&request, bytes.ok_or_else(invalid)?)?;
                if pin.bytes() > bounds.cache.max_file_bytes
                    || pin.bytes() > bounds.cache.max_snapshot_bytes
                {
                    return Err(Error::Limit("Azure offline pin exceeds byte bounds".into()));
                }
                Selection::Offline {
                    pin: Box::new(pin),
                    snapshot: cached.ok_or_else(invalid)?,
                }
            } else {
                Selection::Object(request)
            }
        }
    };
    Ok(AzureOptions {
        selection,
        auth: auth.unwrap_or(AuthInput::Anonymous),
        common,
    })
}
fn cache_options(common: &Options) -> Result<CacheOptions> {
    let mut result = CacheOptions::new(common.root.clone(), common.disk);
    result.limits = limits(common)?.cache;
    Ok(result)
}
fn describe(pin: &AzureObjectPin) {
    println!(
        "{}: {:?}\nExact bytes: {}",
        if pin.revision().is_snapshot() {
            "Snapshot ID"
        } else {
            "Version ID"
        },
        pin.revision().value(),
        pin.bytes()
    );
}
fn describe_conditional(bytes: u64, etag: &str) {
    let etag = serde_json::to_string(etag).expect("string serialization");
    println!(
        "Conditional observation: FirstSeen\nETag (JSON string): {etag}\nExact bytes: {bytes}"
    );
}
fn describe_content(bytes: u64, sha256: &str) {
    println!(
        "Content contract: caller-authorized SHA256\nExact bytes: {bytes}\nExpected SHA256: {sha256}"
    );
}
fn describe_selected(pin: Option<&SelectedPin>, plan: &acquisition::ResolvedPullPlan) {
    if let Some(pin) = pin {
        match pin {
            SelectedPin::Version(pin) => describe(pin),
            SelectedPin::Conditional(pin) => describe_conditional(pin.bytes(), pin.etag()),
        }
    } else {
        let file = &plan.files()[0];
        describe_content(
            file.bytes().expect("exact content length"),
            file.sha256().expect("trusted content SHA256"),
        );
    }
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        help();
        return Ok(());
    }
    let options = parse(args)?;
    let bounds = limits(&options.common)?;
    let cancel = Arc::new(AtomicBool::new(false));
    if let Selection::Offline { pin, snapshot } = &options.selection {
        // This branch precedes every env/file/stdin credential read, client,
        // admission and signal registration, including hostile FIFO/SAS input.
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = pin.open_cached(&store, snapshot, &bounds, &cancel)?;
        describe(pin);
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached Azure pin/data only; no present remote membership, freshness or inferred checksum trust."
        );
        return Ok(());
    }
    if let Selection::Content {
        spec,
        snapshot: Some(snapshot),
    } = &options.selection
    {
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = spec.open_cached(&store, snapshot, &bounds, &cancel)?;
        describe_content(spec.bytes(), spec.sha256());
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached Azure caller content contract; no remote version or freshness assertion."
        );
        return Ok(());
    }
    if let Selection::ConditionalOffline { pin, snapshot } = &options.selection {
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = pin.open_cached(&store, snapshot, &bounds, &cancel)?;
        describe_conditional(pin.bytes(), pin.etag());
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached FirstSeen conditional observation; no present remote version/freshness or caller-authorized raw SHA assertion."
        );
        return Ok(());
    }
    let admission = admission(options.common.memory)?;
    if options.common.dry_run
        && let Selection::Content { spec, .. } = &options.selection
    {
        let plan = spec.prepare(&bounds, &admission, &cancel)?;
        println!(
            "Provider: {}\nPinned contract: {}\nFile: {}",
            plan.source().provider(),
            plan.source().revision(),
            spec.path()
        );
        describe_content(spec.bytes(), spec.sha256());
        println!(
            "DRY RUN: local caller content contract only; no metadata/payload/cache writes or credential/client/signal setup."
        );
        return Ok(());
    }
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let mut http = HttpOptions::default();
    if let Some(n) = options.common.timeout {
        http.total_timeout = std::time::Duration::from_secs(n);
    }
    let origin = match &options.selection {
        Selection::Conditional(request) => request.origin(),
        Selection::Content { spec, .. } => spec.origin(),
        Selection::Object(r) => r.origin(),
        Selection::Prefix(r) => r.origin(),
        Selection::Offline { .. } | Selection::ConditionalOffline { .. } => {
            unreachable!("offline returned")
        }
    };
    let client = match &options.auth {
        AuthInput::Anonymous => HttpsClient::new(origin, http)?,
        AuthInput::Bearer(name) => {
            let token = std::env::var(name)
                .map_err(|_| Error::State("Azure bearer environment value unavailable".into()))?;
            HttpsClient::with_bearer(origin, &token, http)?
        }
        AuthInput::SharedKey(name) => {
            let account = origin
                .strip_prefix("https://")
                .and_then(|v| v.strip_suffix(".blob.core.windows.net"))
                .ok_or_else(invalid)?;
            let key = std::env::var(name).map_err(|_| {
                Error::State("Azure account key environment value unavailable".into())
            })?;
            HttpsClient::with_azure_shared_key(origin, AzureSharedKey::new(account, &key)?, http)?
        }
        AuthInput::SasFile(input) => {
            let query = super::object::read_secret_input(input, 8192)?;
            HttpsClient::with_azure_sas(origin, &query, http)?
        }
    };
    let plans = match &options.selection {
        Selection::Conditional(request) => {
            let (pin, plan) = request.resolve_with_pin(&client, &bounds, &admission, &cancel)?;
            vec![(Some(SelectedPin::Conditional(pin)), plan)]
        }
        Selection::Content { spec, .. } => {
            vec![(None, spec.prepare(&bounds, &admission, &cancel)?)]
        }
        Selection::Object(request) => {
            let (pin, plan) = request.resolve_with_pin(&client, &bounds, &admission, &cancel)?;
            vec![(Some(SelectedPin::Version(pin)), plan)]
        }
        Selection::Prefix(request) => request
            .resolve_with_pins(&client, &bounds, &admission, &cancel)?
            .into_iter()
            .map(|(pin, plan)| (Some(SelectedPin::Version(pin)), plan))
            .collect(),
        Selection::Offline { .. } | Selection::ConditionalOffline { .. } => {
            unreachable!("offline returned")
        }
    };
    if options.common.dry_run {
        for (index, (pin, plan)) in plans.iter().enumerate() {
            println!(
                "Object: {}/{}\nProvider: {}\nPinned source: {}",
                index + 1,
                plans.len(),
                plan.source().provider(),
                plan.source().revision()
            );
            describe_selected(pin.as_ref(), plan);
            for file in plan.files() {
                println!(
                    "File: {}\nDigest provenance: {:?}",
                    file.path(),
                    file.digest_provenance()
                );
            }
        }
        println!(
            "DRY RUN: properties/listing metadata only; no media/cache publication. Prefix members are independent auxiliary data, not an atomic container snapshot or model closure."
        );
        return Ok(());
    }
    let store = CacheStore::open(cache_options(&options.common)?)?;
    for (index, (pin, plan)) in plans.iter().enumerate() {
        eprintln!(
            "Azure object {}/{}; independent publication",
            index + 1,
            plans.len()
        );
        let mut display = ProgressDisplay::default();
        let receipt = acquisition::execute_with_progress(
            &store,
            plan,
            &client,
            &admission,
            &cancel,
            None,
            &mut |event| {
                let _ = display.report(event, &mut std::io::stderr());
            },
        )?;
        match receipt.outcome(){CommitOutcome::Durable(s)=>{println!("Snapshot: {}",s.digest());describe_selected(pin.as_ref(), plan);std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;},CommitOutcome::DurabilityUnknown{..}=>return Err(Error::State("complete Azure data visible but durability unknown; earlier independent data may remain".into()))}
    }
    eprintln!(
        "Verified independent Azure data only; earlier data may survive later failure. No atomic prefix, model closure, license or inference eligibility claim."
    );
    Ok(())
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull azure object --origin https://ACCOUNT.blob.core.windows.net \\
        --container CONTAINER --blob BLOB --file LOGICAL --role ROLE \\
        (--version-id ID | --snapshot ID) [--sha256 RAW_SHA256] [--dry-run]
    zerogravity pull azure object [same object flags] --offline \\
        --cached-snapshot DIGEST --bytes N
    zerogravity pull azure prefix --origin HTTPS_ORIGIN --container CONTAINER \\
        --prefix PATH/ [--dry-run]

OPTIONS:
    --conditional-read            Object-only HEAD ETag/size observation; FirstSeen
    --etag QUOTED_ETAG             Required offline conditional observation only
    --bytes N                     Offline revision/conditional pin or content contract; zero allowed
    --token-env NAME | --shared-key-env NAME | --sas-file ABS_PATH|-
                                  Optional mutually exclusive explicit credentials
    --content-contract --bytes N --sha256 SHA
                                  Explicit object content; no version/snapshot selector
    --cache ABS_PATH               Pull XDG default, or explicit cache root
    --disk-limit-bytes N --memory-limit-bytes N
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N --timeout-seconds N

Native public-cloud BlockBlob versioned mode requires exactly one opaque
version/snapshot; no mutable latest lookup or automatic fallback. Explicit
--content-contract instead requires exact --bytes and caller --sha256, forbids
both revision selectors, and binds content rather than claiming a remote version.
Its dry-run is local before credentials/client/signals, with no HEAD or payload.
Offline requires exact
bytes (zero allowed) and a cached snapshot DIGEST, before ALL env/file/stdin,
client/admission/signal work; no ETag input. Online cached-snapshot and bytes without content mode refuse.
No mutable refs, unknown-size caps or offline prefix.
Explicit --conditional-read observes HEAD exact bytes/strong ETag, then requires
If-Match and matching ETag on every GET/range/retry. No caller SHA or remote version
is inferred: full raw hash is FirstSeen. Online --bytes/--etag and SHA/revision/
content-mode conflicts refuse. Offline adds --bytes, --etag and cached digest
before credentials/client/admission/signals. Its dry-run is HEAD metadata only.
 Roles are weights, config,
tokenizer, index, license, auxiliary. Caller SHA256 alone supplies TrustedSha256;
otherwise full raw hash is FirstSeen. ETag/MD5/CRC64 are not raw SHA authority.
Prefix selects independent current-version Auxiliary/FirstSeen data; no atomic
container snapshot or inferred model/sidecar closure. Earlier data may survive
later failure. --dry-run uses properties/listing metadata only; no payload/cache.
Anonymous is default; no ambient/default/managed credentials, SDK, login or refresh.
Account key is environment-only and account derives from native origin. Service
SAS is an explicitly chosen bounded8KiB nofollow regular file or stdin (which may
block), with one optional final LF/CRLF; query never argv/output/plan. Only ad-hoc
read b for content, bv/bs for revisions or container c (read/list) HTTPS serviceSAS sv2019-12-12..2023-11-03 is
qualified. Other SAS forms/response overrides refuse. Original escaped query is
preserved with unsigned api-version=2023-11-03. Native requests never redirect.
Optional present version echoes must agree in versioned mode; complete identity-encoded
BlockBlob responses are mandatory. Cached bytes grant no present remote access,
freshness, license decision or inference support."
    );
}
