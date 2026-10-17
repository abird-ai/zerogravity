//! Explicit versioned S3 data acquisition, sharing the original signing/HTTP owner.
use super::{Action, Options, ProgressDisplay, admission, limits, signals};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition;
use zerogravity::acquisition::http::{HttpOptions, HttpsClient, S3Credentials};
use zerogravity::acquisition::s3::{
    S3Addressing, S3ConditionalPin, S3ConditionalRequest, S3ContentSpec, S3Endpoint, S3Listing,
    S3ObjectPin, S3ObjectRequest, S3PrefixRequest,
};
use zerogravity::cache::{CacheOptions, CacheStore, CommitOutcome, FileRole};
use zerogravity::{Error, Result};
#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Anonymous,
    SigV4,
}
enum Selection {
    Conditional(S3ConditionalRequest),
    ConditionalOffline {
        pin: Box<S3ConditionalPin>,
        snapshot: String,
    },
    Content {
        spec: Box<S3ContentSpec>,
        snapshot: Option<String>,
    },
    Object(S3ObjectRequest),
    Prefix(S3PrefixRequest),
    Offline {
        pin: Box<S3ObjectPin>,
        snapshot: String,
    },
}
enum SelectedPin {
    Version(S3ObjectPin),
    Conditional(S3ConditionalPin),
}
struct S3Options {
    selection: Selection,
    profile: Profile,
    credential_origin: String,
    region: String,
    access_env: Option<String>,
    secret_env: Option<String>,
    session_env: Option<String>,
    common: Options,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid pull s3 arguments; use pull s3 --help (credential values are environment-only)"
            .into(),
    )
}
fn env_name(value: &str) -> Result<String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(invalid());
    }
    Ok(value.to_owned())
}
fn parse(args: &[String]) -> Result<S3Options> {
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
    let mut endpoint_profile = "aws";
    let mut endpoint_origin = None;
    let mut region = None;
    let mut addressing = None;
    let mut profile = Profile::Anonymous;
    let mut access_env = None;
    let mut secret_env = None;
    let mut session_env = None;
    let mut bucket = None;
    let mut key = None;
    let mut prefix = None;
    let mut path = None;
    let mut role = None;
    let mut version = None;
    let mut bytes = None;
    let mut sha256 = None;
    let mut snapshot = None;
    let mut listing = None;
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
                .filter(|number| *number > 0)
                .ok_or_else(invalid)
        };
        let count = || usize::try_from(positive()?).map_err(|_| invalid());
        match flag.as_str() {
            "--endpoint-profile" => {
                endpoint_profile = match value.as_str() {
                    "aws" => "aws",
                    "compatible" => "compatible",
                    _ => return Err(invalid()),
                }
            }
            "--endpoint" => endpoint_origin = Some(value.clone()),
            "--region" => region = Some(value.clone()),
            "--addressing" => {
                addressing = Some(match value.as_str() {
                    "path" => S3Addressing::PathStyle,
                    "virtual" => S3Addressing::VirtualHosted,
                    _ => return Err(invalid()),
                })
            }
            "--profile" => {
                profile = match value.as_str() {
                    "anonymous" => Profile::Anonymous,
                    "sigv4" => Profile::SigV4,
                    _ => return Err(invalid()),
                }
            }
            "--access-key-env" => access_env = Some(env_name(value)?),
            "--secret-key-env" => secret_env = Some(env_name(value)?),
            "--session-token-env" => session_env = Some(env_name(value)?),
            "--bucket" => bucket = Some(value.clone()),
            "--key" => key = Some(value.clone()),
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
            "--etag" => etag = Some(value.clone()),
            "--bytes" => bytes = Some(value.parse::<u64>().map_err(|_| invalid())?),
            "--sha256" => sha256 = Some(value.clone()),
            "--listing" => {
                listing = Some(match value.as_str() {
                    "current" => S3Listing::CurrentObjects,
                    "latest-versions" => S3Listing::LatestVersions,
                    _ => return Err(invalid()),
                })
            }
            "--snapshot" => {
                if value.len() != 64
                    || !value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                {
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
            "--max-files" => common.max_files = Some(count()?),
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
    match profile {
        Profile::Anonymous
            if access_env.is_some() || secret_env.is_some() || session_env.is_some() =>
        {
            return Err(invalid());
        }
        Profile::SigV4 if access_env.is_none() || secret_env.is_none() => return Err(invalid()),
        _ => {}
    }
    if (common.offline && common.dry_run)
        || (!common.offline && (snapshot.is_some() || (bytes.is_some() && !content_contract)))
    {
        return Err(invalid());
    }
    if conditional_read && content_contract
        || etag.is_some() && (!conditional_read || !common.offline)
    {
        return Err(invalid());
    }
    let bounds = limits(&common)?;
    let region = region.ok_or_else(invalid)?;
    let addressing = addressing.ok_or_else(invalid)?;
    let endpoint = match (endpoint_profile, endpoint_origin) {
        ("aws", None) => S3Endpoint::aws(&region, addressing)?,
        ("compatible", Some(origin)) => S3Endpoint::compatible(&origin, &region, addressing)?,
        _ => return Err(invalid()),
    };
    let bucket = bucket.ok_or_else(invalid)?;
    let credential_origin = endpoint.credential_origin(&bucket)?;
    let selection = if kind == "prefix" {
        if common.offline {
            return Err(Error::Unsupported("offline S3 prefix selection is unavailable; select an explicit object/version/bytes/snapshot digest".into()));
        }
        if conditional_read
            || content_contract
            || key.is_some()
            || path.is_some()
            || role.is_some()
            || version.is_some()
            || sha256.is_some()
        {
            return Err(invalid());
        }
        Selection::Prefix(S3PrefixRequest::new(
            &endpoint,
            &bucket,
            &prefix.ok_or_else(invalid)?,
            listing.unwrap_or(S3Listing::CurrentObjects),
        )?)
    } else {
        if prefix.is_some() || listing.is_some() {
            return Err(invalid());
        }
        if conditional_read {
            if version.is_some() || sha256.is_some() {
                return Err(invalid());
            }
            let request = S3ConditionalRequest::new(
                &endpoint,
                &bucket,
                &key.ok_or_else(invalid)?,
                &path.ok_or_else(invalid)?,
                role.ok_or_else(invalid)?,
            )?;
            if common.offline {
                let pin = S3ConditionalPin::new(
                    &request,
                    bytes.ok_or_else(invalid)?,
                    etag.as_deref().ok_or_else(invalid)?,
                )?;
                if pin.bytes() > bounds.cache.max_file_bytes
                    || pin.bytes() > bounds.cache.max_snapshot_bytes
                {
                    return Err(Error::Limit(
                        "S3 conditional pin exceeds byte bounds".into(),
                    ));
                }
                Selection::ConditionalOffline {
                    pin: Box::new(pin),
                    snapshot: snapshot.ok_or_else(invalid)?,
                }
            } else {
                Selection::Conditional(request)
            }
        } else if content_contract {
            if version.is_some() {
                return Err(invalid());
            }
            let spec = S3ContentSpec::new(
                &endpoint,
                &bucket,
                &key.ok_or_else(invalid)?,
                &path.ok_or_else(invalid)?,
                role.ok_or_else(invalid)?,
                bytes.ok_or_else(invalid)?,
                sha256.as_deref().ok_or_else(invalid)?,
            )?;
            if spec.bytes() > bounds.cache.max_file_bytes
                || spec.bytes() > bounds.cache.max_snapshot_bytes
            {
                return Err(Error::Limit(
                    "S3 content contract exceeds caller byte limits".into(),
                ));
            }
            if common.offline && snapshot.is_none() {
                return Err(invalid());
            }
            Selection::Content {
                spec: Box::new(spec),
                snapshot,
            }
        } else {
            let request = S3ObjectRequest::new(
                &endpoint,
                &bucket,
                &key.ok_or_else(invalid)?,
                &path.ok_or_else(invalid)?,
                role.ok_or_else(invalid)?,
                version.as_deref(),
                sha256.as_deref(),
            )?;
            if common.offline {
                let pin = S3ObjectPin::new(
                    &request,
                    version.as_deref().ok_or_else(invalid)?,
                    bytes.ok_or_else(invalid)?,
                )?;
                if pin.bytes() > bounds.cache.max_file_bytes
                    || pin.bytes() > bounds.cache.max_snapshot_bytes
                {
                    return Err(Error::Limit(
                        "S3 offline pin exceeds caller byte limits".into(),
                    ));
                }
                Selection::Offline {
                    pin: Box::new(pin),
                    snapshot: snapshot.ok_or_else(invalid)?,
                }
            } else {
                Selection::Object(request)
            }
        }
    };
    Ok(S3Options {
        selection,
        profile,
        credential_origin,
        region,
        access_env,
        secret_env,
        session_env,
        common,
    })
}
fn cache_options(common: &Options) -> Result<CacheOptions> {
    let mut result = CacheOptions::new(common.root.clone(), common.disk);
    result.limits = limits(common)?.cache;
    Ok(result)
}
fn describe_pin(pin: &S3ObjectPin) {
    // JSON-string escaping keeps arbitrary validated version labels unambiguous.
    let version = serde_json::to_string(pin.version()).expect("string serialization");
    println!(
        "Version ID (JSON string): {version}\nExact bytes: {}",
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
            SelectedPin::Version(pin) => describe_pin(pin),
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
fn credential(name: Option<&str>) -> Result<String> {
    std::env::var(name.ok_or_else(invalid)?)
        .map_err(|_| Error::State("explicit S3 credential environment value unavailable".into()))
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
        // Terminal before credential reads, client/admission/signal construction.
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = pin.open_cached(&store, snapshot, &bounds, &cancel)?;
        describe_pin(pin);
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached S3 version/data; no remote membership, freshness, SDK credential discovery or inferred checksum trust."
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
            "Verified cached S3 caller content contract; no remote version or freshness assertion."
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
            spec.request().path()
        );
        describe_content(spec.bytes(), spec.sha256());
        println!(
            "DRY RUN: local caller content contract only; no metadata/payload/cache writes or credential/client/signal setup."
        );
        return Ok(());
    }
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let mut http = HttpOptions::default();
    if let Some(seconds) = options.common.timeout {
        http.total_timeout = std::time::Duration::from_secs(seconds);
    }
    let client = match options.profile {
        Profile::Anonymous => HttpsClient::new(&options.credential_origin, http)?,
        Profile::SigV4 => {
            let access = credential(options.access_env.as_deref())?;
            let secret = credential(options.secret_env.as_deref())?;
            let session = options
                .session_env
                .as_deref()
                .map(|name| credential(Some(name)))
                .transpose()?;
            let credentials = S3Credentials::new(&access, &secret, session.as_deref())?;
            HttpsClient::with_s3_credentials(
                &options.credential_origin,
                &options.region,
                credentials,
                http,
            )?
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
            .resolve(&client, &bounds, &admission, &cancel)?
            .into_iter()
            .map(|(pin, plan)| (Some(SelectedPin::Version(pin)), plan))
            .collect(),
        Selection::Offline { .. } | Selection::ConditionalOffline { .. } => {
            unreachable!("offline branch returned")
        }
    };
    if options.common.dry_run {
        for (index, (pin, plan)) in plans.iter().enumerate() {
            println!(
                "Object: {}/{}\nProvider: {}\nPinned contract: {}",
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
            "DRY RUN: listing/HEAD metadata only; no payload transfer/cache publication. Prefix objects are independent auxiliary data, not an atomic bucket snapshot or model closure."
        );
        return Ok(());
    }
    let store = CacheStore::open(cache_options(&options.common)?)?;
    for (index, (pin, plan)) in plans.iter().enumerate() {
        eprintln!(
            "S3 object {}/{}; independent publication",
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
        match receipt.outcome() {
            CommitOutcome::Durable(snapshot) => {
                println!("Snapshot: {}", snapshot.digest());
                describe_selected(pin.as_ref(), plan);
                std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;
            }
            CommitOutcome::DurabilityUnknown { .. } => return Err(Error::State("complete S3 object data is visible but durability is unknown; earlier independent snapshots may remain".into())),
        }
    }
    eprintln!(
        "Verified independent S3 data only; later failure may retain earlier objects. No atomic prefix, model closure, signature, licensing or inference eligibility claim."
    );
    Ok(())
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull s3 object --region REGION --addressing path|virtual \\
        --bucket BUCKET --key KEY --file LOGICAL --role ROLE [--version-id ID] \\
        [--sha256 RAW_SHA256] [--dry-run]
    zerogravity pull s3 object [same request flags] --offline --snapshot DIGEST \\
        --version-id ID --bytes N
    zerogravity pull s3 prefix --region REGION --addressing path|virtual \\
        --bucket BUCKET --prefix PATH/ [--listing current|latest-versions] [--dry-run]

OPTIONS:
    --conditional-read            Object-only HEAD ETag/size observation; FirstSeen
    --etag QUOTED_ETAG             Required offline conditional observation only
    --endpoint-profile aws|compatible  Default aws; explicit region/addressing required
    --endpoint HTTPS_ORIGIN      Required only for compatible; AWS origin derived from region
    --profile anonymous|sigv4     Default anonymous; no ambient AWS credential chain
    --access-key-env NAME --secret-key-env NAME  Required only for sigv4
    --session-token-env NAME      Optional explicit session token for sigv4
    --role ROLE                   weights|config|tokenizer|index|license|auxiliary (object only)
    --version-id ID               Online optional; offline exact non-null immutable version
    --sha256 RAW_SHA256           Independent caller SHA256, object only
    --offline --snapshot DIGEST   Full verified cache pin before credentials/client/admission
    --bytes N                     Offline version/conditional pin or content contract; zero allowed
    --content-contract            Object-only caller SHA256 +exact bytes; no version ID
    --cache ABS_PATH              Existing pull XDG default, or explicit cache root
    --disk-limit-bytes N --memory-limit-bytes N
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N --timeout-seconds N

Versioned dry-run uses listing/HEAD metadata only; no payload transfer or cache publication.
Explicit --content-contract requires --bytes and --sha256 and forbids --version-id.
It supports unversioned/current locators, uses no HEAD, and binds source to caller
raw bytes rather than a remote version. Content dry-run is local before credential
reads/client/signals; offline requires the same contract and --snapshot DIGEST.
No raw SHA means no content mode; there is no fallback from versioned selection.
Explicit --conditional-read observes HEAD exact bytes/strong ETag, then requires
If-Match and matching ETag on every GET/range/retry. No caller SHA or remote version
is inferred: full raw hash is FirstSeen. Online --bytes/--etag and SHA/revision/
content-mode conflicts refuse. Offline adds --bytes, --etag and cached digest
before credentials/client/admission/signals. Its dry-run is HEAD metadata only.

Version IDs are printed as JSON strings for unambiguous offline reconstruction.
Versioned mode HEAD selection resolves once to a non-null version; unversioned/null
objects refuse in that mode. Anonymous sends no credentials. SigV4 reads only the selected environment
names and signs through the original owner at the actual bucket endpoint. No
credential values in argv, SDK/config/metadata-service discovery, role assumption,
region discovery, redirects, writes, requester-pays or archive restore.
ETag/composite provider checksums never authenticate raw SHA; without independent
caller SHA, acquired bytes are FirstSeen. A compatible endpoint is an explicit
protocol profile, not blanket vendor parity. Standard AWS is commercial regional
S3 only; dotted virtual-host buckets and other unqualified endpoints refuse.
Prefix current lists current objects then pins HEAD; latest-versions selects
latest nondeleted versions. Both yield independent Auxiliary/FirstSeen snapshots;
no offline prefix, mutable references, atomic bucket snapshot or model closure.
Earlier data may remain if later objects fail. Only exact snapshot digests are
accepted offline; cached bytes prove no present remote freshness or inference
eligibility."
    );
}
