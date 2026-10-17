//! Native GCS JSON object/prefix frontend, sharing the original transfer owner.
use super::{Action, Options, ProgressDisplay, admission, limits, signals};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition;
use zerogravity::acquisition::gcs::{GcsObjectPin, GcsObjectRequest, GcsPrefixRequest};
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::cache::{CacheOptions, CacheStore, CommitOutcome, FileRole};
use zerogravity::{Error, Result};
const ORIGIN: &str = "https://storage.googleapis.com";
#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Anonymous,
    Bearer,
}
enum Selection {
    Object(GcsObjectRequest),
    Prefix(GcsPrefixRequest),
    Offline {
        pin: Box<GcsObjectPin>,
        snapshot: String,
    },
}
struct GcsOptions {
    selection: Selection,
    profile: Profile,
    token_env: Option<String>,
    common: Options,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid pull gcs arguments; use pull gcs --help (tokens are environment-only)".into(),
    )
}
fn parse(args: &[String]) -> Result<GcsOptions> {
    if args.len() > 64 || args.iter().any(|value| value.len() > 4096) {
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
    let mut origin = ORIGIN.to_owned();
    let mut profile = Profile::Anonymous;
    let mut token_env = None;
    let mut bucket = None;
    let mut object = None;
    let mut prefix = None;
    let mut path = None;
    let mut role = None;
    let mut generation = None;
    let mut metageneration = None;
    let mut bytes = None;
    let mut sha256 = None;
    let mut snapshot = None;
    let mut seen = BTreeSet::new();
    let mut rest = args[1..].iter();
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
                .filter(|number| *number > 0)
                .ok_or_else(invalid)
        };
        let count = || usize::try_from(positive()?).map_err(|_| invalid());
        match flag.as_str() {
            "--origin" => origin = value.clone(),
            "--profile" => {
                profile = match value.as_str() {
                    "anonymous" => Profile::Anonymous,
                    "bearer" => Profile::Bearer,
                    _ => return Err(invalid()),
                }
            }
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
            "--bucket" => bucket = Some(value.clone()),
            "--object" => object = Some(value.clone()),
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
            "--generation" => generation = Some(value.clone()),
            "--metageneration" => metageneration = Some(value.clone()),
            "--bytes" => bytes = Some(value.parse::<u64>().map_err(|_| invalid())?),
            "--sha256" => sha256 = Some(value.clone()),
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
    if (profile == Profile::Bearer) != token_env.is_some()
        || (common.offline && common.dry_run)
        || (!common.offline && (snapshot.is_some() || metageneration.is_some() || bytes.is_some()))
    {
        return Err(invalid());
    }
    let bounds = limits(&common)?;
    let bucket = bucket.ok_or_else(invalid)?;
    let selection = if kind == "prefix" {
        if common.offline {
            return Err(Error::Unsupported("offline GCS prefix selection is unavailable; select an explicit object with generation, metageneration, bytes and snapshot digest".into()));
        }
        if object.is_some()
            || path.is_some()
            || role.is_some()
            || generation.is_some()
            || sha256.is_some()
        {
            return Err(invalid());
        }
        Selection::Prefix(GcsPrefixRequest::new(
            &origin,
            &bucket,
            &prefix.ok_or_else(invalid)?,
        )?)
    } else {
        if prefix.is_some() {
            return Err(invalid());
        }
        let request = GcsObjectRequest::new(
            &origin,
            &bucket,
            &object.ok_or_else(invalid)?,
            &path.ok_or_else(invalid)?,
            role.ok_or_else(invalid)?,
            generation.as_deref(),
            sha256.as_deref(),
        )?;
        if common.offline {
            let pin = GcsObjectPin::new(
                &request,
                generation.as_deref().ok_or_else(invalid)?,
                metageneration.as_deref().ok_or_else(invalid)?,
                bytes.ok_or_else(invalid)?,
            )?;
            if pin.bytes() > bounds.cache.max_file_bytes
                || pin.bytes() > bounds.cache.max_snapshot_bytes
            {
                return Err(Error::Limit(
                    "GCS offline pin exceeds caller byte limits".into(),
                ));
            }
            Selection::Offline {
                pin: Box::new(pin),
                snapshot: snapshot.ok_or_else(invalid)?,
            }
        } else {
            Selection::Object(request)
        }
    };
    Ok(GcsOptions {
        selection,
        profile,
        token_env,
        common,
    })
}
fn cache_options(common: &Options) -> Result<CacheOptions> {
    let mut options = CacheOptions::new(common.root.clone(), common.disk);
    options.limits = limits(common)?.cache;
    Ok(options)
}
fn describe_pin(pin: &GcsObjectPin) {
    println!(
        "Generation: {}\nMetageneration: {}\nExact bytes: {}",
        pin.generation(),
        pin.metageneration(),
        pin.bytes()
    );
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
        // No URL input, credentials, client, admission or signal construction.
        let store = CacheStore::open_existing(cache_options(&options.common)?)?;
        let lease = pin.open_cached(&store, snapshot, &bounds, &cancel)?;
        describe_pin(pin);
        std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
        eprintln!(
            "Verified cached GCS pin/data; no present remote membership, freshness or inferred checksum trust."
        );
        return Ok(());
    }
    let admission = admission(options.common.memory)?;
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let mut http = HttpOptions::default();
    if let Some(seconds) = options.common.timeout {
        http.total_timeout = std::time::Duration::from_secs(seconds);
    }
    let origin = match &options.selection {
        Selection::Object(request) => request.origin(),
        Selection::Prefix(request) => request.origin(),
        Selection::Offline { .. } => unreachable!("offline branch returned"),
    };
    let client = match options.profile {
        Profile::Anonymous => HttpsClient::new(origin, http)?,
        Profile::Bearer => {
            let token = std::env::var(options.token_env.as_deref().ok_or_else(invalid)?)
                .map_err(|_| Error::State("GCS bearer environment value unavailable".into()))?;
            HttpsClient::with_bearer(origin, &token, http)?
        }
    };
    let plans = match &options.selection {
        Selection::Object(request) => {
            vec![request.resolve_with_pin(&client, &bounds, &admission, &cancel)?]
        }
        Selection::Prefix(request) => {
            request.resolve_with_pins(&client, &bounds, &admission, &cancel)?
        }
        Selection::Offline { .. } => unreachable!("offline branch returned"),
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
            describe_pin(pin);
            for file in plan.files() {
                println!(
                    "File: {}\nDigest provenance: {:?}",
                    file.path(),
                    file.digest_provenance()
                );
            }
        }
        println!(
            "DRY RUN: object/listing metadata only; no media transfer/cache publication. Prefix objects are independent auxiliary data, not an atomic bucket snapshot or model closure."
        );
        return Ok(());
    }
    let store = CacheStore::open(cache_options(&options.common)?)?;
    for (index, (pin, plan)) in plans.iter().enumerate() {
        eprintln!(
            "GCS object {}/{}; independent publication",
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
                describe_pin(pin);
                std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;
            }
            CommitOutcome::DurabilityUnknown { .. } => return Err(Error::State("complete GCS object data is visible but durability is unknown; earlier independent snapshots may remain".into())),
        }
    }
    eprintln!(
        "Verified independent GCS data only; later-member failure may retain earlier data. No atomic prefix, model closure, signature, licensing or inference eligibility claim."
    );
    Ok(())
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull gcs object --bucket BUCKET --object OBJECT --file LOGICAL \\
        --role ROLE [--generation N] [--sha256 RAW_SHA256] [--dry-run]
    zerogravity pull gcs object [same object flags] --offline --snapshot DIGEST \\
        --generation N --metageneration M --bytes N
    zerogravity pull gcs prefix --bucket BUCKET --prefix PATH/ [--dry-run]

OPTIONS:
    --origin HTTPS_ORIGIN        Native https://storage.googleapis.com only
    --profile anonymous|bearer   Default anonymous; no ambient credential discovery
    --token-env NAME             Required only for bearer; value never argv
    --role ROLE                  weights|config|tokenizer|index|license|auxiliary (object only)
    --sha256 RAW_SHA256          Independently supplied raw SHA256 (object only)
    --dry-run                    Object/listing metadata only; no media/cache publication
    --offline --snapshot DIGEST  Exact local pin; no token/client/admission/signal setup
    --cache ABS_PATH             Existing pull XDG default, or explicit cache root
    --disk-limit-bytes N --memory-limit-bytes N
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N --timeout-seconds N

Offline requires exact positive generation/metageneration and byte count (zero
allowed); only snapshot digests are accepted. Metageneration/bytes are offline-only
flags. Latest online selection resolves once. Receipts print the observed pin.
Without independent SHA256, content is FirstSeen; ETag/MD5/CRC32C are not raw SHA.
Prefix selects independent auxiliary FirstSeen objects with prefix-stripped paths;
offline prefix is unsupported. No mutable references or atomic prefix publication;
earlier complete data may survive later failure. No model/sidecar closure inferred.
Media generation/metageneration query preconditions are mandatory; contradictory
present version headers refuse, absent headers are allowed. Media redirects refuse.
No ADC, metadata-service credentials, SDK, signing, requester-pays, login or token
refresh. Cached data grants no present bucket membership, remote freshness,
licensing decision or inference support."
    );
}
