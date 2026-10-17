//! Explicit offline cache preparation, then the ordinary CPU execution loop.
use super::{Command, cpu_options, infer_prepared, parse};
use std::path::{Path, PathBuf};
use zerogravity::cache::{CacheOptions, CacheStore, SnapshotLease, TransferAdmission};
use zerogravity::prepare::PreparedQwen;
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity::{Error, Result};

fn invalid() -> Error {
    Error::Malformed("invalid run-cache arguments; use run-cache --help".into())
}
fn snapshot(store: &CacheStore, selector: &str) -> Result<SnapshotLease> {
    if selector.len() == 64
        && selector
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        store.open_snapshot(selector)
    } else {
        store.resolve_ref(selector)
    }
}

pub(super) fn run(args: Vec<String>) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        println!(
            "USAGE:
    zerogravity run-cache SNAPSHOT_OR_REF --model-file model.gguf [CPU run options]

OPTIONS:
    --cache ABS_PATH           Override XDG cache root (abird/zerogravity/models)
    --config-snapshot REF      Separate explicit config snapshot (default model snapshot)
    --tokenizer-snapshot REF   Separate explicit tokenizer snapshot (default model snapshot)
    --config-file PATH        Config path within snapshot (default colocated config.json)
    --tokenizer-file PATH     Tokenizer path within snapshot (default colocated tokenizer.json)
    --disk-limit-bytes N       Existing cooperative cache policy (default16GiB)
    --source-memory-limit-bytes N  This process raw-source/staging cap (default5GiB)

Uses the ordinary CPU prompt/raw/max-tokens/prefill/audit options. No network,
credentials or implicit acquisition. Source GGUF remains capped at4GiB; bounded
verified original copies feed the same model engine. Memory accounting covers
raw source and sidecar staging, not all model/tokenizer/session memory or RAM."
        );
        return Ok(());
    }
    if args.is_empty()
        || args.len() > 8192
        || args
            .iter()
            .try_fold(0usize, |total, value| total.checked_add(value.len()))
            .is_none_or(|total| total > 2 * 1024 * 1024)
    {
        return Err(invalid());
    }
    let model_snapshot = &args[0];
    if model_snapshot.is_empty()
        || model_snapshot.len() > 4096
        || model_snapshot.chars().any(char::is_control)
    {
        return Err(invalid());
    }
    let mut root = None;
    let mut model_file = None;
    let mut config_file = None;
    let mut tokenizer_file = None;
    let mut config_snapshot = model_snapshot.clone();
    let mut tokenizer_snapshot = model_snapshot.clone();
    let mut disk = 16 * 1024 * 1024 * 1024u64;
    let mut memory = 5 * 1024 * 1024 * 1024u64;
    let mut cpu = vec!["run".to_string(), "verified cached model".to_string()];
    let mut seen = std::collections::BTreeSet::new();
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        if !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--raw" => cpu.push(flag.clone()),
            "--prompt"
            | "-p"
            | "--max-tokens"
            | "-n"
            | "--audit-logits-jsonl"
            | "--audit-layers-jsonl"
            | "--teacher-ids"
            | "--prefill-chunk" => {
                cpu.push(flag.clone());
                cpu.push(rest.next().ok_or_else(invalid)?.clone());
            }
            "--cache"
            | "--model-file"
            | "--config-file"
            | "--tokenizer-file"
            | "--config-snapshot"
            | "--tokenizer-snapshot"
            | "--disk-limit-bytes"
            | "--source-memory-limit-bytes" => {
                let value = rest.next().ok_or_else(invalid)?;
                if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
                    return Err(invalid());
                }
                match flag.as_str() {
                    "--cache" => {
                        if !Path::new(value).is_absolute() {
                            return Err(invalid());
                        }
                        root = Some(PathBuf::from(value));
                    }
                    "--model-file" => model_file = Some(value.clone()),
                    "--config-file" => config_file = Some(value.clone()),
                    "--tokenizer-file" => tokenizer_file = Some(value.clone()),
                    "--config-snapshot" => config_snapshot = value.clone(),
                    "--tokenizer-snapshot" => tokenizer_snapshot = value.clone(),
                    "--disk-limit-bytes" | "--source-memory-limit-bytes" => {
                        let number = value
                            .parse::<u64>()
                            .ok()
                            .filter(|v| *v > 0)
                            .ok_or_else(invalid)?;
                        if flag == "--disk-limit-bytes" {
                            disk = number;
                        } else {
                            memory = number;
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(invalid()),
        }
    }
    let model_file = model_file.ok_or_else(invalid)?;
    let parent = model_file.rsplit_once('/').map(|(parent, _)| parent);
    let colocated =
        |name: &str| parent.map_or_else(|| name.to_string(), |parent| format!("{parent}/{name}"));
    let config_file = config_file.unwrap_or_else(|| colocated("config.json"));
    let tokenizer_file = tokenizer_file.unwrap_or_else(|| colocated("tokenizer.json"));
    for file in [&model_file, &config_file, &tokenizer_file] {
        zerogravity::cache::paths::validate_relative_path(file)?;
    }
    let command = parse(cpu.iter());
    if !matches!(command, Command::Run { .. }) {
        return Err(invalid());
    }
    let (_, options) = cpu_options(command);
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "cache-model-source")?;
    arbiter.register(resource.clone(), memory)?;
    let admission = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("cached-model", "cli")?,
    };
    let store = CacheStore::open(CacheOptions::new(root, disk))?;
    let model = snapshot(&store, model_snapshot)?;
    let config = if &config_snapshot == model_snapshot {
        None
    } else {
        Some(snapshot(&store, &config_snapshot)?)
    };
    let tokenizer =
        if &tokenizer_snapshot == model_snapshot || tokenizer_snapshot == config_snapshot {
            None
        } else {
            Some(snapshot(&store, &tokenizer_snapshot)?)
        };
    let config_ref = config.as_ref().unwrap_or(&model);
    let tokenizer_ref = if &tokenizer_snapshot == model_snapshot {
        &model
    } else if tokenizer_snapshot == config_snapshot {
        config_ref
    } else {
        tokenizer
            .as_ref()
            .expect("distinct tokenizer snapshot opened")
    };
    let prepared = PreparedQwen::from_snapshots(
        &model,
        &model_file,
        config_ref,
        &config_file,
        tokenizer_ref,
        &tokenizer_file,
        &admission,
    )?;
    infer_prepared("verified cached model", prepared, options)
}
