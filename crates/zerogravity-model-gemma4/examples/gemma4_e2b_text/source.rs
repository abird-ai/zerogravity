use super::io::*;
use serde_json::Value;
use std::{fs::File, path::Path};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};
use zerogravity_model_gemma4::text::{Gemma4SourcePins, OriginalGemma4Text};
pub(super) fn pins() -> Proof<Value> {
    Ok(serde_json::from_slice(include_bytes!(
        "../../../../tools/gemma4-e2b-reference/source-pin.json"
    ))?)
}
pub(super) struct Held {
    file: File,
    bytes: u64,
    sha: String,
    before: [u64; 7],
}
pub(super) fn load(
    directory: &Path,
    host: &TransferAdmission,
) -> Proof<(OriginalGemma4Text, Vec<Held>)> {
    let pins = pins()?;
    let mut held = Vec::new();
    let mut shards = Vec::new();
    let mut config = None;
    let files = pins["files"]
        .as_array()
        .ok_or("complete original source registry")?;
    need(files.len() == 8, "all actual acquired source members")?;
    for row in files {
        let name = string(row, "name")?;
        need(
            Path::new(name).file_name().and_then(|s| s.to_str()) == Some(name),
            "contained source name",
        )?;
        let file = open(&directory.join(name))?;
        let before = stamp(&file)?;
        let bytes = row["bytes"].as_u64().ok_or("source bytes")?;
        let expected = string(row, "sha256")?.to_owned();
        need(before[2] == bytes, "exact held original source size")?;
        if name == "model.safetensors" {
            shards.push(ExpectedSourceShard::new(
                name.into(),
                Artifact::from_file(file.try_clone()?, Path::new(name))?,
                bytes,
                expected.clone(),
            )?);
        } else {
            need(
                hash(&file, bytes)? == expected,
                "complete actual source sidecar SHA",
            )?;
            if name == "config.json" {
                config = Some(read(&directory.join(name), 65536, Some(row))?);
            }
        }
        held.push(Held {
            file,
            bytes,
            sha: expected,
            before,
        });
    }
    let index = include_bytes!(
        "../../../../tools/gemma4-e2b-reference/derived-model.safetensors.index.json"
    );
    // This is the authenticated unsharded header-derived closure, not an official HF sidecar.
    let source = SourceWeights::prepare_verified_windows(
        index,
        string(&pins, "index_sha256")?,
        shards,
        WindowOpenLimits {
            max_source_bytes: 12 * 1024 * 1024 * 1024,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 2011,
            scan_chunk_bytes: 1024 * 1024,
        },
        host,
    )?
    .open()?;
    let model = OriginalGemma4Text::bind(
        source,
        &config.ok_or("actual original config")?,
        Gemma4SourcePins {
            repository: string(&pins, "repository")?,
            revision: string(&pins, "revision")?,
            config_sha256: string(&pins, "config_sha256")?,
            source_index_sha256: string(&pins, "index_sha256")?,
            source_bundle_sha256: string(&pins, "bundle_sha256")?,
        },
        None,
    )?;
    need(
        !model.trained_model_qualified(),
        "source binding is not trained numerical acceptance",
    )?;
    Ok((model, held))
}
pub(super) fn after(held: &[Held]) -> Proof<()> {
    for row in held {
        need(
            stamp(&row.file)? == row.before && hash(&row.file, row.bytes)? == row.sha,
            "complete held original source unchanged after computation",
        )?;
    }
    Ok(())
}
