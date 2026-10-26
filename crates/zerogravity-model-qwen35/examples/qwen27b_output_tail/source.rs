use super::io::*;
use serde_json::Value;
use std::{fs::File, path::Path};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};
use zerogravity_model_qwen35::source::binding::{OriginalWindowOperators, QwenNumericalProfile};
pub(super) const CONFIG: &str = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab";
pub(super) const INDEX: &str = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
pub(super) const SNAPSHOT: &str =
    "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
pub(super) fn pins() -> Proof<Value> {
    Ok(serde_json::from_slice(include_bytes!(
        "../../../../tools/qwen27b-output-tail-reference/source-pin.json"
    ))?)
}
pub(super) struct Held {
    pub file: File,
    pub bytes: u64,
    pub sha: String,
    pub before: [u64; 7],
}
pub(super) fn open_source(
    directory: &Path,
    host: &TransferAdmission,
) -> Proof<(OriginalWindowOperators, Vec<Held>)> {
    need(
        directory.is_absolute(),
        "explicit absolute original source directory",
    )?;
    let pins = pins()?;
    need(
        pins["snapshot"] == SNAPSHOT
            && pins["repository"] == "Qwen/Qwen3.8-27B"
            && pins["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "compiled original27B source registry",
    )?;
    let rows = pins["files"].as_array().ok_or("source pins")?;
    need(rows.len() == 31, "complete original source file registry")?;
    let mut weight_bytes = 0u64;
    let mut shards = Vec::new();
    let mut held = Vec::new();
    let mut config = None;
    let mut index = None;
    for row in rows {
        let name = string(row, "name")?;
        need(
            Path::new(name).file_name().and_then(|s| s.to_str()) == Some(name),
            "contained original source filename",
        )?;
        let file = open(&directory.join(name))?;
        let bytes = row["bytes"].as_u64().ok_or("source byte extent")?;
        let before = stamp(&file)?;
        need(before[2] == bytes, "exact original source file bytes")?;
        let expected = string(row, "sha256")?.to_owned();
        if name.ends_with(".safetensors") {
            weight_bytes = weight_bytes
                .checked_add(bytes)
                .ok_or("whole source extent overflow")?;
            shards.push(ExpectedSourceShard::new(
                name.into(),
                Artifact::from_file(file.try_clone()?, Path::new(name))?,
                bytes,
                expected.clone(),
            )?);
        } else {
            need(
                hash(&file, bytes)? == expected,
                "complete original metadata SHA",
            )?;
            if name == "config.json" {
                config = Some(read(&directory.join(name), 65536, Some(row))?);
            }
            if name == "model.safetensors.index.json" {
                index = Some(read(&directory.join(name), 256 * 1024, Some(row))?);
            }
        }
        held.push(Held {
            file,
            bytes,
            sha: expected,
            before,
        });
    }
    need(
        shards.len() == 18 && weight_bytes == 55_563_006_776,
        "all18 authentic shard extents",
    )?;
    let plan = SourceWeights::prepare_verified_windows(
        &index.ok_or("original index")?,
        INDEX,
        shards,
        WindowOpenLimits {
            max_source_bytes: 64 * 1024 * 1024 * 1024,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_shards: 18,
            max_tensors: 1199,
            scan_chunk_bytes: 4 * 1024 * 1024,
        },
        host,
    )?;
    let source = plan.open()?;
    need(
        source.verified_bundle_sha256()
            == "7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2"
            && source.verified_shard_index_sha256() == INDEX,
        "whole original bound bundle/index",
    )?;
    let operators = OriginalWindowOperators::bind(
        source,
        &config.ok_or("original config")?,
        CONFIG,
        INDEX,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )?;
    let b = operators.binding().config().geometry();
    need(
        b.hidden_size == 5120 && b.vocab_size == 248320 && b.layers.len() == 64,
        "complete final63 source geometry",
    )?;
    Ok((operators, held))
}
pub(super) fn verify_after(held: &[Held]) -> Proof<()> {
    for h in held {
        need(
            stamp(&h.file)? == h.before && hash(&h.file, h.bytes)? == h.sha,
            "same full held original source after computation/caller drops",
        )?;
    }
    Ok(())
}
