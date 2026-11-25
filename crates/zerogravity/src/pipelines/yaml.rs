//! Bounded YAML decoding shared by SDK and CLI consumers.
use std::{fs, io::Read, path::Path};

use super::Document;
use crate::{Error, Result};

pub const MAX_CONFIG_BYTES: usize = 256 * 1024;

/// Bound the resolved YAML AST before typed deserialization. Every expanded
/// scalar and map key counts; custom tags cannot select executable components.
fn check_yaml_shape(value: &serde_yaml::Value) -> Result<()> {
    use serde_yaml::Value;
    const MAX_NODES: usize = 16_384;
    const MAX_DEPTH: usize = 32;
    let mut stack = vec![(value, 0usize)];
    let mut nodes = 0usize;
    let mut scalar_bytes = 0usize;
    while let Some((node, depth)) = stack.pop() {
        nodes += 1;
        if nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(Error::Limit("expanded YAML pipeline size/depth".into()));
        }
        match node {
            Value::Tagged(_) => {
                return Err(Error::Unsupported(
                    "YAML tags cannot select a pipeline component or executable constructor".into(),
                ));
            }
            Value::Mapping(map) => {
                for (key, value) in map {
                    if !matches!(key, Value::String(s) if s.len() <= 128) {
                        return Err(Error::Malformed(
                            "YAML mapping keys must be bounded strings".into(),
                        ));
                    }
                    stack.push((value, depth + 1));
                    stack.push((key, depth + 1));
                }
            }
            Value::Sequence(items) => {
                stack.extend(items.iter().map(|item| (item, depth + 1)));
            }
            Value::String(text) => {
                scalar_bytes = scalar_bytes
                    .checked_add(text.len())
                    .ok_or_else(|| Error::Limit("expanded YAML scalar bytes".into()))?;
                if scalar_bytes > MAX_CONFIG_BYTES {
                    return Err(Error::Limit("expanded YAML scalar bytes".into()));
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }
    Ok(())
}

/// Parse one bounded UTF-8 v1/v2 YAML document without filesystem access.
///
/// Anchors and shallow merges are supported, but duplicate/unknown fields,
/// custom tags, multiple documents and v2 null fields are refused. Parsing
/// alone does not validate the graph; call [`Document::check`] afterwards.
pub fn parse_document(bytes: &[u8]) -> Result<Document> {
    if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
        return Err(Error::Limit("pipeline config size limit".into()));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| Error::Malformed("pipeline config not UTF-8".into()))?;
    // No arbitrary code or custom YAML tags; even anchor/merge-expanded
    // values undergo the same bounded type and source-graph verification.
    let mut node: serde_yaml::Value = serde_yaml::from_str(text)
        .map_err(|_| Error::Malformed("invalid bounded pipeline YAML schema".into()))?;
    check_yaml_shape(&node)?;
    // YAML merge keys are a source-level convenience, not dynamic inheritance
    // across model registry, device capabilities or source authorization.
    node.apply_merge()
        .map_err(|_| Error::Malformed("invalid YAML mapping merge".into()))?;
    check_yaml_shape(&node)?;
    if node.get("schema").and_then(serde_yaml::Value::as_str) == Some("zerogravity.pipeline.v2") {
        fn no_null(v: &serde_yaml::Value) -> bool {
            match v {
                serde_yaml::Value::Null => false,
                serde_yaml::Value::Sequence(s) => s.iter().all(no_null),
                serde_yaml::Value::Mapping(m) => m.iter().all(|(k, v)| no_null(k) && no_null(v)),
                _ => true,
            }
        }
        if !no_null(&node) {
            return Err(Error::Malformed(
                "pipeline.v2 does not accept null fields; omit optional values".into(),
            ));
        }
    }
    serde_yaml::from_value(node)
        .map_err(|_| Error::Malformed("invalid typed pipeline YAML schema".into()))
}

/// Read only the explicitly selected config, with pre-read and post-read size
/// bounds. As in the CLI, config-file symlinks are followed; source paths inside
/// the document are neither resolved nor opened. No environment is consulted.
pub fn load_document(path: impl AsRef<Path>) -> Result<Document> {
    parse_document(&read_document_bytes(path)?)
}

/// Read once so compatibility classification and canonical parsing share
/// the same bounded bytes, rather than reopening a selected document.
pub fn read_document_bytes(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let bytes = fs::metadata(path)?.len();
    if bytes == 0 || bytes > MAX_CONFIG_BYTES as u64 {
        return Err(Error::Limit("pipeline config size limit".into()));
    }
    // Recheck after read: the source can change between metadata and read.
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
        return Err(Error::Limit(
            "pipeline config changed or exceeds size".into(),
        ));
    }
    Ok(bytes)
}
