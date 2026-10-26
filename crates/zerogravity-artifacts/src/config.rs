//! Bounded config bytes and JSON; architecture interpretation belongs to model crates.
use crate::{Error, Result};
use serde_json::Value;
use std::io::Read;
use std::path::Path;

pub fn read(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let limit = maximum
        .checked_add(1)
        .ok_or_else(|| Error::Limit("config byte bound overflow".into()))?;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(Error::Limit("config JSON exceeds byte bound".into()));
    }
    Ok(bytes)
}

pub fn json(bytes: &[u8], maximum: u64) -> Result<Value> {
    if bytes.len() as u64 > maximum {
        return Err(Error::Limit("config JSON exceeds byte bound".into()));
    }
    serde_json::from_slice(bytes).map_err(|e| Error::Malformed(format!("model config JSON: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_bound_precedes_json_and_file_access() {
        assert!(json(b"null", 4).is_ok());
        assert!(matches!(json(b"null", 3), Err(Error::Limit(_))));
        assert!(matches!(json(b"{", 4), Err(Error::Malformed(_))));
        assert!(matches!(
            read(Path::new("/unused-config-path"), u64::MAX),
            Err(Error::Limit(_))
        ));
    }
}
