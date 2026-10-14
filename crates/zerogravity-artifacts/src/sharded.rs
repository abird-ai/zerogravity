//! Bounded safetensors shard-index metadata. No shard is opened or verified.
use crate::{Error, Result};
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_INDEX: usize = 8 * 1024 * 1024;
const MAX_TENSORS: usize = 200_000;

/// A published name-to-shard directory, not a verified checkpoint allocation.
#[derive(Debug)]
pub struct SafetensorsIndex {
    total_size: u64,
    tensors: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndex {
    metadata: Metadata,
    weight_map: UniqueWeightMap,
}
#[derive(Deserialize)]
struct Metadata {
    // Preserve the validated JSON numeric lexeme: Number/as_f64 would round
    // away small fractions or values beyond2^53 before validation sees them.
    total_size: Box<serde_json::value::RawValue>,
}
struct UniqueWeightMap(BTreeMap<String, String>);
impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct WeightMapVisitor;
        impl<'de> Visitor<'de> for WeightMapVisitor {
            type Value = UniqueWeightMap;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("unique tensor names mapped to local safetensors shard basenames")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut tensors = BTreeMap::new();
                while let Some((name, shard)) = map.next_entry::<String, String>()? {
                    if name.is_empty() || name.len() > 65_535 || name.contains('\0') {
                        return Err(serde::de::Error::custom("invalid tensor name"));
                    }
                    if tensors.len() == MAX_TENSORS {
                        return Err(serde::de::Error::custom("too many indexed tensors"));
                    }
                    if !safe_shard(&shard) {
                        return Err(serde::de::Error::custom(
                            "unsafe safetensors shard basename",
                        ));
                    }
                    if tensors.insert(name, shard).is_some() {
                        return Err(serde::de::Error::custom("duplicate tensor name"));
                    }
                }
                Ok(UniqueWeightMap(tensors))
            }
        }
        deserializer.deserialize_map(WeightMapVisitor)
    }
}

fn safe_shard(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name.ends_with(".safetensors")
        && !name.starts_with('.')
        && !name.contains("..")
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

/// Published indexes may spell integer bytes as55562855904.0. Accept decimal
/// u64 digits and an optional ALL-ZERO fractional suffix exactly, never via
/// f64. Other JSON scalar types, signs, fractions and exponents are refused.
fn exact_byte_count(raw: &serde_json::value::RawValue) -> Result<u64> {
    let source = raw.get();
    let (whole, fraction) = source
        .split_once('.')
        .map_or((source, None), |(w, f)| (w, Some(f)));
    if whole.is_empty()
        || !whole.bytes().all(|c| c.is_ascii_digit())
        || fraction.is_some_and(|f| f.is_empty() || !f.bytes().all(|c| c == b'0'))
    {
        return Err(Error::Malformed(
            "safetensors total_size must be exact decimal integer bytes".into(),
        ));
    }
    let count = whole
        .parse::<u64>()
        .map_err(|_| Error::Limit("safetensors total_size exceeds u64".into()))?;
    if count == 0 {
        return Err(Error::Malformed("zero safetensors total_size".into()));
    }
    Ok(count)
}

impl SafetensorsIndex {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_INDEX {
            return Err(Error::Limit("safetensors shard index exceeds 8 MiB".into()));
        }
        let raw: RawIndex = serde_json::from_slice(bytes)
            .map_err(|e| Error::Malformed(format!("safetensors shard index: {e}")))?;
        let total_size = exact_byte_count(&raw.metadata.total_size)?;
        if raw.weight_map.0.is_empty() {
            return Err(Error::Malformed("empty safetensors shard index".into()));
        }
        Ok(Self {
            total_size,
            tensors: raw.weight_map.0,
        })
    }

    /// Declared payload bytes; not the sum of independently inspected shards.
    pub fn declared_total_size(&self) -> u64 {
        self.total_size
    }
    pub fn tensors(&self) -> &BTreeMap<String, String> {
        &self.tensors
    }
    pub fn shards(&self) -> BTreeSet<&str> {
        self.tensors.values().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_float_size_and_shared_shard_are_metadata_only() {
        let index = SafetensorsIndex::from_json(br#"{"metadata":{"total_size":12.0},"weight_map":{"a":"model-01.safetensors","b":"model-01.safetensors"}}"#).unwrap();
        assert_eq!(index.declared_total_size(), 12);
        assert_eq!(index.tensors().len(), 2);
        assert_eq!(index.shards().len(), 1);
    }

    #[test]
    fn duplicate_keys_unsafe_paths_and_bad_sizes_refuse() {
        for bytes in [
            r#"{"metadata":{"total_size":1},"weight_map":{"a":"x.safetensors","a":"y.safetensors"}}"#,
            r#"{"metadata":{"total_size":1},"weight_map":{},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":1},"weight_map":{"a":"../x.safetensors"}}"#,
            r#"{"metadata":{"total_size":1},"weight_map":{"a":"/x.safetensors"}}"#,
            r#"{"metadata":{"total_size":1},"weight_map":{"a":"dir/x.safetensors"}}"#,
            r#"{"metadata":{"total_size":1},"weight_map":{"a":"x.bin"}}"#,
            r#"{"metadata":{"total_size":0},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":1.5},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":55562855904.00000001},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":9007199254740993.1},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":18446744073709551616.0},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":1e3},"weight_map":{"a":"x.safetensors"}}"#,
            r#"{"metadata":{"total_size":"12"},"weight_map":{"a":"x.safetensors"}}"#,
        ] {
            assert!(
                SafetensorsIndex::from_json(bytes.as_bytes()).is_err(),
                "{bytes}"
            );
        }
        assert!(SafetensorsIndex::from_json(&vec![b' '; MAX_INDEX + 1]).is_err());
    }

    #[test]
    fn exact_decimal_size_never_rounds_through_f64() {
        for (literal, expected) in [
            ("55562855904.0", 55_562_855_904),
            ("9007199254740993.0", 9_007_199_254_740_993),
            ("9007199254740994", 9_007_199_254_740_994),
            ("18446744073709551615.00", u64::MAX),
        ] {
            let raw = format!(
                r#"{{"metadata":{{"total_size":{literal}}},"weight_map":{{"a":"x.safetensors"}}}}"#
            );
            assert_eq!(
                SafetensorsIndex::from_json(raw.as_bytes())
                    .unwrap()
                    .declared_total_size(),
                expected
            );
        }
    }
}
