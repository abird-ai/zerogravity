//! Stream only requested source pieces; reuse the original whole-sequence decoder.
use super::*;
use serde::{
    Deserializer,
    de::{self, IgnoredAny, MapAccess, Visitor},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};
use tokenizers::{Decoder, decoders::DecoderWrapper};

struct Wanted<'a>(&'a [u32]);
struct VocabSeed<'a>(Wanted<'a>);
impl<'de> de::DeserializeSeed<'de> for VocabSeed<'_> {
    type Value = BTreeMap<u32, String>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        struct V<'a>(&'a [u32]);
        impl<'de> Visitor<'de> for V<'_> {
            type Value = BTreeMap<u32, String>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("authentic finite vocabulary")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out = BTreeMap::new();
                let mut count = 0;
                while let Some((piece, id)) = a.next_entry::<String, u32>()? {
                    count += 1;
                    if count > 262144 || piece.len() > 93 || id >= 262144 {
                        return Err(de::Error::custom("bounded authentic vocabulary"));
                    }
                    if self.0.contains(&id) && out.insert(id, piece).is_some() {
                        return Err(de::Error::custom("duplicate selected source ID"));
                    }
                }
                if count != 262144 || self.0.iter().any(|id| !out.contains_key(id)) {
                    return Err(de::Error::custom("complete selected source IDs"));
                }
                Ok(out)
            }
        }
        d.deserialize_map(V(self.0.0))
    }
}
struct ModelSeed<'a>(&'a [u32]);
impl<'de> de::DeserializeSeed<'de> for ModelSeed<'_> {
    type Value = BTreeMap<u32, String>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        struct V<'a>(&'a [u32]);
        impl<'de> Visitor<'de> for V<'_> {
            type Value = BTreeMap<u32, String>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("pinned source model")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut seen = BTreeSet::new();
                let mut vocab = None;
                while let Some(key) = a.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom("duplicate model field"));
                    }
                    match key.as_str() {
                        "vocab" => vocab = Some(a.next_value_seed(VocabSeed(Wanted(self.0)))?),
                        "type" => {
                            if a.next_value::<String>()? != "BPE" {
                                return Err(de::Error::custom("pinned BPE type"));
                            }
                        }
                        "dropout"
                        | "unk_token"
                        | "continuing_subword_prefix"
                        | "end_of_word_suffix"
                        | "fuse_unk"
                        | "byte_fallback"
                        | "ignore_merges"
                        | "merges" => {
                            a.next_value::<IgnoredAny>()?;
                        }
                        _ => return Err(de::Error::custom("unknown model field")),
                    }
                }
                if seen.len() != 10 {
                    return Err(de::Error::custom("complete source model"));
                }
                vocab.ok_or_else(|| de::Error::custom("source vocabulary absent"))
            }
        }
        d.deserialize_map(V(self.0))
    }
}
struct Selected {
    pieces: BTreeMap<u32, String>,
    special: BTreeSet<u32>,
    decoder: DecoderWrapper,
}
struct SourceSeed<'a>(&'a [u32]);
impl<'de> de::DeserializeSeed<'de> for SourceSeed<'_> {
    type Value = Selected;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Selected, D::Error> {
        struct V<'a>(&'a [u32]);
        impl<'de> Visitor<'de> for V<'_> {
            type Value = Selected;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("authentic source tokenizer")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Selected, A::Error> {
                let mut seen = BTreeSet::new();
                let (mut pieces, mut special, mut decoder) = (None, None, None);
                while let Some(key) = a.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom("duplicate source field"));
                    }
                    match key.as_str() {
                        "model" => pieces = Some(a.next_value_seed(ModelSeed(self.0))?),
                        "decoder" => decoder = Some(a.next_value::<DecoderWrapper>()?),
                        "added_tokens" => {
                            let added = a.next_value::<Vec<parse::Added>>()?;
                            if added.len() != 24 {
                                return Err(de::Error::custom("exact special source cohort"));
                            }
                            let mut ids = BTreeSet::new();
                            for token in added {
                                if !token.special
                                    || token.normalized
                                    || token.single_word
                                    || token.lstrip
                                    || token.rstrip
                                    || token.content.len() > 93
                                    || token.id >= 262144
                                    || !ids.insert(token.id)
                                {
                                    return Err(de::Error::custom("source special attributes"));
                                }
                            }
                            special = Some(ids);
                        }
                        "version" | "padding" | "truncation" | "normalizer" | "pre_tokenizer"
                        | "post_processor" => {
                            a.next_value::<IgnoredAny>()?;
                        }
                        _ => return Err(de::Error::custom("unknown source field")),
                    }
                }
                if seen.len() != 9 {
                    return Err(de::Error::custom("complete source tokenizer"));
                }
                Ok(Selected {
                    pieces: pieces.ok_or_else(|| de::Error::custom("source vocabulary"))?,
                    special: special.ok_or_else(|| de::Error::custom("source special tokens"))?,
                    decoder: decoder.ok_or_else(|| de::Error::custom("source decoder"))?,
                })
            }
        }
        d.deserialize_map(V(self.0))
    }
}
pub(super) fn decode(
    file: &Held,
    ids: &[u32],
    skip: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<String> {
    use serde::de::DeserializeSeed;
    let reader = BufReader::with_capacity(
        65536,
        Reader {
            source: file,
            at: 0,
            cancelled,
        },
    );
    let mut de = serde_json::Deserializer::from_reader(reader);
    let selected = SourceSeed(ids)
        .deserialize(&mut de)
        .map_err(|e| Error::Malformed(format!("selected source decoder: {e}")))?;
    de.end()
        .map_err(|e| Error::Malformed(format!("complete source decoder: {e}")))?;
    file.check()?;
    let pieces = ids
        .iter()
        .filter(|id| !skip || !selected.special.contains(id))
        .map(|id| {
            selected
                .pieces
                .get(id)
                .cloned()
                .ok_or_else(|| Error::Malformed("absent selected source piece".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    selected
        .decoder
        .decode(pieces)
        .map_err(|e| Error::Malformed(format!("source decoder: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_decoder_preserves_whole_sequence_invalid_bytes_and_special_boundaries() {
        let decoder:DecoderWrapper=serde_json::from_value(serde_json::json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})).unwrap();
        let decode = |pieces: &[&str]| {
            decoder
                .decode(pieces.iter().map(|p| p.to_string()).collect())
                .unwrap()
        };
        assert_eq!(decode(&["<0xE2>", "<0x82>"]), "��");
        assert_eq!(decode(&["<0xC3>", "<bos>", "<0xA9>"]), "�<bos>�");
        assert_eq!(decode(&["<0xC3>", "<0xA9>"]), "é");
        assert_eq!(decode(&["▁one", "<0x00>"]), " one\0");
    }
}
