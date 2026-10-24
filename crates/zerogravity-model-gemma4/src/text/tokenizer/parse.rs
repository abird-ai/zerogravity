//! Stream the authenticated source into upstream BPE; no whole JSON Value.
use super::*;
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use std::fmt;
use tokenizers::{
    AddedToken,
    models::bpe::{BPE, Vocab},
};

const VOCAB: usize = 262144;
const MERGES: usize = 514906;

struct Vocabulary(Vocab);
impl<'de> Deserialize<'de> for Vocabulary {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vocabulary;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("finite source BPE vocabulary")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out = Vocab::with_capacity(VOCAB);
                let mut ids = vec![false; VOCAB];
                while let Some((token, id)) = a.next_entry::<String, u32>()? {
                    if out.len() == VOCAB
                        || token.len() > 93
                        || id as usize >= VOCAB
                        || ids[id as usize]
                    {
                        return Err(de::Error::custom("bounded original vocabulary IDs/strings"));
                    }
                    ids[id as usize] = true;
                    if out.insert(token, id).is_some() {
                        return Err(de::Error::custom("duplicate source vocabulary"));
                    }
                }
                if out.len() != VOCAB {
                    return Err(de::Error::custom("complete source vocabulary"));
                }
                Ok(Vocabulary(out))
            }
        }
        d.deserialize_map(V)
    }
}
struct MergePairs(Vec<(String, String)>);
impl<'de> Deserialize<'de> for MergePairs {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = MergePairs;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("finite ordered source BPE pairs")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                out.try_reserve_exact(MERGES).map_err(de::Error::custom)?;
                while let Some([left, right]) = a.next_element::<[String; 2]>()? {
                    if out.len() == MERGES || left.len() > 90 || right.len() > 90 {
                        return Err(de::Error::custom("bounded original merge strings/count"));
                    }
                    out.push((left, right));
                }
                if out.len() != MERGES {
                    return Err(de::Error::custom("complete source merges"));
                }
                Ok(MergePairs(out))
            }
        }
        d.deserialize_seq(V)
    }
}
struct SourceBpe(BPE);
impl<'de> Deserialize<'de> for SourceBpe {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = SourceBpe;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("authenticated Gemma BPE source")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut seen = std::collections::BTreeSet::new();
                let (mut vocab, mut merges) = (None, None);
                while let Some(key) = a.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom("duplicate BPE field"));
                    }
                    match key.as_str() {
                        "type" => {
                            if a.next_value::<String>()? != "BPE" {
                                return Err(de::Error::custom("source BPE type"));
                            }
                        }
                        "dropout" | "continuing_subword_prefix" | "end_of_word_suffix" => {
                            if a.next_value::<serde_json::Value>()? != serde_json::Value::Null {
                                return Err(de::Error::custom("source BPE null option"));
                            }
                        }
                        "unk_token" => {
                            if a.next_value::<String>()? != "<unk>" {
                                return Err(de::Error::custom("source unknown token"));
                            }
                        }
                        "fuse_unk" | "byte_fallback" => {
                            if !a.next_value::<bool>()? {
                                return Err(de::Error::custom("source BPE true option"));
                            }
                        }
                        "ignore_merges" => {
                            if a.next_value::<bool>()? {
                                return Err(de::Error::custom("source BPE merge policy"));
                            }
                        }
                        "vocab" => vocab = Some(a.next_value::<Vocabulary>()?.0),
                        "merges" => merges = Some(a.next_value::<MergePairs>()?.0),
                        _ => return Err(de::Error::custom("unknown BPE source field")),
                    }
                }
                if seen.len() != 10 {
                    return Err(de::Error::custom("complete source BPE fields"));
                }
                let bpe = BPE::builder()
                    .vocab_and_merges(
                        vocab.ok_or_else(|| de::Error::custom("missing vocabulary"))?,
                        merges.ok_or_else(|| de::Error::custom("missing merges"))?,
                    )
                    .unk_token("<unk>".into())
                    .fuse_unk(true)
                    .byte_fallback(true)
                    .ignore_merges(false)
                    .cache_capacity(0)
                    .build()
                    .map_err(de::Error::custom)?;
                Ok(SourceBpe(bpe))
            }
        }
        d.deserialize_map(V)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Added {
    pub(super) id: u32,
    pub(super) content: String,
    pub(super) single_word: bool,
    pub(super) lstrip: bool,
    pub(super) rstrip: bool,
    pub(super) normalized: bool,
    pub(super) special: bool,
}
pub(super) struct Parsed(pub Tokenizer);
impl<'de> Deserialize<'de> for Parsed {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Parsed;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("authenticated whole tokenizer source")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut seen = std::collections::BTreeSet::new();
                let (mut model, mut normalizer, mut pre, mut post, mut decoder, mut added) =
                    (None, None, None, None, None, None);
                while let Some(key) = a.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom("duplicate tokenizer field"));
                    }
                    match key.as_str() {
                        "version" => {
                            if a.next_value::<String>()? != "1.0" {
                                return Err(de::Error::custom("source tokenizer version"));
                            }
                        }
                        "padding" | "truncation" => {
                            if a.next_value::<serde_json::Value>()? != serde_json::Value::Null {
                                return Err(de::Error::custom("no source padding/truncation"));
                            }
                        }
                        "model" => model = Some(a.next_value::<SourceBpe>()?.0),
                        "normalizer" => {
                            normalizer =
                                Some(a.next_value::<tokenizers::normalizers::NormalizerWrapper>()?)
                        }
                        "pre_tokenizer" => {
                            pre = Some(
                                a.next_value::<tokenizers::pre_tokenizers::PreTokenizerWrapper>()?,
                            )
                        }
                        "post_processor" => {
                            post = Some(
                                a.next_value::<tokenizers::processors::PostProcessorWrapper>()?,
                            )
                        }
                        "decoder" => {
                            decoder = Some(a.next_value::<tokenizers::decoders::DecoderWrapper>()?)
                        }
                        "added_tokens" => added = Some(a.next_value::<Vec<Added>>()?),
                        _ => return Err(de::Error::custom("unknown tokenizer source field")),
                    }
                }
                if seen.len() != 9 {
                    return Err(de::Error::custom("complete source tokenizer fields"));
                }
                let mut tokenizer =
                    Tokenizer::new(model.ok_or_else(|| de::Error::custom("missing model"))?);
                tokenizer
                    .with_normalizer(normalizer)
                    .with_pre_tokenizer(pre)
                    .with_post_processor(post)
                    .with_decoder(decoder);
                let added =
                    added.ok_or_else(|| de::Error::custom("missing added source tokens"))?;
                if added.len() != 24 {
                    return Err(de::Error::custom("exact24 source added tokens"));
                }
                for row in added {
                    let token = AddedToken {
                        content: row.content,
                        single_word: row.single_word,
                        lstrip: row.lstrip,
                        rstrip: row.rstrip,
                        normalized: row.normalized,
                        special: row.special,
                    };
                    if !token.special
                        || token.normalized
                        || token.single_word
                        || token.lstrip
                        || token.rstrip
                        || tokenizer.token_to_id(&token.content) != Some(row.id)
                    {
                        return Err(de::Error::custom(
                            "exact source special-token ID/attributes",
                        ));
                    }
                    tokenizer.add_special_tokens(&[token]);
                }
                if tokenizer.get_vocab_size(true) != VOCAB {
                    return Err(de::Error::custom("no invented tokenizer vocabulary"));
                }
                Ok(Parsed(tokenizer))
            }
        }
        d.deserialize_map(V)
    }
}
