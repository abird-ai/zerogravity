//! Original BERT WordPiece tokenizer from verified vocab/config. Construct
//! only the known BERT components, not an arbitrary serialized graph.
use super::{
    AllocationClass, Error, GlobalResourceLease, Result, SnapshotLease, TransferAdmission, reserve,
};
use ahash::AHashMap;
use serde::Deserialize;
use tokenizers::{
    AddedToken, Tokenizer, models::wordpiece::WordPiece, normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer, processors::bert::BertProcessing,
};

const MAX_VOCAB_BYTES: u64 = 4 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 16 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(rename = "_name_or_path", default)]
    _path: Option<String>,
    #[serde(rename = "name_or_path", default)]
    _name: Option<String>,
    tokenizer_class: String,
    do_lower_case: bool,
    do_basic_tokenize: bool,
    tokenize_chinese_chars: bool,
    strip_accents: Option<bool>,
    never_split: Option<Vec<String>>,
    model_max_length: usize,
    unk_token: String,
    sep_token: String,
    pad_token: String,
    cls_token: String,
    mask_token: String,
}

pub(super) struct BertTokenizer {
    inner: Tokenizer,
    _memory: GlobalResourceLease,
}
impl BertTokenizer {
    pub(super) fn load(
        snapshot: &SnapshotLease,
        c: &crate::BertSentenceConfig,
        a: &TransferAdmission,
    ) -> Result<Self> {
        let entry = snapshot
            .manifest()
            .entries()
            .iter()
            .find(|e| e.path() == "vocab.txt")
            .ok_or_else(|| Error::Malformed("BERT vocab.txt missing".into()))?;
        if entry.bytes() > MAX_VOCAB_BYTES {
            return Err(Error::Limit("BERT vocabulary byte bound".into()));
        }
        // Bounded construction and retained WordPiece maps (both directions),
        // token strings/hash tables, and five special-token matchers.
        // Retain the conservative construction envelope with this owner until
        // real-source tokenizer heap qualification can tighten the bound.
        let bytes = entry
            .bytes()
            .checked_mul(8)
            .and_then(|n| n.checked_add((c.vocabulary as u64) * 512))
            .and_then(|n| n.checked_add(2 * 1024 * 1024))
            .ok_or_else(super::overflow)?;
        let memory = reserve(a, AllocationClass::ImmutableWeights, bytes)?;
        let config: Config = serde_json::from_slice(
            &snapshot.read_verified("tokenizer_config.json", super::CONFIG_LIMIT)?,
        )
        .map_err(|e| Error::Malformed(format!("BERT tokenizer config: {e}")))?;
        if config.tokenizer_class != "BertTokenizer"
            || !config.do_basic_tokenize
            || config.never_split.as_ref().is_some_and(|v| !v.is_empty())
            || config.model_max_length < c.max_sequence
            || config.model_max_length > c.positions
            || config.unk_token != "[UNK]"
            || config.sep_token != "[SEP]"
            || config.pad_token != "[PAD]"
            || config.cls_token != "[CLS]"
            || config.mask_token != "[MASK]"
        {
            return Err(Error::Unsupported(
                "not the supported original BERT tokenizer profile".into(),
            ));
        }
        let bytes = snapshot.read_verified("vocab.txt", MAX_VOCAB_BYTES)?;
        let vocab = std::str::from_utf8(&bytes)
            .map_err(|_| Error::Malformed("BERT vocabulary UTF-8".into()))?;
        let mut map = AHashMap::with_capacity(c.vocabulary);
        for (id, token) in vocab.lines().enumerate() {
            if id >= c.vocabulary
                || token.is_empty()
                || token.len() > 256
                || map.insert(token.to_owned(), id as u32).is_some()
            {
                return Err(Error::Malformed(
                    "BERT vocabulary count/duplicates/token extent".into(),
                ));
            }
        }
        if map.len() != c.vocabulary {
            return Err(Error::Malformed(
                "BERT vocabulary size does not match checkpoint".into(),
            ));
        }
        let id = |token: &str| {
            map.get(token)
                .copied()
                .ok_or_else(|| Error::Malformed("BERT special token absent from vocab".into()))
        };
        let cls = id("[CLS]")?;
        let sep = id("[SEP]")?;
        id("[UNK]")?;
        id("[MASK]")?;
        if id("[PAD]")? != c.pad_id() {
            return Err(Error::Malformed(
                "BERT padding token/config mismatch".into(),
            ));
        }
        let model = WordPiece::builder()
            .vocab(map)
            .unk_token("[UNK]".into())
            .build()
            .map_err(|e| Error::Malformed(format!("BERT WordPiece: {e}")))?;
        let mut inner = Tokenizer::new(model);
        inner.with_normalizer(Some(BertNormalizer::new(
            true,
            config.tokenize_chinese_chars,
            config.strip_accents,
            config.do_lower_case,
        )));
        inner.with_pre_tokenizer(Some(BertPreTokenizer));
        inner.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".into(), sep),
            ("[CLS]".into(), cls),
        )));
        // These original special strings must not be normalized or split.
        inner.add_special_tokens(
            &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                .map(|s| AddedToken::from(s, true).normalized(false)),
        );
        if inner.get_vocab_size(true) != c.vocabulary {
            return Err(Error::Malformed(
                "BERT added token changed original vocabulary".into(),
            ));
        }
        Ok(Self {
            inner,
            _memory: memory,
        })
    }
    pub(super) fn encode(
        &self,
        text: &str,
        c: &crate::BertSentenceConfig,
        a: &TransferAdmission,
    ) -> Result<(Vec<u32>, GlobalResourceLease)> {
        if text.is_empty() || text.len() > MAX_TEXT_BYTES {
            return Err(Error::InvalidInput("BERT text byte extent".into()));
        }
        // Bounded single WordPiece sequence, no padding/truncation, no
        // arbitrary regex graph or dynamic tokenizer/model cache. Price
        // Unicode normalization/alignment, pieces and encoding generously.
        let bytes = (text.len() as u64)
            .checked_mul(2048)
            .and_then(|n| n.checked_add(256 * 1024))
            .ok_or_else(super::overflow)?;
        let charge = reserve(a, AllocationClass::Scratch, bytes)?;
        let encoded = self
            .inner
            .encode_fast(text, true)
            .map_err(|e| Error::InvalidInput(format!("BERT encode: {e}")))?;
        let ids = encoded.get_ids();
        if ids.len() > c.max_sequence
            || ids.len() <= 2
            || encoded.get_type_ids().iter().any(|&v| v != 0)
            || encoded.get_attention_mask().iter().any(|&v| v != 1)
            || ids.iter().any(|&v| v as usize >= c.vocabulary)
        {
            return Err(Error::InvalidInput(
                "BERT encoded sequence is empty or exceeds source bounds".into(),
            ));
        }
        Ok((ids.to_vec(), charge))
    }
}
