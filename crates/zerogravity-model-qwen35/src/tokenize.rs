//! Version-pinned Rust tokenizer adapter. The trained model's special tokens
//! and pre-tokenizer rules are loaded from its official tokenizer.json.
//! No generated model code or foreign inference runtime is executed.

use crate::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct QwenTokenizer {
    inner: Tokenizer,
    eos: u32,
    begin: u32,
    end: u32,
}

impl QwenTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        if std::fs::metadata(path)?.len() > 64 * 1024 * 1024 {
            return Err(Error::Limit("tokenizer.json exceeds 64 MiB".into()));
        }
        let inner = Tokenizer::from_file(path)
            .map_err(|e| Error::Malformed(format!("tokenizer.json: {e}")))?;
        Self::from_tokenizer(inner)
    }

    /// Parse the actual bounded owned bytes verified by a caller, sharing the
    /// ordinary tokenizer's special-token checks and tokenization behavior.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(Error::Limit("tokenizer.json exceeds 64 MiB".into()));
        }
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|e| Error::Malformed(format!("tokenizer.json: {e}")))?;
        Self::from_tokenizer(inner)
    }

    fn from_tokenizer(inner: Tokenizer) -> Result<Self> {
        let begin = inner
            .token_to_id("<|im_start|>")
            .ok_or_else(|| Error::Malformed("Qwen tokenizer missing im_start".into()))?;
        let end = inner
            .token_to_id("<|im_end|>")
            .ok_or_else(|| Error::Malformed("Qwen tokenizer missing im_end".into()))?;
        let eos = end;
        if begin == end {
            return Err(Error::Malformed(
                "Qwen template start/end token collision".into(),
            ));
        }
        Ok(Self {
            inner,
            eos,
            begin,
            end,
        })
    }

    pub fn prompt_tokens(&self, message: &str, chat: bool) -> Result<Vec<u32>> {
        if message.len() > 1024 * 1024 {
            return Err(Error::Limit("prompt input exceeds 1 MiB".into()));
        }
        let rendered = if chat {
            format!("<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            message.to_owned()
        };
        self.encode(&rendered)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Malformed(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special)
            .map_err(|e| Error::Malformed(format!("tokenizer decode: {e}")))
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn special_ids(&self) -> (u32, u32) {
        (self.begin, self.end)
    }
}
