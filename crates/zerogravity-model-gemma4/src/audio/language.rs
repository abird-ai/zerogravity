//! Audio-conditioned native F32 language reference, sharing the original
//! authenticated text/encoder source and private StateRoot transaction.
use super::{AudioTokens, Gemma4AudioEncoder};
use crate::{
    Error, Result,
    text::{Gemma4ReferenceSession, OriginalGemma4Text},
};
use serde::Deserialize;
use std::{
    ops::Range,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::cache::SnapshotLease;
use zerogravity_core::{
    identity::SourceIdentity, resource_arbiter::TransferAdmission, session_root::RootBinding,
};
use zerogravity_cpu::reference_text::ReferenceTextValue;

#[derive(Deserialize)]
struct ModalConfig {
    audio_token_id: u32,
    boa_token_id: u32,
    eoa_token_id: u32,
    image_token_id: u32,
    video_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
    text_config: TextConfig,
}
#[derive(Deserialize)]
struct TextConfig {
    pad_token_id: u32,
    vocab_size: usize,
    hidden_size: usize,
}
impl ModalConfig {
    fn parse(bytes: &[u8]) -> Result<Self> {
        let c: Self = serde_json::from_slice(bytes)
            .map_err(|e| Error::Malformed(format!("Gemma audio token config: {e}")))?;
        let ids = [
            c.audio_token_id,
            c.boa_token_id,
            c.eoa_token_id,
            c.text_config.pad_token_id,
        ];
        if ids
            .iter()
            .any(|&id| id as usize >= c.text_config.vocab_size)
            || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
        {
            return Err(Error::Malformed(
                "Gemma audio control token geometry".into(),
            ));
        }
        Ok(c)
    }
    fn non_audio_media(&self, id: u32) -> bool {
        [
            self.image_token_id,
            self.video_token_id,
            self.boi_token_id,
            self.eoi_token_id,
        ]
        .contains(&id)
    }
    fn slots(&self, tokens: &[u32], count: usize) -> Result<Range<usize>> {
        if tokens
            .iter()
            .any(|&id| self.non_audio_media(id) || id as usize >= self.text_config.vocab_size)
        {
            return Err(Error::InvalidInput(
                "foreign media or out-of-vocabulary audio prompt".into(),
            ));
        }
        let first = tokens
            .iter()
            .position(|&id| id == self.audio_token_id)
            .ok_or_else(|| {
                Error::InvalidInput("audio prompt requires original placeholder span".into())
            })?;
        let end = first
            .checked_add(count)
            .ok_or_else(|| Error::Limit("audio placeholder extent".into()))?;
        if count == 0
            || first == 0
            || end >= tokens.len()
            || tokens[first - 1] != self.boa_token_id
            || tokens[end] != self.eoa_token_id
            || tokens[first..end]
                .iter()
                .any(|&id| id != self.audio_token_id)
            || tokens
                .iter()
                .filter(|&&id| id == self.audio_token_id)
                .count()
                != count
            || tokens.iter().filter(|&&id| id == self.boa_token_id).count() != 1
            || tokens.iter().filter(|&&id| id == self.eoa_token_id).count() != 1
        {
            return Err(Error::InvalidInput(
                "audio token count/span/delimiters".into(),
            ));
        }
        Ok(first..end)
    }
}
pub struct Gemma4AudioLanguage {
    encoder: Gemma4AudioEncoder,
    text: Arc<OriginalGemma4Text>,
    config: ModalConfig,
}
impl Gemma4AudioLanguage {
    pub fn load(
        snapshot: &SnapshotLease,
        max_source_bytes: u64,
        admission: &TransferAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let source = crate::managed_source::open(snapshot, admission, max_source_bytes, cancelled)?;
        let config_json = snapshot.read_verified("config.json", 64 * 1024)?;
        let config = ModalConfig::parse(&config_json)?;
        let (text, _inventory) =
            crate::managed_source::bind_text(source.clone(), &config_json, cancelled)?;
        let encoder = Gemma4AudioEncoder::bind(snapshot, source, admission, cancelled)?;
        if !text.source().same_issuer(encoder.verified_source()) {
            return Err(Error::State("Gemma text/audio source owner differs".into()));
        }
        Ok(Arc::new(Self {
            encoder,
            text: Arc::new(text),
            config,
        }))
    }
    pub fn source(&self) -> SourceIdentity {
        self.text.source_identity()
    }
    pub fn encoder(&self) -> &Gemma4AudioEncoder {
        &self.encoder
    }
    pub fn session(self: &Arc<Self>, capacity: usize) -> Result<AudioLanguageSession> {
        Ok(AudioLanguageSession {
            model: self.clone(),
            text: Gemma4ReferenceSession::new(self.text.clone(), capacity)?,
        })
    }
}
pub struct AudioLanguageSession {
    model: Arc<Gemma4AudioLanguage>,
    text: Gemma4ReferenceSession,
}
impl AudioLanguageSession {
    pub fn binding(&self) -> RootBinding {
        self.text.binding()
    }
    /// Exactly one completed audio span and original BOS/turn/etc IDs.
    /// Caller owns prompt construction; no implied transcription template.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        audio: &AudioTokens,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ReferenceTextValue> {
        let slots = self.model.config.slots(tokens, audio.rows())?;
        if audio.width() != self.model.config.text_config.hidden_size {
            return Err(Error::InvalidInput("audio/text hidden width".into()));
        }
        self.text
            .forward_audio(
                tokens,
                audio,
                slots,
                self.model.config.text_config.pad_token_id,
                true,
                cancelled,
            )?
            .ok_or_else(|| Error::State("audio prefill did not produce logits".into()))
    }
    pub fn decode(
        &mut self,
        token: u32,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ReferenceTextValue> {
        if self.binding().frontier() == 0
            || self.model.config.non_audio_media(token)
            || [
                self.model.config.audio_token_id,
                self.model.config.boa_token_id,
                self.model.config.eoa_token_id,
            ]
            .contains(&token)
        {
            return Err(Error::InvalidInput(
                "audio language decode requires completed prefill and ordinary token".into(),
            ));
        }
        self.text
            .forward_reference(&[token], true, cancelled)?
            .ok_or_else(|| Error::State("audio decode did not produce logits".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_single_audio_span() {
        let config = ModalConfig {
            audio_token_id: 3,
            boa_token_id: 4,
            eoa_token_id: 5,
            image_token_id: 6,
            video_token_id: 7,
            boi_token_id: 8,
            eoi_token_id: 9,
            text_config: TextConfig {
                pad_token_id: 0,
                vocab_size: 6,
                hidden_size: 4,
            },
        };
        assert_eq!(config.slots(&[2, 4, 3, 3, 5, 2], 2).unwrap(), 2..4);
        assert!(config.slots(&[2, 4, 3, 5, 3, 5], 2).is_err());
        assert!(config.slots(&[3, 3, 5], 2).is_err());
        assert!(config.slots(&[4, 3, 3], 2).is_err());
        assert!(config.slots(&[4, 3, 3, 3, 5], 2).is_err());
    }
}
