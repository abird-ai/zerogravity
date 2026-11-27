//! Bounded deterministic transcription on the original source-bound audio
//! encoder, language traversal, tokenizer and state publisher. Not a wire
//! capability or a general-purpose chat/template interpreter.
use super::{AudioLanguageSession, Gemma4AudioLanguage};
use crate::{
    Error, Result,
    text::tokenizer::{
        Gemma4DecodedText, Gemma4EncodedText, Gemma4TextTokenizer, Gemma4TokenizerFiles,
    },
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use zerogravity_artifacts::cache::SnapshotLease;
use zerogravity_core::{
    identity::SourceIdentity,
    resource_arbiter::{
        AllocationClass, AllocationRequest, GlobalResourceLease, TransferAdmission,
    },
};
use zerogravity_cpu::{ops::encoder::check_cancel, reference_text::ReferenceTextValue};

pub const MAX_TRANSCRIPT_TOKENS: usize = 64;
pub const MAX_TRANSCRIPT_BYTES: usize = 16384;
pub const MAX_CONTEXT: usize = 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptFinish {
    Stop,
    Length,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechTask {
    Transcribe,
    TranslateEnglish,
}
impl SpeechTask {
    fn instruction(self) -> &'static str {
        match self {
            Self::Transcribe => "Transcribe the following speech segment in its original language.",
            Self::TranslateEnglish => {
                "Transcribe the following speech segment, then translate it into English. Output the transcription, then a newline and 'English: ' followed by the translation."
            }
        }
    }
}

/// The original AST prompt asks for two lines. Only the explicitly marked
/// target line is a translation. Never return an unlabelled source transcript.
fn translation_range(text: &str) -> Option<std::ops::Range<usize>> {
    let (source, english) = text.split_once("\nEnglish: ")?;
    if source.trim().is_empty()
        || source.contains('\n')
        || english.trim().is_empty()
        || english.contains('\n')
    {
        return None;
    }
    let start = source.len() + "\nEnglish: ".len();
    Some(start..text.len())
}

pub struct Gemma4Transcriber {
    model: Arc<Gemma4AudioLanguage>,
    tokenizer: Mutex<Gemma4TextTokenizer>,
    admission: TransferAdmission,
}
impl Gemma4Transcriber {
    pub fn load(
        snapshot: &SnapshotLease,
        max_source_bytes: u64,
        admission: &TransferAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        crate::managed_source::require(snapshot)?;
        // Require the actual authenticated template, not merely its manifest
        // metadata. The renderer supports only its pinned ASR turn subset.
        let _scratch = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 65536,
            }],
        )?;
        let _template = snapshot.read_verified("chat_template.jinja", 65536)?;
        let file = |name| snapshot.file(name)?.try_clone().map_err(Error::from);
        let tokenizer = Gemma4TextTokenizer::prepare(
            Gemma4TokenizerFiles {
                tokenizer: file("tokenizer.json")?,
                tokenizer_config: file("tokenizer_config.json")?,
                generation_config: file("generation_config.json")?,
                model_config: file("config.json")?,
            },
            admission,
            cancelled,
        )?;
        let model = Gemma4AudioLanguage::load(snapshot, max_source_bytes, admission, cancelled)?;
        Ok(Arc::new(Self {
            model,
            tokenizer: Mutex::new(tokenizer),
            admission: admission.clone(),
        }))
    }
    pub fn source(&self) -> SourceIdentity {
        self.model.source()
    }
    pub fn max_samples(&self) -> usize {
        self.model.encoder().max_samples()
    }
    pub fn max_context(&self) -> usize {
        MAX_CONTEXT
    }
    pub fn max_output_tokens(&self) -> usize {
        MAX_TRANSCRIPT_TOKENS
    }
    pub fn max_output_bytes(&self) -> usize {
        MAX_TRANSCRIPT_BYTES
    }
    pub fn validate_admission(&self, admission: &TransferAdmission) -> Result<()> {
        self.model.encoder().validate_admission(admission)
    }
    pub fn prompt_tokens(
        &self,
        samples: usize,
        limit: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4EncodedText> {
        self.prompt_tokens_for(SpeechTask::Transcribe, samples, limit, cancelled)
    }
    pub fn prompt_tokens_for(
        &self,
        task: SpeechTask,
        samples: usize,
        limit: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4EncodedText> {
        let rows = self.model.encoder().token_count(samples)?;
        self.tokenizer
            .lock()
            .map_err(|_| Error::State("Gemma tokenizer mutex".into()))?
            .speech_prompt(rows, limit, task.instruction(), cancelled)
    }
    pub fn start(
        self: &Arc<Self>,
        samples: &[f32],
        sample_rate: u32,
        context: usize,
        max_tokens: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4Transcription> {
        self.start_task(
            SpeechTask::Transcribe,
            samples,
            sample_rate,
            context,
            max_tokens,
            cancelled,
        )
    }
    pub fn start_task(
        self: &Arc<Self>,
        task: SpeechTask,
        samples: &[f32],
        sample_rate: u32,
        context: usize,
        max_tokens: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4Transcription> {
        if context > MAX_CONTEXT || !(1..=MAX_TRANSCRIPT_TOKENS).contains(&max_tokens) {
            return Err(Error::Limit(
                "Gemma transcription context/output bound".into(),
            ));
        }
        let prompt_limit = context
            .checked_sub(max_tokens)
            .ok_or_else(|| Error::Limit("Gemma transcription context".into()))?;
        let prompt = self.prompt_tokens_for(task, samples.len(), prompt_limit, cancelled)?;
        let prompt_tokens = prompt.ids().len();
        let memory = self.admission.arbiter.admit(
            &self.admission.resource,
            self.admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: (max_tokens * 4 + 4096) as u64,
            }],
        )?;
        let audio = self
            .model
            .encoder()
            .encode_pcm(samples, sample_rate, cancelled)?;
        let mut session = self.model.session(prompt_tokens + max_tokens)?;
        let logits = session.prefill(prompt.ids(), &audio, cancelled)?;
        Ok(Gemma4Transcription {
            model: self.clone(),
            task,
            session,
            logits: Some(logits),
            ids: Vec::with_capacity(max_tokens),
            prompt_tokens,
            max_tokens,
            _memory: memory,
        })
    }
}

pub struct Gemma4Transcript {
    text: Gemma4DecodedText,
    range: std::ops::Range<usize>,
    input_tokens: usize,
    output_tokens: usize,
    finish: TranscriptFinish,
}
impl Gemma4Transcript {
    pub fn text(&self) -> &str {
        &self.text.as_str()[self.range.clone()]
    }
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }
    pub fn output_tokens(&self) -> usize {
        self.output_tokens
    }
    pub fn finish(&self) -> TranscriptFinish {
        self.finish
    }
}

/// One prefill, then one actual cached decode quantum per call. No internal
/// worker or retry loop; the caller schedules these bounded quanta.
pub struct Gemma4Transcription {
    model: Arc<Gemma4Transcriber>,
    task: SpeechTask,
    session: AudioLanguageSession,
    logits: Option<ReferenceTextValue>,
    ids: Vec<u32>,
    prompt_tokens: usize,
    max_tokens: usize,
    _memory: GlobalResourceLease,
}
impl Gemma4Transcription {
    pub fn input_tokens(&self) -> usize {
        self.prompt_tokens
    }
    pub fn output_ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn step(&mut self, cancelled: Option<&AtomicBool>) -> Result<Option<Gemma4Transcript>> {
        check_cancel(cancelled)?;
        let logits = self
            .logits
            .take()
            .ok_or_else(|| Error::State("completed or failed transcription".into()))?;
        let values = logits.as_slice()?;
        if values.len() != 262144 || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed("Gemma transcript logits".into()));
        }
        // Deterministic greedy selection, first token on ties, not a
        // silent substitution for the original stochastic generation config.
        let token = values
            .iter()
            .enumerate()
            .reduce(|a, b| if b.1 > a.1 { b } else { a })
            .ok_or_else(|| Error::Malformed("empty transcript logits".into()))?
            .0 as u32;
        drop(logits);
        self.ids.push(token);
        let finish = if [1, 106, 50].contains(&token) {
            Some(TranscriptFinish::Stop)
        } else if self.ids.len() == self.max_tokens {
            Some(TranscriptFinish::Length)
        } else {
            None
        };
        if let Some(finish) = finish {
            let content = if finish == TranscriptFinish::Stop {
                &self.ids[..self.ids.len() - 1]
            } else {
                &self.ids
            };
            let text = self
                .model
                .tokenizer
                .lock()
                .map_err(|_| Error::State("Gemma tokenizer mutex".into()))?
                .decode(content, true, cancelled)?;
            let range =
                match self.task {
                    SpeechTask::Transcribe => 0..text.as_str().len(),
                    SpeechTask::TranslateEnglish => match translation_range(text.as_str()) {
                        Some(range) => range,
                        None if finish == TranscriptFinish::Length => 0..0,
                        None => return Err(Error::Malformed(
                            "Gemma translation missing the required bilingual terminal structure"
                                .into(),
                        )),
                    },
                };
            check_cancel(cancelled)?;
            return Ok(Some(Gemma4Transcript {
                text,
                range,
                input_tokens: self.prompt_tokens,
                output_tokens: self.ids.len(),
                finish,
            }));
        }
        self.logits = Some(self.session.decode(token, cancelled)?);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn translation_is_exact_target_line_or_refusal() {
        let text = "Hola mundo\nEnglish: Hello world";
        assert_eq!(&text[translation_range(text).unwrap()], "Hello world");
        let text = "こんにちは\nEnglish: Hello";
        assert_eq!(&text[translation_range(text).unwrap()], "Hello");
        for wrong in [
            "Hola mundo",
            "Hello world",
            "English: Hello world",
            "\nEnglish: Hello",
            "Hola\nEnglish: ",
            "Hola\nEnglish: Hello\nOther: World",
            "Hola\nEnglish: Hello\nEnglish: World",
        ] {
            assert!(translation_range(wrong).is_none(), "{wrong:?}");
        }
    }
}
