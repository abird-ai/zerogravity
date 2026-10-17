//! Ordered, source-neutral model family intent registry.
//!
//! This is NOT the loaded/authorized model catalog. It contains no source
//! credentials, published cache leases, downloaded weights, executable kernels,
//! quant conversion recipes or inferred device capability. A family appearing
//! here must never cause auto-pull, auto-load, auto-codec or fallback.
//!
//! The sole executable model catalog remains the independently source-bound
//! Qwen catalog; modality-specific execution must pass its own source/device
//! gates before adding any executable status.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskKind {
    Language,
    TextImage,
    ImageGeneration,
    SpeechRecognition,
    SpeechSynthesis,
    MultiModal,
}
impl TaskKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Language => "text",
            Self::TextImage => "text+image",
            Self::ImageGeneration => "image-generation",
            Self::SpeechRecognition => "audio-asr",
            Self::SpeechSynthesis => "audio-tts",
            Self::MultiModal => "multimodal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceLayout {
    IndexedSafetensors,
    SingleSafetensors,
    GgufSelection,
    DiffusersAssembly,
    NemoArchive,
    Unknown,
}
impl SourceLayout {
    pub const fn label(self) -> &'static str {
        match self {
            Self::IndexedSafetensors => "pinned-indexed-safetensors",
            Self::SingleSafetensors => "pinned-single-safetensors",
            Self::GgufSelection => "explicit-gguf-quant-family",
            Self::DiffusersAssembly => "diffusers-multi-component-assembly",
            Self::NemoArchive => "nemo-archive",
            Self::Unknown => "unverified-upstream-layout",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceStatus {
    /// Original public Hub pin/index/selected file metadata actually resolved.
    VerifiedMetadataOnly,
    /// A named public upstream repository exists; its exact file closure has
    /// not yet passed the original runtime's source resolver.
    CataloguedUnqualified,
    /// Unverified family placeholder; neither repository nor release inferred.
    Unverified,
}
impl SourceStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::VerifiedMetadataOnly => "verified-metadata-only",
            Self::CataloguedUnqualified => "catalogued-unqualified",
            Self::Unverified => "unverified",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionStatus {
    /// A separate source-specific path exists, NOT proof this registry member
    /// or its upstream HF source can execute.
    Qwen35ResidentGgufBaseline,
    NotImplemented,
}
impl ExecutionStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Qwen35ResidentGgufBaseline => "source-specific-qwen35-gguf-baseline-only",
            Self::NotImplemented => "NOT IMPLEMENTED",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelTarget {
    pub slug: &'static str,
    pub priority: u8,
    pub task: TaskKind,
    pub repository: Option<&'static str>,
    pub source_layout: SourceLayout,
    pub source_status: SourceStatus,
    pub execution_status: ExecutionStatus,
    /// A model may be described without authorizing its acquisition. DeepSeek
    /// V4.1-Flash and GLM5.3-Flash remain on explicit payload hold.
    pub payload_hold: bool,
    pub note: &'static str,
}
// Finite declarative const registry keeps each model tuple readable; callers
// cannot construct executable/source capability from these fields.
#[allow(clippy::too_many_arguments)]
const fn target(
    slug: &'static str,
    priority: u8,
    task: TaskKind,
    repository: Option<&'static str>,
    source_layout: SourceLayout,
    source_status: SourceStatus,
    execution_status: ExecutionStatus,
    payload_hold: bool,
    note: &'static str,
) -> ModelTarget {
    ModelTarget {
        slug,
        priority,
        task,
        repository,
        source_layout,
        source_status,
        execution_status,
        payload_hold,
        note,
    }
}
use ExecutionStatus::{NotImplemented as NI, Qwen35ResidentGgufBaseline as QB};
use SourceLayout::{
    DiffusersAssembly as DA, GgufSelection as GG, IndexedSafetensors as IX, NemoArchive as NA,
    SingleSafetensors as SS, Unknown as U,
};
use SourceStatus::{CataloguedUnqualified as CU, Unverified as UV, VerifiedMetadataOnly as VM};
use TaskKind::{
    ImageGeneration as IG, Language as L, MultiModal as MM, SpeechRecognition as ASR,
    SpeechSynthesis as TTS, TextImage as TI,
};

/// Canonical order, with distinct family and exact source aliases. Additional
/// family placeholders never masquerade as published model identifiers.
pub const MODEL_TARGETS: &[ModelTarget] = &[
    target(
        "qwen3.5-plus",
        1,
        MM,
        None,
        U,
        UV,
        NI,
        false,
        "Qwen 3.5+ future family; no verified source identity. Do not alias an earlier model.",
    ),
    target(
        "qwen3.5-0.8b",
        1,
        TI,
        Some("Qwen/Qwen3.5-0.8B"),
        IX,
        VM,
        QB,
        false,
        "Exact HF checkpoint and separately qualified local Qwen3.5 GGUF are distinct sources.",
    ),
    target(
        "qwen3.5-2b",
        1,
        TI,
        Some("Qwen/Qwen3.5-2B"),
        IX,
        VM,
        QB,
        false,
        "First large-source CPU/GPU target. HF and original resident GGUF are distinct.",
    ),
    target(
        "qwen3.8-27b",
        1,
        TI,
        Some("Qwen/Qwen3.8-27B"),
        IX,
        VM,
        NI,
        false,
        "18-shard source metadata verified. Trained CPU/GPU and SSD execution not implemented.",
    ),
    target(
        "qwen3.8-flash-next",
        1,
        MM,
        Some("Qwen/Qwen3.8-Flash-Next"),
        IX,
        VM,
        NI,
        false,
        "Source-specific sparse/hybrid expert and optional MTP heads need verified contracts; n-gram is a runtime decoder, not downloaded model.",
    ),
    target(
        "qwen3.8-flash-next-fp8",
        1,
        MM,
        Some("Qwen/Qwen3.8-Flash-Next-FP8"),
        IX,
        CU,
        NI,
        false,
        "Original provider FP8 is not Q2/Q3.x/Q4 and needs its own exact codec.",
    ),
    target(
        "qwen-gguf-quant",
        1,
        L,
        None,
        GG,
        UV,
        NI,
        false,
        "User-selected third-party repo/immutable revision/explicit Q2 or Q3 or Q4 filename family required; no guessed quant source or quality parity.",
    ),
    target(
        "gemma4-e2b-it",
        2,
        MM,
        Some("google/gemma-4-E2B-it"),
        SS,
        VM,
        NI,
        false,
        "Actual single-file source metadata verified. PLE and speech/image codecs unqualified.",
    ),
    target(
        "gemma4-26b-a4b-it",
        2,
        TI,
        Some("google/gemma-4-26B-A4B-it"),
        IX,
        VM,
        NI,
        false,
        "Two-shard source metadata verified. Routed MoE reduction not yet trained-qualified.",
    ),
    target(
        "gemma4-31b-it",
        2,
        TI,
        Some("google/gemma-4-31B-it"),
        IX,
        VM,
        NI,
        false,
        "Dense 31B BF16 and separate derived quant pipelines require owned source binding.",
    ),
    target(
        "gemma4-quant",
        2,
        TI,
        None,
        GG,
        UV,
        NI,
        false,
        "Exact third-party GGUF source, quant semantics and trained parity need qualification.",
    ),
    target(
        "deepseek-v4.1-flash",
        3,
        MM,
        Some("deepseek-ai/DeepSeek-V4.1-Flash"),
        IX,
        CU,
        NI,
        true,
        "PIPELINE ONLY: user explicitly forbids payload acquisition at this stage.",
    ),
    target(
        "glm5.3-flash",
        4,
        MM,
        Some("zai-org/GLM-5.3-Flash"),
        IX,
        CU,
        NI,
        true,
        "PIPELINE ONLY: user explicitly forbids payload acquisition at this stage.",
    ),
    target(
        "glm5.3",
        5,
        MM,
        None,
        U,
        UV,
        NI,
        false,
        "Full model source and geometry must be verified independently of Flash.",
    ),
    target(
        "meta-muse-spark-open",
        6,
        MM,
        Some("meta-models/Muse-Spark-Open-816B-A42B-NVFP4"),
        IX,
        CU,
        NI,
        false,
        "Open-weight Spark only; no implied access to closed Muse Wonder or accepted commercial license.",
    ),
    target(
        "mistral",
        7,
        L,
        None,
        U,
        UV,
        NI,
        false,
        "Family-level scaffold; use pinned exact release/shape, including MoE variants.",
    ),
    target(
        "llama",
        8,
        MM,
        None,
        U,
        UV,
        NI,
        false,
        "Family-level scaffold; no automatic Meta EULA acceptance or implied support.",
    ),
    target(
        "qwen-image-2.1",
        9,
        IG,
        Some("Qwen/Qwen-Image-2.1"),
        DA,
        VM,
        NI,
        false,
        "Diffusers model_index and encoder/transformer/VAE/scheduler/processor require one complete pinned assembly; streaming image stages not implemented.",
    ),
    target(
        "qwen3-tts-1.7b",
        9,
        TTS,
        Some("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        DA,
        VM,
        NI,
        false,
        "Speech tokenizer/codec is a distinct nested component. Streaming codec-frame commits not implemented.",
    ),
    target(
        "qwen3-asr-1.7b",
        9,
        ASR,
        Some("Qwen/Qwen3-ASR-1.7B"),
        IX,
        VM,
        NI,
        false,
        "ASR frontend/audio codec and partial transcript streaming remain unqualified.",
    ),
    target(
        "whisper-large-v3-turbo",
        9,
        ASR,
        Some("openai/whisper-large-v3-turbo"),
        SS,
        VM,
        NI,
        false,
        "Explicit upstream Whisper variant; latest means re-verify, never float silently.",
    ),
    target(
        "nvidia-canary-qwen-2.5b",
        9,
        ASR,
        Some("nvidia/canary-qwen-2.5b"),
        SS,
        VM,
        NI,
        false,
        "NeMo encoder plus Qwen decoder; safetensors are not complete processor/runtime proof.",
    ),
    target(
        "nvidia-parakeet-tdt-1.1b",
        9,
        ASR,
        Some("nvidia/parakeet-tdt-1.1b"),
        NA,
        VM,
        NI,
        false,
        "Native .nemo archive requires bounded source extraction and RNNT/TDT decoder.",
    ),
    target(
        "speaker-diarization",
        10,
        ASR,
        None,
        U,
        UV,
        NI,
        false,
        "Placeholder speaker segmentation/embedding source; not a pretrained diarization loader.",
    ),
    target(
        "phi",
        10,
        L,
        None,
        U,
        UV,
        NI,
        false,
        "Additional open family; independently qualified dense architecture.",
    ),
    target(
        "kimi",
        10,
        MM,
        None,
        U,
        UV,
        NI,
        false,
        "Additional open family; independent recurrent/expert/codec branches.",
    ),
    target(
        "minimax",
        10,
        L,
        None,
        U,
        UV,
        NI,
        false,
        "Additional open family; no generic MoE execution fallback.",
    ),
    target(
        "deepseek-other",
        10,
        MM,
        None,
        U,
        UV,
        NI,
        false,
        "Other DeepSeek variants require separately qualified architectures.",
    ),
    target(
        "flux-image",
        10,
        IG,
        None,
        DA,
        UV,
        NI,
        false,
        "Additional OSS image family; separate conditioners/solver/VAE.",
    ),
    target(
        "wan-video",
        10,
        MM,
        None,
        DA,
        UV,
        NI,
        false,
        "Additional OSS video family; temporal VAE/transformer/source identity.",
    ),
    target(
        "kokoro-tts",
        10,
        TTS,
        None,
        U,
        UV,
        NI,
        false,
        "Additional OSS nonautoregressive TTS candidate.",
    ),
];

pub fn find(slug: &str) -> Option<&'static ModelTarget> {
    MODEL_TARGETS.iter().find(|target| target.slug == slug)
}

impl ModelTarget {
    /// Guard against treating the priority/source manifest as load readiness.
    pub fn require_source_bound_execution(&self) -> Result<()> {
        Err(Error::Unsupported(format!(
            "model target '{}' execution NOT IMPLEMENTED for this source; inspect the source-specific runtime qualification",
            self.slug
        )))
    }
    /// Real-time image/audio requires independent media clocks, codecs,
    /// cancellation/commit and physical backpressure, never catalog inference.
    pub fn require_streaming_media(&self) -> Result<()> {
        Err(Error::Unsupported(format!(
            "model target '{}' realtime media NOT IMPLEMENTED",
            self.slug
        )))
    }
    pub fn require_payload_permission(&self) -> Result<()> {
        if self.payload_hold {
            return Err(Error::Unsupported(format!(
                "model target '{}' is PIPELINE ONLY; payload acquisition explicitly held",
                self.slug
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn all_priorities_ids_and_holds_are_explicit_unique_and_ordered() {
        let mut names = BTreeSet::new();
        let mut last = 0;
        for target in MODEL_TARGETS {
            assert!(names.insert(target.slug));
            assert!(target.priority >= last);
            last = target.priority;
            assert!(target.slug.len() < 100 && !target.slug.is_empty());
            assert!(target.require_source_bound_execution().is_err());
            assert!(target.require_streaming_media().is_err());
            if target.payload_hold {
                assert!(target.require_payload_permission().is_err());
            }
        }
        for slug in ["deepseek-v4.1-flash", "glm5.3-flash"] {
            assert!(find(slug).unwrap().payload_hold);
        }
        assert_eq!(find("qwen3.5-2b").unwrap().execution_status, QB);
        assert_eq!(find("gemma4-26b-a4b-it").unwrap().execution_status, NI);
        assert!(find("../bogus").is_none());
    }
    #[test]
    fn release_names_are_never_invented_or_implicitly_upgrade_verified_source() {
        for slug in [
            "qwen3.5-plus",
            "glm5.3",
            "mistral",
            "llama",
            "qwen-gguf-quant",
        ] {
            assert_eq!(find(slug).unwrap().source_status, UV);
            assert!(find(slug).unwrap().repository.is_none());
        }
        for slug in ["qwen3.8-27b", "gemma4-e2b-it", "gemma4-26b-a4b-it"] {
            let entry = find(slug).unwrap();
            assert_eq!(entry.source_status, VM);
            assert!(entry.repository.is_some());
        }
        assert_eq!(find("qwen-image-2.1").unwrap().source_layout, DA);
        assert_eq!(find("nvidia-parakeet-tdt-1.1b").unwrap().source_layout, NA);
    }
}
