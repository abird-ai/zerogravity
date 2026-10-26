//! Original state layout metadata; no payload, transport, codec or publication.

use super::OriginalGdnGeometry;
use crate::{Error, LoadedQwenModel, Result, StateSpec};
use zerogravity_core::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
use zerogravity_core::session_root::RootBinding;

/// Derived from an existing committed binding, never a proposed frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OriginalStatePhase {
    InitialToken,
    CachedToken,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OriginalStateEncoding {
    Bf16,
    /// Exact BF16 values represented in the existing canonical f32 storage.
    Bf16WidenedF32,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OriginalStateLayout {
    /// Per-channel frames ordered oldest→newest, not reversed coefficients.
    ConvolutionChannelFrame,
    RecurrentHeadKeyValue,
    RecurrentHeadValueKey,
    KvHeadTokenWidth,
    KvTokenHeadWidth,
}

/// One tensor layout; KV has two separate components (keys, then values).
/// A singleton original batch axis is omitted; no bytes are allocated here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OriginalStateView {
    layout: OriginalStateLayout,
    encoding: OriginalStateEncoding,
    shape: Vec<usize>,
    components: usize,
}
impl OriginalStateView {
    fn new(
        layout: OriginalStateLayout,
        encoding: OriginalStateEncoding,
        shape: Vec<usize>,
        components: usize,
    ) -> Result<Self> {
        let result = Self {
            layout,
            encoding,
            shape,
            components,
        };
        result.bytes()?;
        Ok(result)
    }
    pub fn layout(&self) -> OriginalStateLayout {
        self.layout
    }
    pub fn encoding(&self) -> OriginalStateEncoding {
        self.encoding
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn components(&self) -> usize {
        self.components
    }
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(self.components, |n, &axis| {
            n.checked_mul(axis)
                .ok_or_else(|| Error::Limit("original state metadata geometry overflow".into()))
        })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.elements()?
            .checked_mul(match self.encoding {
                OriginalStateEncoding::Bf16 => 2,
                OriginalStateEncoding::Bf16WidenedF32 | OriginalStateEncoding::F32 => 4,
            })
            .ok_or_else(|| Error::Limit("original state metadata bytes overflow".into()))
    }
}

/// Binds one existing logical domain to its original/canonical layout metadata.
/// No state owner, numerical profile permit or portable wire schema is issued.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OriginalStateDescriptor {
    binding: RootBinding,
    slot: StateSlot,
    phase: OriginalStatePhase,
    original_cache: Option<OriginalStateView>,
    canonical: OriginalStateView,
}
impl OriginalStateDescriptor {
    pub fn binding(&self) -> RootBinding {
        self.binding
    }
    pub fn slot(&self) -> StateSlot {
        self.slot
    }
    pub fn phase(&self) -> OriginalStatePhase {
        self.phase
    }
    /// Initial source execution has no incoming original cache/state tensor.
    pub fn original_cache(&self) -> Option<&OriginalStateView> {
        self.original_cache.as_ref()
    }
    pub fn canonical(&self) -> &OriginalStateView {
        &self.canonical
    }
}

fn channels(g: OriginalGdnGeometry) -> Result<usize> {
    g.key_heads
        .checked_mul(g.key_width)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| {
            g.value_heads
                .checked_mul(g.value_width)
                .and_then(|value| n.checked_add(value))
        })
        .ok_or_else(|| Error::Limit("original state channel geometry overflow".into()))
}

impl LoadedQwenModel {
    /// Inspect a previously published root's geometry only. The caller remains
    /// responsible for its authority/currentness; historical bindings are valid
    /// metadata inputs. This method never creates or publishes a logical root.
    pub fn original_state_descriptors(
        &self,
        binding: RootBinding,
    ) -> Result<Vec<OriginalStateDescriptor>> {
        use OriginalStateEncoding as E;
        use OriginalStateLayout as L;
        let original = self.original_binding().ok_or_else(|| {
            Error::Unsupported("original HF state metadata requires HF binding".into())
        })?;
        if binding.source() != self.source_identity() {
            return Err(Error::State(
                "original state metadata has foreign source root".into(),
            ));
        }
        if binding.frontier() > self.geometry.max_position_embeddings {
            return Err(Error::Limit(
                "original state metadata frontier exceeds source context".into(),
            ));
        }
        let tokens = usize::try_from(binding.frontier())
            .map_err(|_| Error::Limit("original state frontier exceeds address space".into()))?;
        let phase = if tokens == 0 {
            OriginalStatePhase::InitialToken
        } else {
            OriginalStatePhase::CachedToken
        };
        let mut descriptors = Vec::new();
        let mut append = |slot, raw: OriginalStateView, canonical: OriginalStateView| {
            descriptors.push(OriginalStateDescriptor {
                binding,
                slot,
                phase,
                original_cache: (phase == OriginalStatePhase::CachedToken).then_some(raw),
                canonical,
            });
        };
        let g = original.gdn_geometry();
        for (layer, spec) in self.state_specs.iter().enumerate() {
            match *spec {
                StateSpec::Delta { conv, state } => {
                    let channels = channels(g)?;
                    let history = g.convolution_kernel.checked_sub(1).ok_or_else(|| {
                        Error::Malformed("original convolution requires history geometry".into())
                    })?;
                    let canonical = OriginalStateView::new(
                        L::ConvolutionChannelFrame,
                        E::Bf16WidenedF32,
                        vec![channels, history],
                        1,
                    )?;
                    if canonical.elements()? != conv {
                        return Err(Error::State(
                            "original convolution metadata differs from existing domain".into(),
                        ));
                    }
                    append(
                        StateSlot::Conv(ConvDomain(layer)),
                        OriginalStateView::new(
                            L::ConvolutionChannelFrame,
                            E::Bf16,
                            vec![channels, g.convolution_kernel],
                            1,
                        )?,
                        canonical,
                    );
                    let canonical = OriginalStateView::new(
                        L::RecurrentHeadValueKey,
                        E::F32,
                        vec![g.value_heads, g.value_width, g.key_width],
                        1,
                    )?;
                    if canonical.elements()? != state {
                        return Err(Error::State(
                            "original recurrence metadata differs from existing domain".into(),
                        ));
                    }
                    append(
                        StateSlot::Recurrent(RecurrentDomain(layer)),
                        OriginalStateView::new(
                            L::RecurrentHeadKeyValue,
                            E::F32,
                            vec![g.value_heads, g.key_width, g.value_width],
                            1,
                        )?,
                        canonical,
                    );
                }
                StateSpec::Full { kv_width } => {
                    let heads = usize::try_from(self.geometry.num_key_value_heads)
                        .map_err(|_| Error::Limit("original KV head geometry".into()))?;
                    let width = usize::try_from(self.geometry.head_dim)
                        .map_err(|_| Error::Limit("original KV width geometry".into()))?;
                    if heads.checked_mul(width) != Some(kv_width) {
                        return Err(Error::State(
                            "original KV metadata differs from existing domain".into(),
                        ));
                    }
                    append(
                        StateSlot::Kv(KvDomain(layer)),
                        OriginalStateView::new(
                            L::KvHeadTokenWidth,
                            E::Bf16,
                            vec![heads, tokens, width],
                            2,
                        )?,
                        OriginalStateView::new(
                            L::KvTokenHeadWidth,
                            E::Bf16WidenedF32,
                            vec![tokens, heads, width],
                            2,
                        )?,
                    );
                }
            }
        }
        Ok(descriptors)
    }
}
