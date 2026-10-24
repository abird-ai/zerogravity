//! Explicit diagnostic source-window session. Production HF execution refuses.
use super::*;
mod initial;
use crate::accelerator::program::{ActivationPrecision, StateSlot, TorchCpuGdnContract};
use crate::host_math::{HostMath, HostValue};
pub use initial::InitialCoreStage;
pub(crate) use initial::InitialCoreTape;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::ParameterWindow;
use zerogravity_core::session_root::{RootBinding, SessionAuthority};
#[derive(Clone)]
pub(crate) enum DiagnosticLayerState<V = HostValue> {
    Gdn {
        history: V,
        full_history: Option<V>,
        state: V,
    },
    Attention {
        keys: Option<V>,
        values: Option<V>,
    },
}
pub struct OriginalWindowDiagnosticSession {
    source: Arc<OriginalWindowOperators>,
    layers: Vec<DiagnosticLayerState>,
    authority: SessionAuthority,
    capacity: usize,
    _metadata: WindowWorkspace,
}
pub struct WindowDiagnosticOutput {
    logits: HostValue,
    binding: RootBinding,
}
pub struct WindowDiagnosticReadback<'a> {
    logits: &'a HostValue,
    layers: &'a [DiagnosticLayerState],
    base: RootBinding,
    prior: &'a [DiagnosticLayerState],
    boundaries: &'a [(WindowDiagnosticBoundary, HostValue)],
    initial_core: Option<&'a InitialCoreTape>,
    prefix: bool,
}
pub(crate) struct WindowDiagnosticTrace<'a, V = HostValue> {
    pub(crate) boundaries: &'a mut Vec<(WindowDiagnosticBoundary, V)>,
    pub(crate) layer9: bool,
    pub(crate) prefix: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowDiagnosticBoundary {
    Embedding,
    Layer(usize),
    FinalNorm,
    Layer9(WindowDiagnosticLayer9Stage),
}
/// Borrowed producers from the same genuine layer9; no alternate evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowDiagnosticLayer9Stage {
    InputNorm,
    Qkv,
    Z,
    B,
    A,
    Core,
    NormGate,
    Gated,
    GdnOut,
    PostAttention,
    PostNorm,
    FfnGate,
    FfnUp,
    FfnGated,
    FfnDown,
}
impl WindowDiagnosticReadback<'_> {
    pub fn logits(&self) -> Result<&[f32]> {
        if self.prefix {
            return Err(Error::State(
                "prefix has no final RMS or full vocabulary".into(),
            ));
        }
        self.logits.as_slice()
    }
    pub fn completed_prefix_layers(&self) -> Option<usize> {
        if self.prefix { Some(10) } else { None }
    }
    pub fn initial_core(&self, stage: InitialCoreStage) -> Result<&[f32]> {
        self.initial_core
            .ok_or_else(|| Error::State("initial core not observed".into()))?
            .values(stage)
    }
    pub fn base(&self) -> RootBinding {
        self.base
    }
    /// Conv is `[channel,3]` by default; the explicit full-convolution
    /// constructor retains authentic `[channel,4]` state.
    /// Recurrent is original key-major; KV components are token-major K/V.
    pub fn state(&self, slot: StateSlot) -> Result<Vec<&[f32]>> {
        if self.prefix {
            let layer = match slot {
                StateSlot::Conv(crate::accelerator::program::ConvDomain(i))
                | StateSlot::Recurrent(crate::accelerator::program::RecurrentDomain(i))
                | StateSlot::Kv(crate::accelerator::program::KvDomain(i)) => i,
            };
            if layer > 9 {
                return Err(Error::State("unexecuted prefix state domain".into()));
            }
        }
        read_state(self.layers, slot)
    }
    pub fn prior_state(&self, slot: StateSlot) -> Result<Vec<&[f32]>> {
        if self.prefix {
            let layer = match slot {
                StateSlot::Conv(crate::accelerator::program::ConvDomain(i))
                | StateSlot::Recurrent(crate::accelerator::program::RecurrentDomain(i))
                | StateSlot::Kv(crate::accelerator::program::KvDomain(i)) => i,
            };
            if layer > 9 {
                return Err(Error::State("unexecuted prefix prior domain".into()));
            }
        }
        read_state(self.prior, slot)
    }
    /// Borrowed original computed values; tracing never imports expected data.
    pub fn boundary(&self, boundary: WindowDiagnosticBoundary) -> Result<&[f32]> {
        self.boundaries
            .iter()
            .find(|(b, _)| *b == boundary)
            .ok_or_else(|| Error::State("diagnostic boundary absent/not traced".into()))?
            .1
            .as_slice()
    }
}
impl WindowDiagnosticOutput {
    pub fn logits(&self) -> Result<&[f32]> {
        self.logits.as_slice()
    }
    pub fn binding(&self) -> RootBinding {
        self.binding
    }
    pub fn trained_numerical_qualified(&self) -> bool {
        false
    }
}
impl OriginalWindowOperators {
    pub(crate) fn gdn_contract(&self, layer: usize) -> Result<TorchCpuGdnContract> {
        if self.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention) {
            return Err(Error::Unsupported("diagnostic GDN source layer".into()));
        }
        let g = self.binding.gdn_geometry();
        if g.key_width != 128 || g.value_width != 128 || g.convolution_kernel != 4 {
            return Err(Error::Unsupported(
                "diagnostic GDN128/four-tap source geometry".into(),
            ));
        }
        let roles = crate::hybrid_decoder::ParameterGroup::GdnAll.roles(layer);
        let [convolution, a_log, dt_bias, norm]: [crate::accelerator::program::SourceParameter; 4] =
            roles
                .into_iter()
                .map(|role| self.parameter_descriptor(role))
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| Error::State("family GDN parameter group arity".into()))?;
        let c = TorchCpuGdnContract {
            standalone_exp:
                crate::accelerator::program::StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1,
            convolution,
            a_log,
            dt_bias,
            norm,
            key_heads: g.key_heads,
            value_heads: g.value_heads,
        };
        let (channels, value, _) = c.validate(self.source_identity())?;
        for (role, rows) in [
            (QwenMatrixRole::GdnQkv(layer), channels),
            (QwenMatrixRole::GdnGate(layer), value),
            (QwenMatrixRole::GdnAlpha(layer), g.value_heads),
            (QwenMatrixRole::GdnBeta(layer), g.value_heads),
        ] {
            let d = self.projection_role(role)?;
            if d.rows != rows || d.columns as u64 != self.geometry().hidden_size {
                return Err(Error::State("diagnostic GDN actual projection role".into()));
            }
        }
        Ok(c)
    }
    pub(crate) fn require_gdn_contract(&self, c: &TorchCpuGdnContract) -> Result<()> {
        for (i, k) in self.geometry().layers.iter().enumerate() {
            if *k == crate::model::LayerKind::LinearAttention {
                let a = self.gdn_contract(i)?;
                if a.standalone_exp == c.standalone_exp
                    && a.convolution == c.convolution
                    && a.a_log == c.a_log
                    && a.dt_bias == c.dt_bias
                    && a.norm == c.norm
                    && a.key_heads == c.key_heads
                    && a.value_heads == c.value_heads
                {
                    return Ok(());
                }
            }
        }
        Err(Error::State(
            "diagnostic GDN contract differs from one authentic layer".into(),
        ))
    }
    pub(crate) fn gdn_parameter_windows(
        &self,
        c: &TorchCpuGdnContract,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<ParameterWindow>> {
        self.require_gdn_contract(c)?;
        [&c.convolution, &c.a_log, &c.dt_bias, &c.norm]
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                check_cancel(cancel)?;
                let raw = if i == 0 {
                    self.source().conv4_parameter(&p.name)?
                } else {
                    self.source().parameter(&p.name)?
                };
                raw.validate_source_parameter(p)?;
                let w = match cancel {
                    Some(f) => {
                        self.source()
                            .read_parameter_cancellable(&raw, 0..raw.elements(), f)?
                    }
                    None => self.source().read_parameter(&raw, 0..raw.elements())?,
                };
                if !w.is_complete() {
                    return Err(Error::State(
                        "diagnostic GDN incomplete source parameter".into(),
                    ));
                }
                Ok(w)
            })
            .collect()
    }
    /// Initial-one/cached-one diagnostic only; never an initial64 retag.
    pub fn diagnostic_session(
        self: &Arc<Self>,
        capacity: usize,
        cancel: Option<&AtomicBool>,
    ) -> Result<OriginalWindowDiagnosticSession> {
        self.diagnostic_session_with_convolution(capacity, false, cancel)
    }
    /// Authentic full HF [channel,4] state; same last3 arithmetic input.
    /// Cached MKL-HA exp and trained numerical qualification remain open.
    pub fn diagnostic_full_convolution_session(
        self: &Arc<Self>,
        capacity: usize,
        cancel: Option<&AtomicBool>,
    ) -> Result<OriginalWindowDiagnosticSession> {
        self.diagnostic_session_with_convolution(capacity, true, cancel)
    }
    fn diagnostic_session_with_convolution(
        self: &Arc<Self>,
        capacity: usize,
        full_convolution: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<OriginalWindowDiagnosticSession> {
        check_cancel(cancel)?;
        if capacity == 0
            || capacity > 64
            || capacity as u64 > self.geometry().max_position_embeddings
        {
            return Err(Error::Limit(
                "diagnostic source-window capacity1..64".into(),
            ));
        }
        let metadata = self.source().reserve_workspace(256 * 1024)?;
        let math = HostMath::for_window_operator_slice(self, cancel, None);
        let mut layers = Vec::with_capacity(self.geometry().layers.len());
        for (i, kind) in self.geometry().layers.iter().enumerate() {
            check_cancel(cancel)?;
            layers.push(match kind {
                crate::model::LayerKind::LinearAttention => {
                    let (c, _, s) = self.gdn_contract(i)?.validate(self.source_identity())?;
                    DiagnosticLayerState::Gdn {
                        history: math.window_zeros(c * 3, ActivationPrecision::Bf16WidenedF32)?,
                        full_history: full_convolution
                            .then(|| math.window_zeros(c * 4, ActivationPrecision::Bf16WidenedF32))
                            .transpose()?,
                        state: math.window_zeros(s, ActivationPrecision::F32)?,
                    }
                }
                crate::model::LayerKind::FullAttention => {
                    crate::operator_slice::attention::attention_roles(self.as_ref(), i)?;
                    DiagnosticLayerState::Attention {
                        keys: None,
                        values: None,
                    }
                }
                _ => return Err(Error::Unsupported("diagnostic sliding state".into())),
            });
        }
        check_cancel(cancel)?;
        Ok(OriginalWindowDiagnosticSession {
            source: self.clone(),
            layers,
            authority: SessionAuthority::new(self.source_identity())?,
            capacity,
            _metadata: metadata,
        })
    }
}
impl OriginalWindowDiagnosticSession {
    pub fn binding(&self) -> RootBinding {
        self.authority.binding()
    }
    pub fn domain_count(&self) -> usize {
        self.layers
            .iter()
            .map(|s| {
                if matches!(s, DiagnosticLayerState::Gdn { .. }) {
                    2
                } else {
                    1
                }
            })
            .sum()
    }
    pub fn trained_numerical_qualified(&self) -> bool {
        false
    }
    pub fn step(
        &mut self,
        token: u32,
        cancel: Option<&AtomicBool>,
    ) -> Result<WindowDiagnosticOutput> {
        self.step_with(token, cancel, |_| Ok(()))
    }
    /// Validation runs only after synchronous math, before original root commit.
    pub fn step_with(
        &mut self,
        token: u32,
        cancel: Option<&AtomicBool>,
        validate: impl FnOnce(&WindowDiagnosticReadback<'_>) -> Result<()>,
    ) -> Result<WindowDiagnosticOutput> {
        self.step_inner(token, cancel, None, false, validate)
    }
    /// Retain only original computed boundaries until completed validation.
    /// Existing untraced callers retain neither these buffers nor their charge.
    pub fn step_traced_with(
        &mut self,
        token: u32,
        cancel: Option<&AtomicBool>,
        validate: impl FnOnce(&WindowDiagnosticReadback<'_>) -> Result<()>,
    ) -> Result<WindowDiagnosticOutput> {
        self.step_inner(token, cancel, Some(false), false, validate)
    }
    /// Full original traversal with fifteen additional layer9 producers.
    /// Expected-only diagnostics; actual source/geometry and normal publication
    /// rules remain identical to the uninstrumented diagnostic proposal.
    pub fn step_layer9_traced_with(
        &mut self,
        token: u32,
        cancel: Option<&AtomicBool>,
        validate: impl FnOnce(&WindowDiagnosticReadback<'_>) -> Result<()>,
    ) -> Result<WindowDiagnosticOutput> {
        check_cancel(cancel)?;
        if self.layers.len() != 64
            || self.source.geometry().hidden_size != 5120
            || self.source.gdn_contract(9)?.value_heads != 48
            || self.source.gdn_contract(9)?.key_heads != 16
        {
            return Err(Error::Unsupported(
                "layer9 genuine64/H48 diagnostic source".into(),
            ));
        }
        self.step_inner(token, cancel, Some(true), false, validate)
    }
    /// Unpublished original-source prefix. Reuses the same iterator and math;
    /// no final RMS/Head/full-forward/112-domain commit is attempted.
    pub fn observe_initial_core_with(
        &mut self,
        token: u32,
        cancel: Option<&AtomicBool>,
        validate: impl FnOnce(&WindowDiagnosticReadback<'_>) -> Result<()>,
    ) -> Result<()> {
        check_cancel(cancel)?;
        if self.binding().frontier() != 0
            || self.layers.len() != 64
            || self.source.geometry().hidden_size != 5120
            || self.source.gdn_contract(9)?.value_heads != 48
            || self.source.gdn_contract(9)?.key_heads != 16
        {
            return Err(Error::Unsupported(
                "genuine original initial64-object prefix/H48 source".into(),
            ));
        }
        self.step_inner(token, cancel, Some(true), true, validate)
            .map(drop)
    }
    fn step_inner(
        &mut self,
        token: u32,
        cancel: Option<&AtomicBool>,
        trace_layer9: Option<bool>,
        prefix: bool,
        validate: impl FnOnce(&WindowDiagnosticReadback<'_>) -> Result<()>,
    ) -> Result<WindowDiagnosticOutput> {
        check_cancel(cancel)?;
        if u64::from(token) >= self.source.geometry().vocab_size
            || self.binding().frontier() >= self.capacity as u64
        {
            return Err(Error::Limit("diagnostic actual token/capacity".into()));
        }
        let attempt = self.authority.begin_advance(1)?;
        let count = self
            .layers
            .len()
            .checked_add(2 + usize::from(trace_layer9 == Some(true)) * 15)
            .ok_or_else(|| Error::Limit("diagnostic trace count".into()))?;
        let count = if prefix { 26 } else { count };
        let _trace_metadata = if trace_layer9.is_some() {
            let bytes = count
                .checked_mul(std::mem::size_of::<(WindowDiagnosticBoundary, HostValue)>())
                .ok_or_else(|| Error::Limit("diagnostic trace metadata".into()))?;
            Some(
                self.source
                    .source()
                    .reserve_workspace(bytes.max(64 * 1024))?,
            )
        } else {
            None
        };
        let mut boundaries = Vec::new();
        if trace_layer9.is_some() {
            boundaries
                .try_reserve_exact(count)
                .map_err(|_| Error::Limit("diagnostic trace allocation".into()))?;
        }
        let mut proposed = self.layers.clone();
        let mut math = HostMath::for_window_operator_slice(&self.source, cancel, None);
        let inner = if prefix {
            Some(InitialCoreTape::prepare(&self.source)?)
        } else {
            None
        };
        math.initial_core_tape = inner.clone();
        let logits = crate::forward::window_diagnostic_token(
            &self.source,
            &mut math,
            &mut proposed,
            token,
            self.binding().frontier() as usize,
            trace_layer9.map(|layer9| WindowDiagnosticTrace {
                boundaries: &mut boundaries,
                layer9,
                prefix,
            }),
        )?;
        if trace_layer9.is_some() && boundaries.len() != count {
            return Err(Error::State(
                "complete original diagnostic boundary cohort".into(),
            ));
        }
        drop(math);
        let inner = inner
            .as_ref()
            .map(|t| {
                t.lock()
                    .map_err(|_| Error::State("initial core tape poisoned".into()))
            })
            .transpose()?;
        if let Some(t) = inner.as_ref() {
            t.complete()?;
        }
        validate(&WindowDiagnosticReadback {
            logits: &logits,
            layers: &proposed,
            base: self.binding(),
            prior: &self.layers,
            boundaries: &boundaries,
            initial_core: inner.as_deref(),
            prefix,
        })?;
        check_cancel(cancel)?;
        if prefix {
            return Ok(WindowDiagnosticOutput {
                logits,
                binding: self.binding(),
            });
        }
        let publication = self.authority.prepare_publish(attempt)?;
        self.layers = proposed;
        publication.commit();
        Ok(WindowDiagnosticOutput {
            logits,
            binding: self.binding(),
        })
    }
    /// Minimal `[channel,3]` by default; the explicit full-convolution
    /// constructor retains all four original frames, including the oldest.
    /// Neither state representation grants trained numerical qualification.
    /// Recurrence is key-major FP32; KV is two token-major components.
    pub fn state(&self, slot: StateSlot) -> Result<Vec<&[f32]>> {
        read_state(&self.layers, slot)
    }
    /// Original cache schema, independently retained in authenticated config.
    /// This metadata is not a claim that all four cache frames were retained.
    pub fn original_convolution_cache_frames(&self) -> usize {
        self.source.binding.gdn_geometry().convolution_kernel
    }
}
fn read_state(layers: &[DiagnosticLayerState], slot: StateSlot) -> Result<Vec<&[f32]>> {
    use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain};
    match slot {
        StateSlot::Conv(ConvDomain(i)) => match layers.get(i) {
            Some(DiagnosticLayerState::Gdn {
                history,
                full_history,
                ..
            }) => Ok(vec![full_history.as_ref().unwrap_or(history).as_slice()?]),
            _ => Err(Error::State("diagnostic Conv domain".into())),
        },
        StateSlot::Recurrent(RecurrentDomain(i)) => match layers.get(i) {
            Some(DiagnosticLayerState::Gdn { state, .. }) => Ok(vec![state.as_slice()?]),
            _ => Err(Error::State("diagnostic recurrent domain".into())),
        },
        StateSlot::Kv(KvDomain(i)) => match layers.get(i) {
            Some(DiagnosticLayerState::Attention { keys, values }) => match (keys, values) {
                (None, None) => Ok(vec![&[], &[]]),
                (Some(k), Some(v)) => Ok(vec![k.as_slice()?, v.as_slice()?]),
                _ => Err(Error::State("diagnostic incomplete KV pair".into())),
            },
            _ => Err(Error::State("diagnostic KV domain".into())),
        },
    }
}
