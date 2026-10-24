//! Same original family traversal recorded for the finite initial1/cached1 unit.
//! No native eligibility, live state, completion or public serving authority.
use super::*;
use crate::accelerator::program::{
    ConvDomain, ForwardPlan, KvDomain, MathematicalBackend, PlanValue, RecurrentDomain,
    SourceStateBinding, StandaloneExpProfile, StateSlot, TensorView,
};
use crate::original_binding::DiagnosticLayerState;
use std::sync::Arc;

pub struct RecordedWindowToken {
    source: Arc<OriginalWindowOperators>,
    plan: ForwardPlan,
    position: usize,
    _metadata: zerogravity_artifacts::source_weights::WindowWorkspace,
}
impl RecordedWindowToken {
    pub fn source(&self) -> &Arc<OriginalWindowOperators> {
        &self.source
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn capacity(&self) -> usize {
        2
    }
    /// Preserves the genuine attempt. Metadata alone grants no publication.
    pub fn bind_call(
        &self,
        attempt: zerogravity_core::session_root::RootAttempt,
        id: u32,
    ) -> Result<crate::accelerator::program::Bf16TextCall> {
        let vocabulary = self
            .source
            .matrix_descriptor(QwenMatrixRole::Embedding)?
            .rows;
        if attempt.base().frontier() != self.position as u64 {
            return Err(Error::State(
                "source record phase/actual root attempt".into(),
            ));
        }
        crate::accelerator::program::Bf16TextCall::bind(
            crate::accelerator::program::TextCallGeometry::new(
                self.source.source_identity(),
                vocabulary,
                self.capacity(),
            )?,
            attempt,
            Arc::from([id]),
        )
    }
}
fn one(values: Vec<PlanValue>) -> Result<PlanValue> {
    let [value] = values
        .try_into()
        .map_err(|_| Error::State("original single-component state".into()))?;
    Ok(value)
}
impl OriginalWindowOperators {
    /// Source-authenticated arithmetic recording only. The backend must still
    /// admit this exact record/source/profile and bind genuine original roots.
    pub fn record_full_source_token(
        self: &Arc<Self>,
        position: usize,
    ) -> Result<RecordedWindowToken> {
        if position > 1 || self.geometry().layers.len() != 64 {
            return Err(Error::Unsupported(
                "finite source INITIAL1/cached1 full64".into(),
            ));
        }
        let metadata = self.source().reserve_workspace(4 * 1024 * 1024)?;
        let required = crate::hybrid_decoder::state_domains(self.geometry())?;
        let schema = crate::hybrid_decoder::state_specs(
            self.geometry(),
            self.binding().gdn_geometry(),
            crate::hybrid_decoder::ConvolutionRetention::FullSourceCache,
        )?;
        if required.len() != 112 {
            return Err(Error::State("source full64 logical112 domains".into()));
        }
        // Explicit arithmetic selection, independent of layer/head/model names.
        let exp = StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1;
        let mut math =
            PlanBuilder::new_torch_source(self.source_identity(), 2, required, true, exp)?;
        let mut states = Vec::with_capacity(64);
        for (layer, kind) in self.geometry().layers.iter().enumerate() {
            states.push(match kind {
                crate::model::LayerKind::LinearAttention => {
                    let contract = self.gdn_contract(layer)?;
                    if contract.standalone_exp != exp {
                        return Err(Error::State(
                            "original GDN arithmetic revision changed".into(),
                        ));
                    }
                    let (channels, _, recurrent) = contract.validate(self.source_identity())?;
                    if !matches!(schema[layer],crate::StateSpec::Delta{conv,state} if conv==channels*4 && state==recurrent) {
                        return Err(Error::State("family full-cache source schema differs from arithmetic contract".into()));
                    }
                    let full = one(math.source_state_read(SourceStateBinding::new(
                        StateSlot::Conv(ConvDomain(layer)),
                        vec![match schema[layer] { crate::StateSpec::Delta{conv,..}=>conv,_=>return Err(Error::State("family convolution state".into())) }],
                    )?)?)?;
                    let tail = math.view(
                        &full,
                        TensorView {
                            offset: 1,
                            rows: channels,
                            width: 3,
                            row_stride: 4,
                        },
                    )?;
                    let history = math.contiguous(&tail)?;
                    let state = one(math.source_state_read(SourceStateBinding::new(
                        StateSlot::Recurrent(RecurrentDomain(layer)),
                        vec![match schema[layer] { crate::StateSpec::Delta{state,..}=>state,_=>return Err(Error::State("family recurrent state".into())) }],
                    )?)?)?;
                    DiagnosticLayerState::Gdn {
                        history,
                        full_history: Some(full),
                        state,
                    }
                }
                crate::model::LayerKind::FullAttention => {
                    let r =
                        crate::operator_slice::attention::attention_roles(self.as_ref(), layer)?;
                    let crate::StateSpec::Full{kv_width}=schema[layer] else { return Err(Error::State("family KV state".into())); };
                    if kv_width != r.kv_heads*r.width {return Err(Error::State("family KV source geometry".into()));}
                    let (keys, values) = if position == 0 {
                        (None, None)
                    } else {
                        let shape = position * kv_width;
                        let [k, v] = math
                            .source_state_read(SourceStateBinding::new(
                                StateSlot::Kv(KvDomain(layer)),
                                vec![shape, shape],
                            )?)?
                            .try_into()
                            .map_err(|_| Error::State("source two-plane KV read".into()))?;
                        (Some(k), Some(v))
                    };
                    DiagnosticLayerState::Attention { keys, values }
                }
                _ => return Err(Error::Unsupported("source full64 layer kind".into())),
            });
        }
        let output = crate::forward::window_source_token(
            self,
            &mut math,
            &mut states,
            PlanBuilder::inputs().token,
            position,
            None,
        )?;
        for (layer, state) in states.iter().enumerate() {
            match state {
                DiagnosticLayerState::Gdn {
                    full_history,
                    state,
                    ..
                } => {
                    let contract = self.gdn_contract(layer)?;
                    let (channels, _, recurrent) = contract.validate(self.source_identity())?;
                    let full = full_history
                        .ok_or_else(|| Error::State("source full4 producer absent".into()))?;
                    math.source_state_write(
                        SourceStateBinding::new(
                            StateSlot::Conv(ConvDomain(layer)),
                            vec![channels * 4],
                        )?,
                        &[full],
                    )?;
                    math.source_state_write(
                        SourceStateBinding::new(
                            StateSlot::Recurrent(RecurrentDomain(layer)),
                            vec![recurrent],
                        )?,
                        &[*state],
                    )?;
                }
                DiagnosticLayerState::Attention { keys, values } => {
                    let r =
                        crate::operator_slice::attention::attention_roles(self.as_ref(), layer)?;
                    let k =
                        keys.ok_or_else(|| Error::State("source final K producer absent".into()))?;
                    let v = values
                        .ok_or_else(|| Error::State("source final V producer absent".into()))?;
                    let shape = (position + 1) * r.kv_heads * r.width;
                    math.source_state_write(
                        SourceStateBinding::new(
                            StateSlot::Kv(KvDomain(layer)),
                            vec![shape, shape],
                        )?,
                        &[k, v],
                    )?;
                }
            }
        }
        math.advance_frontier()?;
        let plan = math.finish(Some(output))?;
        if plan.required_domains().len() != 112
            || plan.proposed_domains().len() != 112
            || plan.standalone_exp_profile() != Some(exp)
        {
            return Err(Error::State(
                "original source full112/effective arithmetic record".into(),
            ));
        }
        Ok(RecordedWindowToken {
            source: self.clone(),
            plan,
            position,
            _metadata: metadata,
        })
    }
}
