//! Source-derived one-token plans over the shared family traversal.
//! Exact CPU-profile arithmetic; no native, root or serving eligibility.
use super::*;
use crate::accelerator::program::{
    ConvDomain, KvDomain, MathematicalBackend, PlanValue, RecurrentDomain, SourceStateBinding,
    StandaloneExpProfile, StateSlot, TensorView, TextCallGeometry, TextStepBindings,
};
use crate::source::binding::{SourceLayerState, SourceOutput};
use std::sync::Arc;

// Former backend wrapper: RecordedWindowToken; bindings now live in core.
fn one(values: Vec<PlanValue>) -> Result<PlanValue> {
    let [value] = values
        .try_into()
        .map_err(|_| Error::State("original single-component state".into()))?;
    Ok(value)
}
impl OriginalWindowOperators {
    /// Record actual source geometry at a requested frontier, without token IDs.
    /// The explicit CPU attention arithmetic currently supports at most64 tokens;
    /// backend kernel coverage and admission remain independent requirements.
    pub fn record_source_token(
        self: &Arc<Self>,
        position: usize,
        capacity: usize,
    ) -> Result<TextStepBindings> {
        if capacity == 0
            || capacity > 64
            || position >= capacity
            || capacity as u64 > self.geometry().max_position_embeddings
        {
            return Err(Error::Unsupported(
                "source token CPU arithmetic context bound".into(),
            ));
        }
        if self.geometry().layers.is_empty() || self.geometry().layers.len() > 64 {
            return Err(Error::Limit(
                "source token prepared metadata layer bound".into(),
            ));
        }
        let metadata = self.source().reserve_workspace(4 * 1024 * 1024)?;
        let required = crate::hybrid_decoder::state_domains(self.geometry())?;
        let schema = crate::hybrid_decoder::state_specs(
            self.geometry(),
            self.binding().gdn_geometry(),
            crate::hybrid_decoder::ConvolutionRetention::FullSourceCache,
        )?;
        let domains = required.len();
        // Explicit arithmetic selection, independent of layer/head/model names.
        let exp = StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1;
        let mut math =
            PlanBuilder::new_torch_source(self.source_identity(), capacity, required, true, exp)?;
        let mut states = Vec::with_capacity(self.geometry().layers.len());
        for (layer, kind) in self.geometry().layers.iter().enumerate() {
            states.push(match kind {
                crate::config::LayerKind::LinearAttention => {
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
                    SourceLayerState::Gdn {
                        history,
                        full_history: Some(full),
                        state,
                    }
                }
                crate::config::LayerKind::FullAttention => {
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
                    SourceLayerState::Attention { keys, values }
                }
                _ => return Err(Error::Unsupported("source token layer kind".into())),
            });
        }
        let output = crate::forward::window_source_token(
            self,
            &mut math,
            &mut states,
            PlanBuilder::inputs().token,
            position,
            SourceOutput {
                layers: self.geometry().layers.len(),
                logits: true,
            },
            (),
        )?;
        for (layer, state) in states.iter().enumerate() {
            match state {
                SourceLayerState::Gdn {
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
                SourceLayerState::Attention { keys, values } => {
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
        if plan.required_domains().len() != domains
            || plan.proposed_domains().len() != domains
            || plan.standalone_exp_profile() != Some(exp)
        {
            return Err(Error::State(
                "source state population/effective arithmetic record".into(),
            ));
        }
        let geometry = TextCallGeometry::new(
            self.source_identity(),
            self.matrix_descriptor(QwenMatrixRole::Embedding)?.rows,
            capacity,
        )?;
        TextStepBindings::new(plan, geometry, position, metadata.into_host_workspace()?)
    }
}
