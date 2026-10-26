use super::*;
macro_rules! unaware_callbacks {
    ($($name:ident($($arg:ident:$kind:ty),*) -> $result:ty;)+) => {$(
        fn $name(&mut self,$($arg:$kind),*) -> $result {
            panic!("unaware backend callback must not run");
        }
    )+};
}
pub(super) struct UnawareBackend;
#[allow(clippy::too_many_arguments)]
impl MathematicalBackend for UnawareBackend {
    type Value = Vec<f32>;
    type Index = usize;
    fn elements(&self, _value: &Self::Value) -> usize {
        panic!("unaware metadata callback");
    }
    unaware_callbacks! {
        embed(_tensor:SourceTensor,_token:usize)->Result<Self::Value>;
        project(_tensor:SourceTensor,_input:&Self::Value)->Result<Self::Value>;
        view(_input:&Self::Value,_view:TensorView)->Result<Self::Value>;
        contiguous(_input:&Self::Value)->Result<Self::Value>;
        rms(_name:&str,_input:&Self::Value,_gamma:SourceVector<'_>,_eps:f32)->Result<Self::Value>;
        grouped_rms(_name:&str,_input:&Self::Value,_gamma:SourceVector<'_>,_groups:usize,_eps:f32)->Result<Self::Value>;
        silu(_name:&str,_input:&Self::Value,_multiplier:Option<&Self::Value>)->Result<Self::Value>;
        add(_name:&str,_left:&Self::Value,_right:&Self::Value)->Result<Self::Value>;
        sigmoid_mul(_name:&str,_value:&Self::Value,_gate:&Self::Value)->Result<Self::Value>;
        conv_step(_name:&str,_domain:ConvDomain,_input:&Self::Value,_weights:SourceVector<'_>)->Result<Self::Value>;
        coefficients(_name:&str,_alpha:&Self::Value,_beta:&Self::Value,_dt:SourceVector<'_>,_a:SourceVector<'_>,_scale:f32)->Result<CoefficientValues<Self::Value>>;
        recurrent_step(_name:&str,_domain:RecurrentDomain,_q:&Self::Value,_k:&Self::Value,_v:&Self::Value,_coefficients:&Self::Value,_width:usize)->Result<Self::Value>;
        validate_kv_prefix(_domain:KvDomain,_position:usize)->Result<()>;
        rope(_name:&str,_q:&Self::Value,_k:&Self::Value,_v:&Self::Value,_position:usize,_geometry:RopeGeometry)->Result<RopeValues<Self::Value>>;
        append_kv(_domain:KvDomain,_k:&Self::Value,_v:&Self::Value)->Result<()>;
        attention_qk(_name:&str,_domain:KvDomain,_query:&Self::Value,_valid_tokens:usize,_geometry:AttentionGeometry,_scale:f32)->Result<QkValues<Self::Value>>;
        attention_softmax(_name:&str,_scores:&Self::Value,_heads:usize,_valid_tokens:usize)->Result<Self::Value>;
        attention_value(_name:&str,_domain:KvDomain,_probabilities:&Self::Value,_valid_tokens:usize,_geometry:AttentionGeometry)->Result<Self::Value>;
        assert_finite(_layer:usize,_value:&Self::Value)->Result<()>;
        observe(_layer:usize,_stage:&'static str,_value:&Self::Value)->Result<()>;
        observe_state(_layer:usize,_stage:&'static str,_domain:StateSlot)->Result<()>;
        advance_frontier()->Result<()>;
    }
}

impl UnawareBackend {
    pub(super) fn source_operator(
        &mut self,
        descriptor: &QwenOperator,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        MathematicalBackend::source_operator(self, &descriptor.mathematical_operator()?, inputs)
    }
    pub(super) fn source_embedding(&mut self, descriptor: &QwenEmbedding) -> Result<Vec<f32>> {
        let call = descriptor.teachers().source_call(SourceCallMode::Prefill)?;
        let operator = SourceOperator::new(
            call,
            SourceOperatorKind::EmbeddingRows,
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            vec![descriptor.tensor().clone()],
            vec![],
        )?;
        let values = MathematicalBackend::source_operator(self, &operator, &[])?;
        let [value] = values
            .try_into()
            .map_err(|_| Error::State("test source embedding output arity".into()))?;
        Ok(value)
    }
}
