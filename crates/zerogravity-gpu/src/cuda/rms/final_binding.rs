//! Actual final gamma/header owner; model roles/config remain the cold factory.
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceGamma, SourceNumericalContract, SourceOperator, SourceOperatorKind, SourceParameter,
    SourceRmsGeometry,
};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};
struct FinalRmsOwner {
    source: VerifiedTensorSource,
    parameter: SourceParameter,
    geometry: SourceRmsGeometry,
    _bindings: Option<Arc<crate::resident_program::source_bindings::SourceBindings>>,
    _metadata: Arc<WindowWorkspace>,
}
#[derive(Clone)]
pub struct BoundFinalRmsSource(Arc<FinalRmsOwner>);
impl BoundFinalRmsSource {
    pub fn bind(
        source: VerifiedTensorSource,
        parameter: SourceParameter,
        geometry: SourceRmsGeometry,
        contract: SourceNumericalContract,
        metadata: Arc<WindowWorkspace>,
    ) -> Result<Self> {
        if !matches!(geometry.width, 1024 | 2048)
            || geometry.epsilon.to_bits() != 1e-6f32.to_bits()
            || geometry.gamma != SourceGamma::OnePlus
            || contract != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || parameter.source != source.source_identity()
            || parameter.kind != QuantKind::Bf16
            || parameter.shape != [geometry.width as u64]
            || parameter.bytes != geometry.width * 2
        {
            return Err(Error::Unsupported(
                "final RMS actual native geometry/arithmetic/gamma differs".into(),
            ));
        }
        source
            .parameter(&parameter.name)?
            .validate_source_parameter(&parameter)?;
        if !metadata.source().same_issuer(&source) || metadata.bytes() < 64 * 1024 {
            return Err(Error::State(
                "final RMS actual metadata source issuer differs".into(),
            ));
        }
        Ok(Self(Arc::new(FinalRmsOwner {
            source,
            parameter,
            geometry,
            _bindings: None,
            _metadata: metadata,
        })))
    }
    pub(crate) fn bind_for_plan(
        bindings: Arc<crate::resident_program::source_bindings::SourceBindings>,
        operator: SourceOperator,
    ) -> Result<Self> {
        let actual = super::BoundModelRmsSource::bind_for_plan(bindings.clone(), operator)?;
        let normalized = bindings
            .selection()
            .normalized()
            .ok_or_else(|| Error::State("final RMS normalized selector absent".into()))?;
        let producer = bindings
            .plan()
            .source_value_provenance(normalized)?
            .ok_or_else(|| Error::State("final RMS actual normalized producer absent".into()))?;
        if !matches!(actual.operator().kind(), SourceOperatorKind::ResidualRms(_))
            || producer.output_index() != 0
            || bindings.operator(producer.node_index())? != Some(actual.operator())
        {
            return Err(Error::State(
                "final RMS actual selected producer differs".into(),
            ));
        }
        let metadata = Arc::new(bindings.source().reserve_workspace(64 * 1024)?);
        let mut result = Self::bind(
            bindings.source().clone(),
            actual.parameter().clone(),
            actual.geometry(),
            actual.operator().numerical_contract(),
            metadata,
        )?;
        Arc::get_mut(&mut result.0)
            .ok_or_else(|| Error::State("fresh final RMS owner".into()))?
            ._bindings = Some(bindings);
        Ok(result)
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.0.source
    }
    pub(crate) fn gamma_name(&self) -> &str {
        &self.0.parameter.name
    }
    pub(crate) fn geometry(&self) -> SourceRmsGeometry {
        self.0.geometry
    }
    pub(crate) fn parameter(&self) -> &SourceParameter {
        &self.0.parameter
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !source.same_issuer(self.source()) {
            return Err(Error::State(
                "final RMS original source issuer differs".into(),
            ));
        }
        source
            .parameter(self.gamma_name())?
            .validate_source_parameter(&self.0.parameter)
    }
}
