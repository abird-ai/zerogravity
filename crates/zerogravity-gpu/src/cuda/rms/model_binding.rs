//! Actual immutable RMS mathematical/source binding; no model roles or recipe.
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceGamma, SourceNumericalContract, SourceOperator, SourceOperatorKind, SourceParameter,
    SourceRmsGeometry,
};
use zerogravity_core::{Error, Result};

struct ModelRmsOwner {
    operator: SourceOperator,
    source: VerifiedTensorSource,
    geometry: SourceRmsGeometry,
    _bindings: Option<Arc<crate::resident_program::source_bindings::SourceBindings>>,
    _metadata: WindowWorkspace,
}
#[derive(Clone)]
pub(crate) struct BoundModelRmsSource(Arc<ModelRmsOwner>);
impl BoundModelRmsSource {
    /// Model/app checks source roles and finite eligibility before this boundary.
    pub(crate) fn bind(source: VerifiedTensorSource, operator: SourceOperator) -> Result<Self> {
        operator.validate()?;
        let geometry = match operator.kind() {
            SourceOperatorKind::Rms(geometry) | SourceOperatorKind::ResidualRms(geometry) => {
                geometry
            }
            _ => {
                return Err(Error::Unsupported(
                    "source RMS mathematical binding only".into(),
                ));
            }
        };
        if !matches!(geometry.width, 1024 | 2048)
            || geometry.epsilon != 1e-6
            || geometry.gamma != SourceGamma::OnePlus
            || operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || operator.call().source() != source.source_identity()
            || operator.parameters().len() != 1
            || !operator.matrices().is_empty()
        {
            return Err(Error::Unsupported(
                "source RMS supported native arithmetic/source".into(),
            ));
        }
        let metadata = source.reserve_workspace(64 * 1024)?;
        source
            .parameter(&operator.parameters()[0].name)?
            .validate_source_parameter(&operator.parameters()[0])?;
        Ok(Self(Arc::new(ModelRmsOwner {
            operator,
            source,
            geometry,
            _bindings: None,
            _metadata: metadata,
        })))
    }
    /// Retain the same admitted source/graph owner before private uploads.
    pub(crate) fn bind_for_plan(
        bindings: Arc<crate::resident_program::source_bindings::SourceBindings>,
        operator: SourceOperator,
    ) -> Result<Self> {
        if operator.call()!=bindings.call()
            || !bindings.plan().nodes().iter().any(|node| {
                matches!(&node.operation,zerogravity_core::accelerator::program::PlanOperation::SourceOperator{descriptor,..}
                    if descriptor.as_ref()==&operator)
            }) {
            return Err(Error::State("RMS actual prepared graph/operator issuer differs".into()));
        }
        let mut result = Self::bind(bindings.source().clone(), operator)?;
        Arc::get_mut(&mut result.0)
            .ok_or_else(|| Error::State("fresh RMS owner".into()))?
            ._bindings = Some(bindings);
        Ok(result)
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.0.source
    }
    pub(crate) fn parameter(&self) -> &SourceParameter {
        &self.0.operator.parameters()[0]
    }
    pub(crate) fn gamma_name(&self) -> &str {
        &self.parameter().name
    }
    pub(crate) fn operator(&self) -> &SourceOperator {
        &self.0.operator
    }
    pub(crate) fn geometry(&self) -> SourceRmsGeometry {
        self.0.geometry
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !self.source().same_issuer(source)
            || source.source_identity() != self.operator().call().source()
        {
            return Err(Error::State(
                "RMS actual immutable source issuer differs".into(),
            ));
        }
        // Header checked at bind; the immutable issuer retains that proof.
        Ok(())
    }
}
