//! Actual RMS operator/source join; family config/roles are checked by its factory.
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::{
    SourceGamma, SourceNumericalContract, SourceOperator, SourceOperatorKind, SourceParameter,
    SourceRmsGeometry,
};
use zerogravity_core::{Error, Result};

#[derive(Clone)]
pub(crate) struct BoundRmsSource {
    source: VerifiedTensorSource,
    operator: SourceOperator,
}
impl BoundRmsSource {
    pub(crate) fn bind(source: VerifiedTensorSource, operator: SourceOperator) -> Result<Self> {
        operator.validate()?;
        let geometry = match operator.kind() {
            SourceOperatorKind::Rms(geometry) | SourceOperatorKind::ResidualRms(geometry) => {
                geometry
            }
            _ => {
                return Err(Error::Unsupported(
                    "source RMS mathematical operator only".into(),
                ));
            }
        };
        if geometry
            != (SourceRmsGeometry {
                width: 1024,
                epsilon: 1e-6,
                gamma: SourceGamma::OnePlus,
            })
            || operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || operator.call().source() != source.source_identity()
            || operator.parameters().len() != 1
        {
            return Err(Error::Unsupported(
                "source RMS exact native arithmetic/source".into(),
            ));
        }
        source
            .parameter(&operator.parameters()[0].name)?
            .validate_source_parameter(&operator.parameters()[0])?;
        Ok(Self { source, operator })
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn operator(&self) -> &SourceOperator {
        &self.operator
    }
    pub(crate) fn parameter(&self) -> &SourceParameter {
        &self.operator.parameters()[0]
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !self.source.same_issuer(source)
            || self.operator.call().source() != source.source_identity()
        {
            return Err(Error::State(
                "RMS actual immutable source issuer differs".into(),
            ));
        }
        // Header checked at bind; the immutable issuer retains that proof.
        Ok(())
    }
}
