//! Actual final gamma/header owner; model roles/config remain the cold factory.
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceGamma, SourceNumericalContract, SourceParameter, SourceRmsGeometry,
};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};
struct FinalRmsOwner {
    source: VerifiedTensorSource,
    parameter: SourceParameter,
    geometry: SourceRmsGeometry,
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
            _metadata: metadata,
        })))
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
