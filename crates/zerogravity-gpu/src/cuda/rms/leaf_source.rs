//! Common observed RMS-leaf source/header data; no family or execution permit.
use super::{BoundFinalRmsSource, Columns};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{
    VerifiedMatrix, VerifiedParameter, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::accelerator::program::{
    MatrixAxes, SourceNumericalContract, SourceParameter, SourceRmsGeometry, SourceTensor,
};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};
struct LeafOwner {
    source: VerifiedTensorSource,
    embedding: SourceTensor,
    input_gamma: SourceParameter,
    final_binding: BoundFinalRmsSource,
    retained: Option<Arc<crate::cuda::native::RetainedSource>>,
}
#[derive(Clone)]
pub struct RmsLeafSource(Arc<LeafOwner>);
impl RmsLeafSource {
    pub fn bind(
        source: VerifiedTensorSource,
        embedding: SourceTensor,
        input_gamma: SourceParameter,
        final_gamma: SourceParameter,
        geometry: SourceRmsGeometry,
        contract: SourceNumericalContract,
        metadata: Arc<WindowWorkspace>,
    ) -> Result<Self> {
        if embedding.source != source.source_identity()
            || embedding.kind != QuantKind::Bf16
            || embedding.axes != MatrixAxes::OutputInput
            || embedding.rows != 248320
            || embedding.columns != geometry.width
            || input_gamma.source != source.source_identity()
            || input_gamma.kind != QuantKind::Bf16
            || input_gamma.shape != [geometry.width as u64]
            || input_gamma.bytes != geometry.width * 2
        {
            return Err(Error::Unsupported(
                "RMS actual leaf embedding/input gamma headers differ".into(),
            ));
        }
        let matrix = source.matrix(&embedding.name, embedding.axes)?;
        if matrix.source_identity() != embedding.source
            || matrix.rows() != embedding.rows
            || matrix.columns() != embedding.columns
            || matrix.tensor().kind != embedding.kind
        {
            return Err(Error::State(
                "RMS actual leaf embedding source differs".into(),
            ));
        }
        source
            .parameter(&input_gamma.name)?
            .validate_source_parameter(&input_gamma)?;
        let final_binding =
            BoundFinalRmsSource::bind(source.clone(), final_gamma, geometry, contract, metadata)?;
        Ok(Self(Arc::new(LeafOwner {
            source,
            embedding,
            input_gamma,
            final_binding,
            retained: None,
        })))
    }
    /// Retain original cold metadata without importing its family into the backend.
    pub fn retain_source(
        mut self,
        retained: Arc<crate::cuda::native::RetainedSource>,
    ) -> Result<Self> {
        if !self.source().same_issuer(retained.source()) {
            return Err(Error::State(
                "RMS retained original source issuer differs".into(),
            ));
        }
        let source = Arc::get_mut(&mut self.0)
            .ok_or_else(|| Error::State("unique cold RMS leaf source required".into()))?;
        if source.retained.is_some() {
            return Err(Error::State(
                "RMS cold source owner already retained".into(),
            ));
        }
        source.retained = Some(retained);
        Ok(self)
    }
    pub(crate) fn retained_source(&self) -> Result<&Arc<crate::cuda::native::RetainedSource>> {
        self.0
            .retained
            .as_ref()
            .ok_or_else(|| Error::State("RMS original cold source owner absent".into()))
    }
    pub(crate) fn completion_owner(&self) -> crate::physical_quarantine::CompletionOwner {
        self.0.clone()
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.0.source
    }
    pub(crate) fn final_binding(&self) -> &BoundFinalRmsSource {
        &self.0.final_binding
    }
    pub(super) fn columns(&self) -> Columns {
        if self.0.embedding.columns == 2048 {
            Columns::Qwen2b
        } else {
            Columns::Qwen08
        }
    }
    pub(crate) fn final_gamma_name(&self) -> &str {
        self.0.final_binding.gamma_name()
    }
    pub(crate) fn ordinary_members(&self) -> Result<(VerifiedMatrix, VerifiedParameter)> {
        self.require_source(self.source())?;
        Ok((
            self.source()
                .matrix(&self.0.embedding.name, self.0.embedding.axes)?,
            self.source().parameter(&self.0.input_gamma.name)?,
        ))
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !source.same_issuer(self.source()) {
            return Err(Error::State("RMS leaf actual source issuer differs".into()));
        }
        source
            .parameter(&self.0.input_gamma.name)?
            .validate_source_parameter(&self.0.input_gamma)?;
        self.0.final_binding.require_source(source)
    }
}
