//! SumF32, source gamma and both public outputs remain dedicated.
use super::*;
use crate::cuda::rms::pooled::{FullModelRmsScratch, ScratchRole};
impl FusedRmsPrepared {
    /// GammaRaw and original unrounded sum; public outputs are priced by ModelValues.
    pub(crate) fn text_private_layout(rows: usize, width: usize) -> Result<Vec<usize>> {
        let rows = FullModelRmsScratch::row_spec(rows, width)?;
        let columns = super::super::pooled::physical_columns(width)?;
        Ok(Buffer::owned(rows, columns)
            .into_iter()
            .filter(|(r, _)| {
                Self::scratch_role(*r).is_none()
                    && !matches!(r, Buffer::NormalizedBf16 | Buffer::ResidualBf16)
            })
            .map(|(_, bytes)| bytes)
            .collect())
    }
    pub(super) fn scratch_role(role: Buffer) -> Option<ScratchRole> {
        Some(match role {
            Buffer::GammaF32 => ScratchRole::GammaF32,
            Buffer::GammaShifted => ScratchRole::GammaShifted,
            Buffer::Squares => ScratchRole::Squares,
            Buffer::Mean => ScratchRole::Mean,
            Buffer::Epsilon => ScratchRole::Epsilon,
            Buffer::Inverse => ScratchRole::Inverse,
            Buffer::NormalizedF32 => ScratchRole::Normalized,
            Buffer::Weighted => ScratchRole::Weighted,
            Buffer::GammaRaw
            | Buffer::Attention
            | Buffer::PriorResidual
            | Buffer::SumF32
            | Buffer::NormalizedBf16
            | Buffer::ResidualBf16 => return None,
        })
    }
    pub(crate) async fn prepare_text_pooled(
        scope: Arc<FullModelRmsScratch>,
        node: usize,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let binding = scope.binding(node)?.clone();
        if !matches!(
            binding.operator().kind(),
            zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)
        ) {
            return Err(Error::State("pooled dual actual source site".into()));
        }
        let rows = scope.rows()?;
        let actor = scope.actor(node)?;
        let prepared = Self::prepare_rows_storage(
            scope.service().clone(),
            RmsSource::Model(binding.clone()),
            rows,
            branch,
            residual,
            (correlation, cancelled),
            Some(actor),
        )
        .await?;
        prepared.require_full_model_operator(binding.operator())?;
        Ok(prepared)
    }
    /// Fresh actual record over the SAME gamma, outputs and pooled private owners.
    pub(crate) fn bind_prompt_call(
        self: &Arc<Self>,
        scope: Arc<FullModelRmsScratch>,
        node: usize,
        branch: &ResidentLease,
        residual: &ResidentLease,
    ) -> Result<Arc<Self>> {
        self.service.ensure_usable()?;
        let binding = scope.binding(node)?.clone();
        let old = self
            .model_binding
            .as_ref()
            .or(self.owner._model_binding.as_ref())
            .ok_or_else(|| Error::State("text dual cold source binding absent".into()))?;
        let (a, b) = (old.operator(), binding.operator());
        binding.require_source(&self.owner.source)?;
        if !matches!(
            b.kind(),
            zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)
        ) || a.kind() != b.kind()
            || a.matrices() != b.matrices()
            || a.parameters() != b.parameters()
            || a.call().source() != b.call().source()
            || a.call().rows().len() != b.call().rows().len()
            || scope.rows()? != self.owner.rows
            || scope.columns() != self.owner.columns
        {
            return Err(Error::State(
                "text dual source/site/gamma/storage geometry differs".into(),
            ));
        }
        crate::resident_program::natural_binding::same_lease(
            self.lease(Buffer::Attention),
            branch,
        )?;
        crate::resident_program::natural_binding::same_lease(
            self.lease(Buffer::PriorResidual),
            residual,
        )?;
        let pooled = self
            .pooled
            .as_ref()
            .ok_or_else(|| Error::State("text dual cold pool actor absent".into()))?
            .rebind(&scope, node)?;
        let metadata = Arc::new(self.owner.source.reserve_workspace(64 * 1024)?);
        let result = Arc::new(Self {
            service: self.service.clone(),
            leases: self.leases.clone(),
            stages: self.stages.clone(),
            accesses: self.accesses.clone(),
            owner: self.owner.clone(),
            model_binding: Some(binding),
            pooled: Some(pooled),
            _text_binding: Some(metadata),
        });
        result.require_full_model_operator(scope.binding(node)?.operator())?;
        Ok(result)
    }
    /// Complete ten-stage actual access union, including dedicated source gamma
    /// and unrounded sum; no high-level-plan-only substitute.
    pub(crate) fn text_storage_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if self.pooled.is_none()
            || self.model_binding.as_ref().is_none_or(|b| {
                !matches!(
                    b.operator().kind(),
                    zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)
                )
            }) && self.owner._model_binding.as_ref().is_none_or(|b| {
                !matches!(
                    b.operator().kind(),
                    zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)
                )
            })
        {
            return Err(Error::State(
                "text dual actual pooled source binding absent".into(),
            ));
        }
        Ok(self.accesses.to_vec())
    }
    pub(crate) fn require_rms_scratch(
        &self,
        scope: &Arc<FullModelRmsScratch>,
        node: usize,
    ) -> Result<()> {
        self.pooled
            .as_ref()
            .ok_or_else(|| Error::State("dual RMS is dedicated".into()))?
            .require_scope(scope, node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_runtime::kernel::BufferIOAttr;
    use std::collections::BTreeSet;
    #[test]
    fn pooled_roles_are_first_written_inside_each_complete_actor() {
        for (rows, columns) in [
            (Rows::One, Columns::Qwen08),
            (Rows::One, Columns::Qwen2b),
            (Rows::FixedPrompt, Columns::Qwen08),
        ] {
            let mut written = BTreeSet::new();
            for stage in Stage::ORDER {
                let roles = Buffer::stage(stage);
                let io = if stage == Stage::InputWiden {
                    add_bindings(rows, columns)
                } else {
                    stage.buffers_for(rows, columns)
                };
                assert_eq!(roles.len(), io.len());
                for (role, (_, access)) in roles.iter().zip(io.iter()) {
                    if let Some(scratch) = FusedRmsPrepared::scratch_role(*role) {
                        if *access == BufferIOAttr::ReadOnly {
                            assert!(written.contains(&scratch), "{stage:?}/{role:?}");
                        }
                        if *access == BufferIOAttr::WriteOnly {
                            written.insert(scratch);
                        }
                    }
                }
            }
            assert_eq!(written.len(), 8);
        }
    }
    #[test]
    fn every_declared_late_readback_is_dedicated() {
        for role in [
            Buffer::GammaRaw,
            Buffer::SumF32,
            Buffer::NormalizedBf16,
            Buffer::ResidualBf16,
        ] {
            assert!(FusedRmsPrepared::scratch_role(role).is_none(), "{role:?}");
        }
    }
}

impl FusedRmsPrepared {
    pub(crate) fn text_source_host_bytes(rows: usize, width: usize) -> Result<usize> {
        FullModelRmsScratch::row_spec(rows, width)?;
        pooled_source_host_bytes(width)?
            .checked_add(pooled_metadata_bytes()?)
            .ok_or_else(|| Error::Limit("text dual retained metadata".into()))
    }
}

#[cfg(test)]
mod source_host_tests {
    use super::*;
    #[test]
    fn pooled_gamma_host_budget_is_independent_of_device_rows_and_full_diagnostic() {
        for h in [1024, 2048] {
            assert_eq!(
                pooled_source_host_bytes(h).unwrap(),
                3 * (h * 2) + 64 * 1024
            );
        }
        assert!(pooled_source_host_bytes(usize::MAX).is_err());
    }
}
