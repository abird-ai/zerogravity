//! Dedicated public/input owners plus the opt-in original private scratch.
use super::*;
use crate::cuda::rms::pooled::{FullModelRmsScratch, ScratchRole};
impl RmsPrepared {
    /// Dedicated non-public allocations outside the nine-role pool.
    pub(crate) fn text_private_layout(rows: usize, width: usize) -> Result<Vec<usize>> {
        let rows = FullModelRmsScratch::row_spec(rows, width)?;
        let columns = super::super::pooled::physical_columns(width)?;
        Ok(Buffer::layout(rows, columns)
            .into_iter()
            .filter(|(r, _)| {
                Self::scratch_role(*r).is_none()
                    && !matches!(r, Buffer::Embedding | Buffer::Bf16Output)
            })
            .map(|(_, bytes)| bytes)
            .collect())
    }
    pub(super) fn scratch_role(role: Buffer) -> Option<ScratchRole> {
        Some(match role {
            Buffer::GammaF32 => ScratchRole::GammaF32,
            Buffer::GammaShifted => ScratchRole::GammaShifted,
            Buffer::InputF32 => ScratchRole::InputF32,
            Buffer::Squares => ScratchRole::Squares,
            Buffer::Mean => ScratchRole::Mean,
            Buffer::Epsilon => ScratchRole::Epsilon,
            Buffer::Inverse => ScratchRole::Inverse,
            Buffer::Normalized => ScratchRole::Normalized,
            Buffer::Weighted => ScratchRole::Weighted,
            Buffer::GammaRaw | Buffer::Embedding | Buffer::Bf16Output => return None,
        })
    }
    pub(crate) async fn prepare_text_pooled(
        scope: Arc<FullModelRmsScratch>,
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        require_not_cancelled(cancelled)?;
        let binding = scope.binding(1)?;
        if !matches!(
            binding.operator().kind(),
            zerogravity_core::accelerator::program::SourceOperatorKind::Rms(_)
        ) {
            return Err(Error::State("pooled ordinary actual source site".into()));
        }
        let embedding = scope
            .record()
            .operator(0)?
            .ok_or_else(|| Error::State("pooled embedding absent".into()))?;
        let [member] = embedding.matrices() else {
            return Err(Error::State("pooled embedding member arity".into()));
        };
        let matrix = scope.record().source().matrix(&member.name, member.axes)?;
        let parameter = scope
            .record()
            .source()
            .parameter(&binding.parameter().name)?;
        parameter.validate_source_parameter(binding.parameter())?;
        let mut owner = source_owner_rows_storage_bound(
            scope.record().source().clone(),
            scope.record().call().rows().ids(),
            scope.columns(),
            scope.rows()?,
            Some(scope.definitions()),
            matrix,
            parameter,
        )?;
        Arc::get_mut(&mut owner)
            .ok_or_else(|| Error::State("fresh pooled owner".into()))?
            ._bindings = Some(scope.record().clone());
        let mut prepared = Self::prepare_owner_storage(
            scope.service().clone(),
            owner,
            correlation,
            cancelled,
            Some(scope.actor(1)?),
        )
        .await?;
        let rows = text::EmbeddingRows::from_cold(scope, prepared.owner.clone())?;
        let instance = Arc::get_mut(&mut prepared)
            .ok_or_else(|| Error::State("fresh pooled ordinary source instance".into()))?;
        instance.text = Some(rows);
        let embedding = instance.embedding().id();
        for (lease, access) in &mut instance.accesses {
            if lease.id() == embedding {
                *access = DeviceAccess::Write;
            }
        }
        Ok(prepared)
    }
    pub(crate) fn require_rms_scratch(
        &self,
        scope: &Arc<FullModelRmsScratch>,
        node: usize,
    ) -> Result<()> {
        self.pooled
            .as_ref()
            .ok_or_else(|| Error::State("ordinary RMS is dedicated".into()))?
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
                let io = stage.buffers_for(rows, columns);
                assert_eq!(roles.len(), io.len());
                for (role, (_, access)) in roles.iter().zip(io.iter()) {
                    if let Some(scratch) = RmsPrepared::scratch_role(*role) {
                        if *access == BufferIOAttr::ReadOnly {
                            assert!(written.contains(&scratch), "{stage:?}/{role:?}");
                        }
                        if *access == BufferIOAttr::WriteOnly {
                            written.insert(scratch);
                        }
                    }
                }
            }
            assert_eq!(written.len(), 9);
        }
    }
    #[test]
    fn every_declared_late_readback_is_dedicated() {
        for role in [Buffer::GammaRaw, Buffer::Embedding, Buffer::Bf16Output] {
            assert!(RmsPrepared::scratch_role(role).is_none(), "{role:?}");
        }
    }
}

impl RmsPrepared {
    pub(crate) fn text_source_host_bytes(rows: usize, width: usize) -> Result<usize> {
        FullModelRmsScratch::row_spec(rows, width)?;
        pooled_source_host_bytes(rows, width)?
            .checked_add(
                pooled_metadata_bytes()?
                    + 64 * 1024
                    + super::super::definitions::RmsDefinitions::metadata_bound()?,
            )
            .ok_or_else(|| Error::Limit("text ordinary retained metadata".into()))
    }
}

#[cfg(test)]
mod source_host_tests {
    use super::*;
    #[test]
    fn retained_source_budget_counts_payload_and_init_not_shared_scratch_or_d2h() {
        for h in [1024, 2048] {
            for rows in [1, 47, 57, 62, 67, 72] {
                assert_eq!(
                    pooled_source_host_bytes(rows, h).unwrap(),
                    3 * ((rows + 1) * h * 2) + 64 * 1024
                );
            }
        }
        assert!(pooled_source_host_bytes(usize::MAX, 2048).is_err());
        assert!(pooled_source_host_bytes(72, usize::MAX).is_err());
    }
}
