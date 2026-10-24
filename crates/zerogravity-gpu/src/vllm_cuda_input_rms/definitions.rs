//! Immutable source/ABI only; gamma, operands and current producers remain local.
use super::*;
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;

pub(super) struct RmsDefinitions {
    rows: Rows,
    columns: Columns,
    ordinary: Arc<[Spec]>,
    fused: Arc<[Spec]>,
    _metadata: WindowWorkspace,
}
impl RmsDefinitions {
    /// Eleven unique definitions; construction can hold three source copies.
    /// Exact headers/entry suffixes fit512B and physical ABI has at most4 slots.
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(super) fn metadata_bound() -> Result<usize> {
        let each = (SOURCE.len().max(fused::ADD_SOURCE.len()) + 512)
            .checked_mul(4)
            .and_then(|n| n.checked_add(PinnedKernel::KEY_METADATA_BYTES + 4 * 32 + 256))
            .ok_or_else(|| Error::Limit("RMS immutable definition extent".into()))?;
        each.checked_mul(11)
            .and_then(|n| {
                n.checked_add(20 * std::mem::size_of::<Spec>() + std::mem::size_of::<Self>() + 4096)
            })
            .ok_or_else(|| Error::Limit("RMS immutable definition extent".into()))
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    fn build_specs(rows: Rows, columns: Columns) -> Result<(Vec<Spec>, Vec<Spec>)> {
        if matches!(rows, Rows::CohortB4 { initial: true }) && columns != Columns::Vllm08 {
            return Err(Error::Unsupported(
                "B4 RMS definitions require original1024 columns".into(),
            ));
        }
        let ordinary = Stage::ORDER
            .into_iter()
            .map(|s| specification_for(s, rows, columns))
            .collect::<Result<Vec<_>>>()?;
        let mut dual = Vec::with_capacity(Stage::ORDER.len());
        for (stage, spec) in Stage::ORDER.into_iter().zip(&ordinary) {
            dual.push(if stage == Stage::InputWiden {
                fused::fused_specification(stage, rows, columns)?
            } else {
                spec.clone()
            });
        }
        Ok((ordinary, dual))
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn readiness_temporary_host_bytes(cohort: bool) -> Result<usize> {
        (64 * 1024usize)
            .checked_add(if cohort { Self::metadata_bound()? } else { 0 })
            .ok_or_else(|| Error::Limit("RMS readiness temporary metadata extent".into()))
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(super) fn prepare(
        source: &VerifiedTensorSource,
        rows: Rows,
        columns: Columns,
    ) -> Result<Arc<Self>> {
        let bound = Self::metadata_bound()?;
        let metadata = source.reserve_workspace(bound)?;
        let (ordinary, dual) = Self::build_specs(rows, columns)?;
        let mut retained = 20 * std::mem::size_of::<Spec>() + std::mem::size_of::<Self>() + 4096;
        for spec in ordinary.iter().chain(dual.iter().skip(2).take(1)) {
            retained = retained
                .checked_add(spec.kernel.retained_metadata_bytes()?)
                .ok_or_else(|| Error::Limit("RMS retained definitions".into()))?;
        }
        if retained > bound {
            return Err(Error::Limit(
                "RMS actual retained source/ABI/key exceeds admitted definitions".into(),
            ));
        }
        Ok(Arc::new(Self {
            rows,
            columns,
            ordinary: ordinary.into(),
            fused: dual.into(),
            _metadata: metadata,
        }))
    }
    pub(super) fn specs(&self, rows: Rows, columns: Columns, dual: bool) -> Result<Arc<[Spec]>> {
        if rows != self.rows || columns != self.columns {
            return Err(Error::State(
                "RMS immutable definitions differ from actual geometry".into(),
            ));
        }
        Ok(if dual {
            self.fused.clone()
        } else {
            self.ordinary.clone()
        })
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn kernels(&self) -> impl Iterator<Item = &PinnedKernel> {
        self.ordinary
            .iter()
            .map(|s| &s.kernel)
            .chain(self.fused.iter().skip(2).take(1).map(|s| &s.kernel))
    }
}

#[cfg(all(test, feature = "vllm-cuda-qualification", target_os = "linux"))]
mod tests {
    use super::*;
    use cubecl_runtime::kernel::KernelMetadata;
    #[test]
    fn compiler_context_pooled_rms_shares_nine_exact_keys_and_preserves_geometry() -> Result<()> {
        for (rows, columns) in [
            (Rows::One, Columns::Vllm08),
            (Rows::FortySeven, Columns::Vllm08),
            (Rows::One, Columns::Qwen2b),
            (
                Rows::ShortHf(zerogravity_core::accelerator::program::ShortHfWorkload::Prompt47),
                Columns::Qwen2b,
            ),
            (Rows::CohortB4 { initial: true }, Columns::Vllm08),
            (Rows::CohortB4 { initial: false }, Columns::Vllm08),
            (Rows::CohortB4 { initial: false }, Columns::Qwen2b),
        ] {
            let (ordinary, dual) = RmsDefinitions::build_specs(rows, columns)?;
            assert_eq!((ordinary.len(), dual.len()), (10, 10));
            let mut retained =
                20 * std::mem::size_of::<Spec>() + std::mem::size_of::<RmsDefinitions>() + 4096;
            for (index, (a, b)) in ordinary.iter().zip(&dual).enumerate() {
                assert_eq!(a.grid, b.grid);
                assert_eq!(a.kernel.id() == b.kernel.id(), index != 2);
                retained += a.kernel.retained_metadata_bytes()?;
            }
            retained += dual[2].kernel.retained_metadata_bytes()?;
            assert!(retained <= RmsDefinitions::metadata_bound()?);
        }
        assert!(
            RmsDefinitions::build_specs(Rows::CohortB4 { initial: true }, Columns::Qwen2b).is_err()
        );
        Ok(())
    }
    #[test]
    fn compiler_context_derived_rms_reserves_fund_backend_without_repricing_payloads() -> Result<()>
    {
        let ordinary = resident::pooled_metadata_bytes()?;
        let fused = fused::pooled_metadata_bytes()?;
        let definitions = RmsDefinitions::metadata_bound()?;
        let old = 2 * 1024 * 1024 + 10 * PinnedKernel::KEY_METADATA_BYTES;
        let savings = 6 * (old - ordinary + 48 * (old - fused) - definitions);
        let host = crate::physical_service::CUDA_COMPILER_HOST_BYTES as usize;
        let old_b4 = 4_285_247_927usize;
        let extra = RmsDefinitions::readiness_temporary_host_bytes(true)?;
        let new_b4 = old_b4 - savings + host + extra;
        let new_2b = 3_643_438_035usize - savings
            + host
            + RmsDefinitions::readiness_temporary_host_bytes(false)?;
        println!(
            "CUDA_READINESS_RMS_COST ordinary={ordinary} fused={fused} definitions={definitions} savings={savings} backend_host={host} anchored_b4_upper={new_b4} anchored_2b_upper={new_2b} readiness_metadata_peak={extra}"
        );
        assert!(ordinary < old && fused < old);
        assert!(savings > host);
        assert!(new_b4 <= 4096 * 1024 * 1024 && new_2b <= 4096 * 1024 * 1024);
        Ok(())
    }
}
