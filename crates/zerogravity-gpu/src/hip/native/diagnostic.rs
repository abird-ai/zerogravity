//! Finite qualification orchestration around the shared compiler owner.
//! Ordinary preparation must not execute probe workloads.
use super::*;

impl ColdNativePreparation {
    pub(crate) async fn prepare_diagnostic(
        self,
        mut kernels: Vec<Box<dyn CubeKernel>>,
        catalogue_metadata: WindowWorkspace,
    ) -> Result<Arc<PreparedSourceNative>> {
        // Keep the original pre-probe issuer/extent check and catalogue order.
        self.require_catalogue_metadata(&catalogue_metadata)?;
        let service = self.service.clone();
        let source = self.source.clone();
        kernels.push(crate::source_float::qualification::kernel(service.client()));
        let owner = self.prepare(kernels, catalogue_metadata).await?;
        crate::source_float::qualification::run(&service, &source).await?;
        Ok(owner)
    }
}
