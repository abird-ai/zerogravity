//! Eight exact model-call templates over two physical layouts.
use super::*;
use crate::gpu::SourceBindings;
use zerogravity_artifacts::source_weights::WindowWorkspace;
pub(super) struct Templates {
    source: Arc<OriginalWindowModelSource>,
    calls: Vec<Arc<SourceBindings>>,
    _metadata: WindowWorkspace,
}
impl Templates {
    pub(super) fn prepare(source: Arc<OriginalWindowModelSource>) -> Result<Self> {
        let metadata = source.source().reserve_workspace(64 * 1024)?;
        let mut calls = Vec::with_capacity(8);
        for base in std::iter::once(0).chain(47..=53) {
            let ids: &[u32] = if base == 0 {
                OriginalWindowModelSource::fixed_prompt_ids()
            } else {
                &[0]
            };
            let prepared = source.prepare_fixed_prompt_call(base, ids)?;
            calls.push(super::super::bindings::prepare(&source, prepared, None)?);
        }
        Ok(Self {
            source,
            calls,
            _metadata: metadata,
        })
    }
    pub(super) fn physical_scopes(&self) -> [Arc<SourceBindings>; 2] {
        [self.calls[0].clone(), self.calls[1].clone()]
    }
    pub(super) fn bind(&self, base: usize, ids: &[u32]) -> Result<Arc<SourceBindings>> {
        let index = match base {
            0 => 0,
            47..=53 => base - 46,
            _ => {
                return Err(Error::Unsupported(
                    "fixed prompt semantic frontier absent".into(),
                ));
            }
        };
        let template = &self.calls[index];
        let bytes = template
            .rebind_metadata_bytes()?
            .checked_add(4096)
            .ok_or_else(|| Error::Limit("fixed prompt call metadata extent".into()))?;
        let metadata = self.source.source().reserve_workspace(bytes)?;
        let call = self.source.bind_fixed_prompt_call(base, ids)?;
        Ok(Arc::new(
            template.rebind(call.request().source_call()?, metadata)?,
        ))
    }
}
