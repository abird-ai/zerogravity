//! Source-owned, provider-neutral preparation through the original loader.
use crate::{Error, Result, executable::ExecutableBundle};
use std::sync::Arc;
use zerogravity_core::{
    model::{ModelProgram, ModelRegion},
    provider::{ExecutionMode, PreparedExecutable, RegionExecutionProvider, SelectedMode},
    source::{ProgramSource, RegionSource},
};

/// No cancellation race: submitted preparation is awaited, never dropped to
/// service a cancellation flag. The provider owns completion/failure retirement.
pub async fn prepare<P: RegionExecutionProvider>(
    program: ModelProgram<<P::Executable as PreparedExecutable>::Source>,
    provider: &P,
) -> Result<ExecutableBundle<P::Executable>> {
    let expected = program.program().clone();
    let source = program.source().clone();
    let selected = provider.selected();
    let prepared = provider.prepare(program).await?;
    verify(prepared, source, expected, selected, None).await
}

/// No implicit whole-model fallback. The provider must finish initialization
/// or retirement before returning, exactly as for whole-program preparation.
pub async fn prepare_region<P: RegionExecutionProvider>(
    region: ModelRegion<<P::Executable as PreparedExecutable>::Source>,
    provider: &P,
) -> Result<ExecutableBundle<P::Executable>>
where
    <P::Executable as PreparedExecutable>::Source: RegionSource,
{
    let expected = region.program().clone();
    let source = region.source().clone();
    let scope = region.region().range();
    let selected = provider.selected();
    let prepared = provider.prepare_region(region).await?;
    verify(prepared, source, expected, selected, Some(scope)).await
}

async fn verify<E: PreparedExecutable>(
    prepared: E,
    source: E::Source,
    expected: Arc<zerogravity_core::semantic::ForwardPlan>,
    selected: SelectedMode,
    scope: Option<std::ops::Range<usize>>,
) -> Result<ExecutableBundle<E>> {
    if !source.same_issuer(&prepared.source().await?)
        || !Arc::ptr_eq(&expected, &prepared.program().await?)
        || prepared.selected()? != selected
        || match (scope, prepared.prepared_region(), prepared.prepared_text()) {
            // Older native tensor-source owners have their own typed layout.
            // Compiled preparation cannot omit the typed call/region contract.
            (None, None, None) => {
                expected.source_plan_geometry().is_some()
                    || matches!(selected.mode, ExecutionMode::Compiled(_))
            }
            (None, None, Some(text)) => {
                text.selected() != selected
                    || !source.same_issuer(text.model().source())
                    || !Arc::ptr_eq(&expected, text.model().program())
            }
            (Some(range), Some(region), None) => {
                region.model().region().range() != range
                    || region.selected() != selected
                    || !source.same_issuer(region.model().source())
                    || !Arc::ptr_eq(&expected, region.model().program())
            }
            _ => true,
        }
    {
        return Err(Error::State(
            "prepared provider replaced the original source/program/scope/selection".into(),
        ));
    }
    Ok(ExecutableBundle { prepared })
}
