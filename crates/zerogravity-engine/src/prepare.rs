//! Source-owned, provider-neutral preparation through the original loader.
use crate::{Error, Result, executable::ExecutableBundle};
use std::sync::Arc;
use zerogravity_core::{
    model::ModelProgram,
    provider::{PreparedExecutable, RegionExecutionProvider},
    source::ProgramSource,
};

/// No cancellation race: submitted preparation is awaited, never dropped to
/// service a cancellation flag. The provider owns completion/failure retirement.
pub async fn prepare<P: RegionExecutionProvider>(
    program: ModelProgram<<P::Executable as PreparedExecutable>::Source>,
    provider: &P,
) -> Result<ExecutableBundle<P::Executable>> {
    let expected = program.program().clone();
    let source = program.source().clone();
    let prepared = provider.prepare(program).await?;
    if !source.same_issuer(&prepared.source().await?)
        || !Arc::ptr_eq(&expected, &prepared.program().await?)
    {
        return Err(Error::State(
            "prepared provider replaced the original source/program".into(),
        ));
    }
    Ok(ExecutableBundle { prepared })
}
